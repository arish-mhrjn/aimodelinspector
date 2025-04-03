from typing import Dict, Any, Tuple, List, Optional, Set
import logging
import os
import re
import json
import struct
from pathlib import Path

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer
from ..sandbox import Sandbox
from ..models.permissions import Permission


class TensorRTAnalyzer(BaseAnalyzer):
    """
    Analyzer for TensorRT engine files.

    TensorRT is NVIDIA's platform for high-performance deep learning inference,
    and this analyzer can identify various TensorRT model types and architectures.
    """

    # TensorRT file signatures
    TRT_MAGIC = bytes.fromhex('6a8b4567')  # 32-bit int 0x6a8b4567

    # Common network architectures
    MODEL_ARCHITECTURES = {
        'resnet': 'ResNet',
        'efficientnet': 'EfficientNet',
        'densenet': 'DenseNet',
        'inception': 'Inception',
        'mobilenet': 'MobileNet',
        'ssd': 'SSD',
        'faster-rcnn': 'FasterRCNN',
        'yolo': 'YOLO',
        'bert': 'BERT',
        'transformer': 'Transformer',
        'unet': 'UNet',
    }

    def __init__(self):
        """Initialize the TensorRT analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.engine', '.plan', '.trt'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        TensorRT files are generally safe to analyze without executing code.

        Args:
            file_path: Path to the file

        Returns:
            True as TensorRT files are generally safe
        """
        return True

    def analyze(
            self,
            file_path: str,
            sandbox: Optional[Sandbox] = None
    ) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a TensorRT model file.

        Args:
            file_path: Path to the TensorRT file
            sandbox: Optional sandbox for safety (not required for TensorRT)

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Ensure we have read permission if sandbox provided
            if sandbox:
                sandbox.check_format_permission('.trt', Permission.READ_FILE)

            # Extract metadata
            metadata = self._extract_tensorrt_metadata(file_path)

            # Determine model type
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing TensorRT file {file_path}: {e}")
            raise

    def _extract_tensorrt_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a TensorRT file.

        Args:
            file_path: Path to the TensorRT file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'tensorrt'
        }

        try:
            # Try to use TensorRT if available
            import tensorrt as trt

            # Create a logger
            trt_logger = trt.Logger(trt.Logger.ERROR)

            # Create a runtime
            runtime = trt.Runtime(trt_logger)

            # Load the engine
            with open(file_path, 'rb') as f:
                engine_data = f.read()

            engine = runtime.deserialize_cuda_engine(engine_data)

            # Extract basic information
            metadata['num_layers'] = engine.num_layers
            metadata['max_batch_size'] = engine.max_batch_size
            metadata['max_workspace_size'] = engine.max_workspace_size
            metadata['has_implicit_batch_dimension'] = engine.has_implicit_batch_dimension

            # Extract input/output info
            metadata['inputs'] = []
            metadata['outputs'] = []

            for i in range(engine.num_bindings):
                binding_name = engine.get_binding_name(i)
                binding_shape = tuple(engine.get_binding_shape(i))
                binding_dtype = engine.get_binding_dtype(i)

                binding_info = {
                    'name': binding_name,
                    'shape': binding_shape,
                    'dtype': str(binding_dtype)
                }

                if engine.binding_is_input(i):
                    metadata['inputs'].append(binding_info)
                else:
                    metadata['outputs'].append(binding_info)

        except ImportError:
            # Fallback to basic file analysis if TensorRT not available
            metadata.update(self._basic_tensorrt_analysis(file_path))

        return metadata

    def _basic_tensorrt_analysis(self, file_path: str) -> Dict[str, Any]:
        """
        Perform a basic analysis of TensorRT file without using the TensorRT library.

        Args:
            file_path: Path to the TensorRT file

        Returns:
            Metadata dictionary
        """
        result = {}

        try:
            with open(file_path, 'rb') as f:
                # Check for TensorRT magic number
                magic = f.read(4)
                if magic == self.TRT_MAGIC:
                    result['is_valid_trt'] = True
                else:
                    result['is_valid_trt'] = False

                # Read a chunk to look for architecture hints
                f.seek(0)
                chunk = f.read(16384)  # Read a 16KB chunk

                # Check for architecture identifiers
                for arch_name, model_name in self.MODEL_ARCHITECTURES.items():
                    if arch_name.encode() in chunk:
                        result['architecture_hint'] = model_name
                        break

                # Look for CUDA version identifiers
                cuda_version_match = re.search(rb'CUDA (\d+\.\d+)', chunk)
                if cuda_version_match:
                    result['cuda_version'] = cuda_version_match.group(1).decode('utf-8')

                # Look for TensorRT version identifiers
                trt_version_match = re.search(rb'TensorRT-(\d+\.\d+\.\d+)', chunk)
                if trt_version_match:
                    result['tensorrt_version'] = trt_version_match.group(1).decode('utf-8')

        except Exception as e:
            self.logger.warning(f"Error in basic TensorRT analysis: {e}")

        return result

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check for architecture hint from basic analysis
        if 'architecture_hint' in metadata:
            return f"TensorRT-{metadata['architecture_hint']}", ModelConfidence.MEDIUM

        # If we have input info, we can try to guess the model type
        if 'inputs' in metadata and metadata['inputs']:
            inputs = metadata['inputs']

            # Check for specific input shapes that suggest model types
            for input_info in inputs:
                shape = input_info.get('shape')
                if shape and len(shape) == 4:
                    # Likely a vision model with NCHW or NHWC format
                    return 'TensorRT-VisionModel', ModelConfidence.MEDIUM

            # Check number of inputs/outputs
            if len(metadata['inputs']) == 1 and len(metadata['outputs']) >= 1:
                if 'outputs' in metadata and len(metadata['outputs']) > 10:
                    # Many outputs often indicates a detection model
                    return 'TensorRT-DetectionModel', ModelConfidence.LOW
                else:
                    # Single input, few outputs often indicates a classification model
                    return 'TensorRT-ClassificationModel', ModelConfidence.LOW

        # Check layer count if available
        if 'num_layers' in metadata:
            if metadata['num_layers'] > 100:
                return 'TensorRT-DeepModel', ModelConfidence.LOW

        # Default to generic TensorRT model
        return 'TensorRT', ModelConfidence.LOW

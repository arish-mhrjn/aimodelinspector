from typing import Dict, Any, Tuple, List, Optional, Set
import logging
import os
import json
import struct
import re
from pathlib import Path
from collections import Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class ORTAnalyzer(BaseAnalyzer):
    """
    Analyzer for ONNX Runtime (.ort) model files.

    ONNX Runtime is a performance-focused inference engine for ONNX models,
    and .ort files are optimized serialized models for the ONNX Runtime.
    This analyzer can identify model types from .ort files.

    Potential improvements:
    1. Add more specific detection patterns for newer model architectures
    2. Implement extraction of ONNX Runtime specific optimizations that were applied
    3. Add detection of quantization levels and optimization settings
    4. Improve model topology analysis to better identify complex architectures
    5. Add capability to extract runtime configuration parameters
    6. Implement validation of model compatibility with different hardware targets
       (CPU, GPU, NPU, etc.) based on operator usage
    """

    # Common operator patterns for model types - similar to ONNX but aware of ORT optimizations
    MODEL_PATTERNS = {
        # Vision model patterns
        'Conv': ('VisionModel', ModelConfidence.LOW),
        'FusedConv': ('OptimizedVisionModel', ModelConfidence.MEDIUM),
        'ConvTranspose': ('ImageGenerator', ModelConfidence.LOW),
        'MaxPool': ('VisionModel', ModelConfidence.LOW),
        'BatchNormalization': ('NeuralNetwork', ModelConfidence.LOW),

        # Language model patterns
        'Attention': ('LanguageModel', ModelConfidence.MEDIUM),
        'FusedAttention': ('OptimizedLanguageModel', ModelConfidence.HIGH),
        'Embedding': ('LanguageModel', ModelConfidence.LOW),
        'LSTM': ('RNN', ModelConfidence.MEDIUM),
        'GRU': ('RNN', ModelConfidence.MEDIUM),

        # Specific architecture patterns
        'Gemm': ('LinearModel', ModelConfidence.LOW),
        'FusedMatMul': ('OptimizedLinearModel', ModelConfidence.MEDIUM),
        'Softmax': ('Classifier', ModelConfidence.LOW),
        'ReduceMean': ('StatisticalModel', ModelConfidence.LOW),
    }

    # Model naming patterns
    NAME_PATTERNS = {
        r'vgg': ('VGG', ModelConfidence.HIGH),
        r'resnet': ('ResNet', ModelConfidence.HIGH),
        r'inception': ('Inception', ModelConfidence.HIGH),
        r'mobilenet': ('MobileNet', ModelConfidence.HIGH),
        r'efficientnet': ('EfficientNet', ModelConfidence.HIGH),
        r'bert': ('BERT', ModelConfidence.HIGH),
        r'gpt': ('GPT', ModelConfidence.HIGH),
        r'yolo': ('YOLO', ModelConfidence.HIGH),
        r'ssd': ('SSD', ModelConfidence.HIGH),
        r'faster_rcnn': ('FasterRCNN', ModelConfidence.HIGH),
        r'mask_rcnn': ('MaskRCNN', ModelConfidence.HIGH),
        r'unet': ('UNet', ModelConfidence.HIGH),
        r'llm': ('LargeLanguageModel', ModelConfidence.HIGH),
        r'llama': ('LlamaModel', ModelConfidence.HIGH),
        r'whisper': ('WhisperModel', ModelConfidence.HIGH),
    }

    def __init__(self):
        """Initialize the ONNX Runtime analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.ort'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        ORT files are generally safe to analyze as they don't contain executable code.

        Args:
            file_path: Path to the file

        Returns:
            True as ORT files are safe to analyze
        """
        # ORT files are generally safe
        return True

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze an ONNX Runtime model file.

        Args:
            file_path: Path to the ONNX Runtime model file

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
            # Extract metadata from file
            metadata = self._extract_ort_metadata(file_path)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing ORT file {file_path}: {e}")
            raise

    def _extract_ort_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from an ONNX Runtime file.

        Args:
            file_path: Path to the ORT file

        Returns:
            Metadata dictionary
        """
        try:
            # Try to use onnxruntime library if available
            import onnxruntime as ort

            # Load the model - ORT files typically contain the model in optimized form
            sess_options = ort.SessionOptions()
            # Don't actually run the model, just load it
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            # Create session but avoid actual inference
            session = ort.InferenceSession(file_path, sess_options)

            metadata = {
                'format': 'ort',
                'file_size': os.path.getsize(file_path),
                'provider_options': session.get_providers(),
            }

            # Extract model metadata if available
            model_metadata = session.get_modelmeta()
            if model_metadata:
                metadata.update({
                    'producer_name': model_metadata.producer_name,
                    'graph_name': model_metadata.graph_name,
                    'domain': model_metadata.domain,
                    'description': model_metadata.description,
                    'graph_description': model_metadata.graph_description,
                    'version': model_metadata.version,
                    'custom_metadata_map': model_metadata.custom_metadata_map,
                })

            # Extract input and output information
            metadata['inputs'] = []
            for input_info in session.get_inputs():
                metadata['inputs'].append({
                    'name': input_info.name,
                    'shape': list(input_info.shape),
                    'type': input_info.type,
                })

            metadata['outputs'] = []
            for output_info in session.get_outputs():
                metadata['outputs'].append({
                    'name': output_info.name,
                    'shape': list(output_info.shape),
                    'type': output_info.type,
                })

            # Try to extract node information if possible
            metadata['operator_counts'] = {}
            try:
                # This is a bit of a hack as ORT doesn't directly expose nodes
                # Some versions support this, others don't
                nodes = session._model_meta.nodes if hasattr(session, '_model_meta') else []
                op_types = [node.op_type for node in nodes]
                op_counts = Counter(op_types)
                metadata['operator_counts'] = dict(op_counts)
            except (AttributeError, Exception):
                # Fallback if node information is not available
                pass

            return metadata

        except ImportError:
            # Fallback to basic file analysis if onnxruntime not available
            return self._parse_ort_header(file_path)

    def _parse_ort_header(self, file_path: str) -> Dict[str, Any]:
        """
        Parse basic ONNX Runtime header information without the onnxruntime library.

        Args:
            file_path: Path to the ORT file

        Returns:
            Basic metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'ort'
        }

        # ORT files have a specific header format
        try:
            with open(file_path, 'rb') as f:
                # Read the first 1KB to check for identifiable information
                header_data = f.read(1024)

                # Check for ORT magic number (may vary by version)
                if b'ORTM' in header_data:
                    metadata['format_identified'] = True

                # Look for strings that might indicate model info
                producer_match = re.search(b'onnxruntime', header_data, re.IGNORECASE)
                if producer_match:
                    metadata['producer_identified'] = 'onnxruntime'

                # Attempt to find version info
                version_match = re.search(rb'version\s*[:=]\s*([0-9.]+)', header_data)
                if version_match:
                    version_str = version_match.group(1).decode('utf-8', errors='ignore')
                    metadata['version_string'] = version_str

        except Exception as e:
            self.logger.warning(f"Error parsing ORT header: {e}")

        return metadata

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check for model name in metadata
        description = metadata.get('description', '')
        graph_name = metadata.get('graph_name', '')

        # Check custom metadata for clues about model type
        custom_metadata = metadata.get('custom_metadata_map', {})
        model_type_from_metadata = custom_metadata.get('model_type', '')

        # Combine strings that might contain model architecture information
        name_sources = [description, graph_name, model_type_from_metadata]

        # Check against name patterns
        for name in name_sources:
            if not name:
                continue

            name_lower = name.lower()
            for pattern, (model_type, confidence) in self.NAME_PATTERNS.items():
                if re.search(pattern, name_lower):
                    return model_type, confidence

        # Check operator distribution if available
        if metadata.get('operator_counts'):
            op_counts = metadata['operator_counts']

            # Look for fused operators that indicate optimized models
            if any(op.startswith('Fused') for op in op_counts.keys()):
                if 'FusedConv' in op_counts or 'FusedConvolution' in op_counts:
                    return 'OptimizedVisionModel', ModelConfidence.MEDIUM
                if 'FusedAttention' in op_counts:
                    return 'OptimizedTransformerModel', ModelConfidence.MEDIUM

            # Standard operator checks
            conv_count = op_counts.get('Conv', 0) + op_counts.get('ConvTranspose', 0)
            if conv_count > 5:
                return 'VisionModel', ModelConfidence.MEDIUM

            attention_count = op_counts.get('Attention', 0) + op_counts.get('MultiHeadAttention', 0)
            if attention_count > 0:
                return 'TransformerModel', ModelConfidence.MEDIUM

            if 'LSTM' in op_counts or 'GRU' in op_counts:
                return 'RNNModel', ModelConfidence.MEDIUM

        # Check input/output shapes for clues
        if metadata.get('inputs'):
            inputs = metadata['inputs']

            # Image models typically have 4D input (batch, channels, height, width)
            if inputs and len(inputs) >= 1:
                input_shape = inputs[0].get('shape', [])

                # Check if shape implies an image model
                if len(input_shape) == 4:
                    # NCHW format common in vision models
                    return 'VisionModel', ModelConfidence.MEDIUM

                # Check if shape implies a text/sequence model
                if len(input_shape) == 2 or len(input_shape) == 3:
                    # Many language models have 2D (batch, sequence) or
                    # 3D (batch, sequence, features) inputs
                    return 'LanguageModel', ModelConfidence.LOW

        # Check for quantized models
        if 'quantized' in str(metadata).lower() or any('Int8' in str(i) for i in metadata.get('inputs', [])):
            return 'QuantizedModel', ModelConfidence.LOW

        # Last resort: check providers for hints about target hardware
        providers = metadata.get('provider_options', [])
        if providers:
            if any('TensorRT' in p for p in providers):
                return 'GPUOptimizedModel', ModelConfidence.LOW
            if any('CUDA' in p for p in providers):
                return 'GPUModel', ModelConfidence.LOW
            if any('OpenVINO' in p for p in providers):
                return 'IntelOptimizedModel', ModelConfidence.LOW

        return 'ONNXRuntimeModel', ModelConfidence.LOW

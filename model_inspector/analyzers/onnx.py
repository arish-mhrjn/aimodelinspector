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


class ONNXAnalyzer(BaseAnalyzer):
    """
    Analyzer for ONNX model files.

    This analyzer can identify various model types stored in the ONNX format,
    including computer vision models, language models, and custom architectures.
    """

    # Common operator patterns for model types
    MODEL_PATTERNS = {
        # Vision model patterns
        'Conv': ('VisionModel', ModelConfidence.LOW),
        'ConvTranspose': ('ImageGenerator', ModelConfidence.LOW),
        'MaxPool': ('VisionModel', ModelConfidence.LOW),
        'BatchNormalization': ('NeuralNetwork', ModelConfidence.LOW),

        # Language model patterns
        'Attention': ('LanguageModel', ModelConfidence.MEDIUM),
        'Embedding': ('LanguageModel', ModelConfidence.LOW),
        'LSTM': ('RNN', ModelConfidence.MEDIUM),
        'GRU': ('RNN', ModelConfidence.MEDIUM),

        # Specific architecture patterns
        'Gemm': ('LinearModel', ModelConfidence.LOW),
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
    }

    def __init__(self):
        """Initialize the ONNX analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.onnx'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        ONNX files are generally safe to analyze as they don't contain executable code.

        Args:
            file_path: Path to the file

        Returns:
            True as ONNX files are safe to analyze
        """
        # ONNX files are generally safe
        return True

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze an ONNX model file.

        Args:
            file_path: Path to the ONNX model file

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
            metadata = self._extract_onnx_metadata(file_path)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing ONNX file {file_path}: {e}")
            raise

    def _extract_onnx_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from an ONNX file.

        Args:
            file_path: Path to the ONNX file

        Returns:
            Metadata dictionary
        """
        try:
            # Try to use onnx library if available
            import onnx
            model = onnx.load(file_path, load_external_data=False)

            metadata = {
                'ir_version': model.ir_version,
                'producer_name': model.producer_name,
                'producer_version': model.producer_version,
                'domain': model.domain,
                'model_version': model.model_version,
                'doc_string': model.doc_string,
            }

            # Extract graph metadata
            graph = model.graph
            metadata['graph'] = {
                'name': graph.name,
                'doc_string': graph.doc_string,
                'node_count': len(graph.node),
                'input_count': len(graph.input),
                'output_count': len(graph.output),
                'initializer_count': len(graph.initializer),
            }

            # Extract operator types
            op_types = [node.op_type for node in graph.node]
            op_counts = Counter(op_types)
            metadata['operator_counts'] = dict(op_counts)

            # Extract input and output shapes
            metadata['inputs'] = []
            for input_info in graph.input:
                input_shape = []
                if input_info.type.tensor_type.shape:
                    for dim in input_info.type.tensor_type.shape.dim:
                        if dim.dim_value:
                            input_shape.append(dim.dim_value)
                        else:
                            input_shape.append('dynamic')

                metadata['inputs'].append({
                    'name': input_info.name,
                    'shape': input_shape,
                    'data_type': input_info.type.tensor_type.elem_type
                })

            metadata['outputs'] = []
            for output_info in graph.output:
                output_shape = []
                if output_info.type.tensor_type.shape:
                    for dim in output_info.type.tensor_type.shape.dim:
                        if dim.dim_value:
                            output_shape.append(dim.dim_value)
                        else:
                            output_shape.append('dynamic')

                metadata['outputs'].append({
                    'name': output_info.name,
                    'shape': output_shape,
                    'data_type': output_info.type.tensor_type.elem_type
                })

            return metadata

        except ImportError:
            # Fallback to basic header parsing if onnx not available
            return self._parse_onnx_header(file_path)

    def _parse_onnx_header(self, file_path: str) -> Dict[str, Any]:
        """
        Parse basic ONNX header information without the onnx library.

        Args:
            file_path: Path to the ONNX file

        Returns:
            Basic metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'onnx'
        }

        # ONNX files use Protocol Buffers format
        # This is a basic implementation that doesn't fully parse the format
        with open(file_path, 'rb') as f:
            # Read first 1KB to check headers
            header = f.read(1024)

            # Look for producer info in the binary data
            producer_match = re.search(b'producer_name\x00([^\x00]{1,50})', header)
            if producer_match:
                metadata['producer_name'] = producer_match.group(1).decode('utf-8', errors='ignore')

            # Look for model domain
            domain_match = re.search(b'domain\x00([^\x00]{1,50})', header)
            if domain_match:
                metadata['domain'] = domain_match.group(1).decode('utf-8', errors='ignore')

        return metadata

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check for model name in doc string or graph name
        doc_string = metadata.get('doc_string', '')
        graph_name = metadata.get('graph', {}).get('name', '')

        # Check against name patterns
        for name in [doc_string, graph_name]:
            name_lower = name.lower()
            for pattern, (model_type, confidence) in self.NAME_PATTERNS.items():
                if re.search(pattern, name_lower):
                    return model_type, confidence

        # Check operator distribution
        if 'operator_counts' in metadata:
            op_counts = metadata['operator_counts']

            # If there are many convolutions, it's likely a vision model
            conv_count = op_counts.get('Conv', 0) + op_counts.get('ConvTranspose', 0)
            if conv_count > 10:
                return 'VisionModel', ModelConfidence.MEDIUM

            # Check for attention mechanisms indicating a transformer
            if 'Attention' in op_counts or 'MatMul' in op_counts and 'Softmax' in op_counts:
                return 'TransformerModel', ModelConfidence.MEDIUM

            # Check for RNN cells
            if 'LSTM' in op_counts or 'GRU' in op_counts:
                return 'RNNModel', ModelConfidence.MEDIUM

        # Check input/output shapes for clues
        if 'inputs' in metadata and metadata['inputs']:
            # Image models typically have 4D input (batch, channels, height, width)
            input_shape = metadata['inputs'][0].get('shape', [])
            if len(input_shape) == 4:
                # Check if middle dimensions look like image dimensions
                if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                    if input_shape[2] in [224, 256, 299, 320, 384, 512]:
                        return 'ImageClassifier', ModelConfidence.MEDIUM
                    return 'VisionModel', ModelConfidence.LOW

            # Text models might have 2D or 3D inputs (batch, sequence_length[, embedding_dim])
            if len(input_shape) == 2 or len(input_shape) == 3:
                return 'SequenceModel', ModelConfidence.LOW

        # If we can't determine the type, use the producer as a hint
        producer = metadata.get('producer_name', '')
        if 'tensorflow' in producer.lower():
            return 'TensorFlowModel', ModelConfidence.LOW
        if 'pytorch' in producer.lower():
            return 'PyTorchModel', ModelConfidence.LOW
        if 'onnxruntime' in producer.lower():
            return 'ONNXModel', ModelConfidence.LOW

        return 'Unknown', ModelConfidence.UNKNOWN

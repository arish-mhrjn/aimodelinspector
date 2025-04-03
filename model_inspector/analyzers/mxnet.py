# mxnet.py
"""
MXNet model analyzer module for analyzing MXNet params files.

This module analyzes model files in MXNet format (.params) to extract metadata
about the model architecture, parameters, and structure. It supports inspection
of parameter shape, size, data type, and additional model metadata.

Potential improvements:
1. Add support for MXNet symbol files (.json) to extract more detailed architecture information
2. Improve model architecture detection by recognizing common patterns in layer naming
3. Add specialized handling for popular architectures (ResNet, DenseNet, etc.)
4. Implement tensor data analysis to detect quantization or precision patterns
5. Extract training metadata from the model if available
6. Support for hybrid MXNet models (combining symbol and params)
7. Add statistical analysis of weight distributions
"""

from typing import Dict, Any, Tuple, Optional, List, Set
import struct
import logging
import json
from pathlib import Path
import numpy as np
import re
from collections import defaultdict, Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class MXNetAnalyzer(BaseAnalyzer):
    """
    Analyzer for MXNet .params model files.

    This analyzer extracts information from MXNet parameter files, which store
    the weights and biases of neural networks. It can determine model architecture,
    layer structure, and parameter statistics for MXNet models.
    """

    # Known model architecture patterns
    MODEL_ARCHITECTURES = {
        'resnet': 'ResNet',
        'densenet': 'DenseNet',
        'vgg': 'VGG',
        'inception': 'Inception',
        'mobilenet': 'MobileNet',
        'squeezenet': 'SqueezeNet',
        'alexnet': 'AlexNet',
        'bert': 'BERT',
        'gluon': 'Gluon',
        'lstm': 'LSTM',
        'gru': 'GRU',
        'fcn': 'FCN',
        'yolo': 'YOLO',
        'faster_rcnn': 'Faster R-CNN',
        'mask_rcnn': 'Mask R-CNN',
    }

    def __init__(self):
        """Initialize the MXNet analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.params'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze an MXNet params file to determine its model type and metadata.

        Args:
            file_path: Path to the MXNet params file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid MXNet params file
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            metadata = self._parse_mxnet_params(file_path)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing MXNet params file {file_path}: {e}")
            raise

    def _parse_mxnet_params(self, file_path: str) -> Dict[str, Any]:
        """
        Parse an MXNet params file to extract metadata.

        Args:
            file_path: Path to the MXNet params file

        Returns:
            Extracted metadata

        Raises:
            ValueError: If the file is not a valid MXNet params file
        """
        metadata = {
            "format": "mxnet_params",
            "parameters": {},
            "layer_counts": {},
            "total_params": 0,
            "total_size_bytes": 0,
            "data_types": {},
        }

        try:
            # Try to load the params file using ndarray
            import mxnet as mx
            params = mx.nd.load(file_path)

            # Extract parameter information
            for key, value in params.items():
                shape = value.shape
                dtype = str(value.dtype)
                size = np.prod(shape)

                metadata["parameters"][key] = {
                    "shape": shape,
                    "size": size,
                    "dtype": dtype,
                }

                metadata["total_params"] += size
                metadata["total_size_bytes"] += size * value.dtype.itemsize

                # Count layer types based on parameter naming
                layer_type = self._get_layer_type(key)
                if layer_type:
                    metadata["layer_counts"][layer_type] = metadata["layer_counts"].get(layer_type, 0) + 1

                # Track data types
                if dtype in metadata["data_types"]:
                    metadata["data_types"][dtype] += 1
                else:
                    metadata["data_types"][dtype] = 1

            # Add summary statistics
            metadata["parameter_count"] = len(metadata["parameters"])
            metadata["size_mb"] = metadata["total_size_bytes"] / (1024 * 1024)

            # Attempt to load symbol file if it exists (same name but .json extension)
            symbol_file = str(Path(file_path).with_suffix('.json'))
            if Path(symbol_file).exists():
                try:
                    with open(symbol_file, 'r') as f:
                        symbol_data = json.load(f)
                    metadata["has_symbol_file"] = True
                    metadata["symbol_data"] = self._extract_symbol_data(symbol_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse symbol file {symbol_file}: {e}")
                    metadata["has_symbol_file"] = False
            else:
                metadata["has_symbol_file"] = False

        except ImportError:
            # Fallback to parsing without MXNet
            self.logger.warning("MXNet not available, using limited file analysis")

            file_size = Path(file_path).stat().st_size
            metadata["file_size_bytes"] = file_size
            metadata["size_mb"] = file_size / (1024 * 1024)
            metadata["limited_analysis"] = True

            # Try to detect if it's an MXNet file by checking file header
            with open(file_path, 'rb') as f:
                header = f.read(10)

                # MXNet NDArray files typically start with a serialization header
                # This is a basic check and may not work for all versions
                if len(header) >= 4 and header[:4] in [b'\x12\x01\x00\x00', b'\x01\x00\x00\x00']:
                    metadata["is_valid_mxnet_file"] = True
                else:
                    metadata["is_valid_mxnet_file"] = False
                    raise ValueError("File does not appear to be a valid MXNet params file")

        return metadata

    def _get_layer_type(self, param_name: str) -> Optional[str]:
        """
        Determine the layer type from parameter name.

        Args:
            param_name: The parameter name from the params file

        Returns:
            Layer type or None if can't be determined
        """
        # Common patterns in MXNet parameter names
        if 'conv' in param_name.lower():
            return 'conv'
        elif 'bn' in param_name.lower() or 'batchnorm' in param_name.lower():
            return 'batchnorm'
        elif 'fc' in param_name.lower() or 'dense' in param_name.lower() or 'fullyconnected' in param_name.lower():
            return 'dense'
        elif 'lstm' in param_name.lower():
            return 'lstm'
        elif 'gru' in param_name.lower():
            return 'gru'
        elif 'rnn' in param_name.lower():
            return 'rnn'
        elif 'pool' in param_name.lower():
            return 'pooling'
        elif 'embed' in param_name.lower():
            return 'embedding'
        elif 'bias' in param_name.lower():
            return 'bias'
        elif 'weight' in param_name.lower():
            return 'weight'
        elif 'gamma' in param_name.lower() or 'beta' in param_name.lower():
            return 'normalization'
        else:
            return None

    def _extract_symbol_data(self, symbol_data: Dict) -> Dict[str, Any]:
        """
        Extract useful information from symbol JSON data.

        Args:
            symbol_data: Loaded JSON data from a symbol file

        Returns:
            Dictionary with extracted symbol information
        """
        extracted = {}

        if 'nodes' in symbol_data:
            nodes = symbol_data['nodes']
            extracted['node_count'] = len(nodes)

            # Count operators by type
            op_counts = Counter(node.get('op', 'unknown') for node in nodes)
            extracted['operator_counts'] = dict(op_counts)

            # Extract input and output shapes if available
            for node in nodes:
                if 'attr' in node and 'shape' in node['attr']:
                    try:
                        shape_data = json.loads(node['attr']['shape'])
                        if 'name' in node:
                            name = node['name']
                            extracted.setdefault('shapes', {})[name] = shape_data
                    except:
                        pass

        return extracted

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check if we found layer counts
        if 'layer_counts' in metadata and metadata['layer_counts']:
            # If symbol file was parsed successfully, we can be more confident
            if metadata.get('has_symbol_file', False) and 'symbol_data' in metadata:
                # Look for architecture clues in the symbol data
                symbol_data = metadata['symbol_data']
                if 'operator_counts' in symbol_data:
                    op_counts = symbol_data['operator_counts']

                    # Detect architecture based on operator patterns
                    if 'Convolution' in op_counts and 'Pooling' in op_counts:
                        # Examine parameter names for architecture hints
                        param_names = ' '.join(metadata['parameters'].keys()).lower()

                        for key, name in self.MODEL_ARCHITECTURES.items():
                            if key in param_names:
                                # Try to extract version number
                                if key == 'resnet':
                                    match = re.search(r'resnet(\d+)', param_names)
                                    if match:
                                        return f"{name}-{match.group(1)}", ModelConfidence.HIGH

                                return name, ModelConfidence.HIGH

                        # If there are a lot of convolution layers, it's likely a CNN
                        if op_counts.get('Convolution', 0) > 5:
                            return "CNN", ModelConfidence.MEDIUM

                    # RNN/LSTM detection
                    if 'RNN' in op_counts or 'LSTM' in op_counts or 'GRU' in op_counts:
                        if 'LSTM' in op_counts:
                            return "LSTM", ModelConfidence.HIGH
                        elif 'GRU' in op_counts:
                            return "GRU", ModelConfidence.HIGH
                        else:
                            return "RNN", ModelConfidence.HIGH

                    # Transformer detection
                    if 'MultiHeadAttention' in op_counts or 'LayerNorm' in op_counts:
                        return "Transformer", ModelConfidence.MEDIUM

            # No symbol file, use parameter naming
            param_names = ' '.join(metadata['parameters'].keys()).lower()

            for key, name in self.MODEL_ARCHITECTURES.items():
                if key in param_names:
                    return name, ModelConfidence.MEDIUM

            # Use layer composition to guess
            layer_counts = metadata['layer_counts']
            if 'conv' in layer_counts and layer_counts['conv'] > 3:
                return "CNN", ModelConfidence.MEDIUM
            elif 'lstm' in layer_counts or 'gru' in layer_counts:
                return "RNN", ModelConfidence.MEDIUM
            elif 'dense' in layer_counts and layer_counts.get('conv', 0) == 0:
                return "MLP", ModelConfidence.MEDIUM

        # If we have size information, at least identify as neural network
        if metadata.get('total_params', 0) > 0:
            # Check parameter count to guess model size
            param_count = metadata['total_params']
            if param_count > 1e9:
                return "LargeNN-1B+", ModelConfidence.LOW
            elif param_count > 1e8:
                return "LargeNN-100M+", ModelConfidence.LOW
            elif param_count > 1e7:
                return "MediumNN-10M+", ModelConfidence.LOW
            else:
                return "NeuralNetwork", ModelConfidence.LOW

        # If we have a file but limited analysis
        if metadata.get('limited_analysis', False):
            if metadata.get('is_valid_mxnet_file', False):
                return "MXNet-Model", ModelConfidence.LOW

        return "Unknown", ModelConfidence.UNKNOWN

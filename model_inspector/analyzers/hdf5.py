from typing import Dict, Any, Tuple, List, Optional, Set
import logging
import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class HDF5Analyzer(BaseAnalyzer):
    """
    Analyzer for HDF5 (.h5, .hdf5) model files.

    HDF5 is commonly used for storing neural network weights and configurations
    in frameworks like Keras and PyTorch. This analyzer can identify model
    architectures and extract metadata from HDF5 files.
    """

    # HDF5 magic number (first 8 bytes of file)
    HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'

    # Common model architecture patterns found in HDF5 files
    MODEL_PATTERNS = {
        r'resnet': ('ResNet', ModelConfidence.HIGH),
        r'vgg': ('VGG', ModelConfidence.HIGH),
        r'inception': ('Inception', ModelConfidence.HIGH),
        r'densenet': ('DenseNet', ModelConfidence.HIGH),
        r'efficientnet': ('EfficientNet', ModelConfidence.HIGH),
        r'mobilenet': ('MobileNet', ModelConfidence.HIGH),
        r'bert': ('BERT', ModelConfidence.HIGH),
        r'lstm': ('LSTM', ModelConfidence.HIGH),
        r'gru': ('GRU', ModelConfidence.HIGH),
        r'transformer': ('Transformer', ModelConfidence.MEDIUM),
        r'unet': ('UNet', ModelConfidence.HIGH),
    }

    # Keras-specific group patterns
    KERAS_GROUPS = [
        'model_weights',
        'optimizer_weights',
        'keras_version',
        'backend',
    ]

    # Typical layer types that hint at model architecture
    LAYER_TYPES = {
        'Conv2D': 'ConvolutionalNetwork',
        'Dense': 'NeuralNetwork',
        'LSTM': 'RecurrentNetwork',
        'GRU': 'RecurrentNetwork',
        'BatchNormalization': 'NeuralNetwork',
        'Dropout': 'NeuralNetwork',
        'Embedding': 'LanguageModel',
        'Attention': 'AttentionModel',
        'MultiHeadAttention': 'TransformerModel',
    }

    def __init__(self):
        """Initialize the HDF5 analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.h5', '.hdf5'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze an HDF5 model file.

        Args:
            file_path: Path to the HDF5 file

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
            # Verify it's an HDF5 file
            if not self._is_hdf5_file(file_path):
                raise ValueError(f"File is not a valid HDF5 file: {file_path}")

            # Extract metadata
            metadata = self._extract_hdf5_metadata(file_path)

            # Determine model type
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing HDF5 file {file_path}: {e}")
            raise

    def _is_hdf5_file(self, file_path: str) -> bool:
        """
        Check if the file is a valid HDF5 file.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is an HDF5 file
        """
        with open(file_path, 'rb') as f:
            magic = f.read(8)
            return magic == self.HDF5_MAGIC

    def _extract_hdf5_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from an HDF5 file.

        Args:
            file_path: Path to the HDF5 file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'hdf5'
        }

        try:
            # Try to use h5py if available
            import h5py
            with h5py.File(file_path, 'r') as h5f:
                # Extract file structure
                structure = self._extract_structure(h5f)
                metadata['structure'] = structure

                # Check for Keras model
                is_keras = self._is_keras_model(h5f)
                metadata['is_keras_model'] = is_keras

                if is_keras:
                    # Extract Keras-specific metadata
                    keras_metadata = self._extract_keras_metadata(h5f)
                    metadata.update(keras_metadata)

                # Extract attributes
                attrs = {}
                for key, value in h5f.attrs.items():
                    if isinstance(value, bytes):
                        try:
                            attrs[key] = value.decode('utf-8')
                        except UnicodeDecodeError:
                            attrs[key] = str(value)
                    else:
                        attrs[key] = value
                metadata['attributes'] = attrs

                # Extract dataset info
                datasets = self._extract_datasets_info(h5f)
                if datasets:
                    metadata['datasets'] = datasets

        except ImportError:
            # If h5py is not available, use basic file analysis
            self.logger.warning("h5py not available. Using basic analysis.")
            with open(file_path, 'rb') as f:
                # Read the first 4MB for analysis
                content = f.read(4 * 1024 * 1024)

                # Look for architecture patterns in binary content
                found_architectures = []
                for pattern, (arch, _) in self.MODEL_PATTERNS.items():
                    if re.search(pattern.encode(), content):
                        found_architectures.append(arch)

                if found_architectures:
                    metadata['architecture_hints'] = found_architectures

                # Look for Keras-specific strings
                is_keras = False
                for group in self.KERAS_GROUPS:
                    if group.encode() in content:
                        is_keras = True
                        break

                metadata['is_keras_model'] = is_keras

                # Look for layer types
                found_layers = []
                for layer, _ in self.LAYER_TYPES.items():
                    if layer.encode() in content:
                        found_layers.append(layer)

                if found_layers:
                    metadata['layer_types'] = found_layers

        # Check for companion files
        model_dir = Path(file_path).parent
        model_name = Path(file_path).stem

        # Look for JSON model definition (common for Keras)
        json_file = model_dir / f"{model_name}.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    model_json = json.load(f)
                    metadata['model_json'] = True

                    # Extract architecture info from JSON
                    if 'config' in model_json and 'layers' in model_json['config']:
                        layers = model_json['config']['layers']
                        layer_types = [layer['class_name'] for layer in layers]
                        layer_count = Counter(layer_types)
                        metadata['layer_counts'] = dict(layer_count)
            except (json.JSONDecodeError, IOError):
                pass

        return metadata

    def _extract_structure(self, h5f) -> Dict[str, Any]:
        """
        Extract the structure of an HDF5 file.

        Args:
            h5f: Open HDF5 file object

        Returns:
            Dictionary representation of the file structure
        """

        def _get_item_info(name, item):
            if isinstance(item, h5py.Dataset):
                return {
                    'type': 'dataset',
                    'shape': item.shape,
                    'dtype': str(item.dtype)
                }
            elif isinstance(item, h5py.Group):
                return {
                    'type': 'group',
                    'attrs': {k: str(v) for k, v in item.attrs.items()}
                }

        # Extract top-level structure
        structure = {}

        # Limit to first level to avoid huge outputs
        for key in h5f.keys():
            item = h5f[key]
            structure[key] = _get_item_info(key, item)

            # Add second level for important groups
            if isinstance(item, h5py.Group) and key in ['model_weights', 'optimizer_weights', 'metadata']:
                structure[key]['children'] = {}
                for child_key in item.keys():
                    child = item[child_key]
                    structure[key]['children'][child_key] = _get_item_info(child_key, child)

        return structure

    def _is_keras_model(self, h5f) -> bool:
        """
        Check if the HDF5 file contains a Keras model.

        Args:
            h5f: Open HDF5 file object

        Returns:
            True if the file contains a Keras model
        """
        # Check for Keras-specific attributes or groups
        keras_indicators = ['model_weights', 'backend', 'keras_version']

        # Check attributes
        for key in keras_indicators:
            if key in h5f.attrs:
                return True

        # Check top-level groups
        for key in keras_indicators:
            if key in h5f:
                return True

        return False

    def _extract_keras_metadata(self, h5f) -> Dict[str, Any]:
        """
        Extract Keras-specific metadata from an HDF5 file.

        Args:
            h5f: Open HDF5 file object

        Returns:
            Keras-specific metadata
        """
        keras_metadata = {
            'format_details': 'keras_model'
        }

        # Get Keras version
        if 'keras_version' in h5f.attrs:
            keras_metadata['keras_version'] = h5f.attrs['keras_version'].decode('utf-8')

        # Get backend
        if 'backend' in h5f.attrs:
            keras_metadata['backend'] = h5f.attrs['backend'].decode('utf-8')

        # Try to extract model configuration if present
        if 'model_config' in h5f.attrs:
            try:
                config_json = h5f.attrs['model_config']
                if isinstance(config_json, bytes):
                    config_json = config_json.decode('utf-8')

                config = json.loads(config_json)

                # Extract key information
                if 'config' in config:
                    model_config = config['config']

                    # Get the model class name
                    if 'class_name' in config:
                        keras_metadata['model_class'] = config['class_name']

                    # Extract layer types
                    if 'layers' in model_config:
                        layer_types = [layer['class_name'] for layer in model_config['layers']]
                        layer_count = Counter(layer_types)
                        keras_metadata['layer_counts'] = dict(layer_count)

                        # Store first few layers for architecture identification
                        keras_metadata['first_layers'] = [layer['class_name'] for layer in model_config['layers'][:5]]

                        # Count layer types for model type identification
                        conv_count = sum(1 for layer in layer_types if 'Conv' in layer)
                        dense_count = sum(1 for layer in layer_types if layer == 'Dense')
                        rnn_count = sum(1 for layer in layer_types if layer in ['LSTM', 'GRU', 'SimpleRNN'])
                        attention_count = sum(1 for layer in layer_types
                                              if 'Attention' in layer or layer == 'TransformerBlock')

                        keras_metadata['layer_type_counts'] = {
                            'conv': conv_count,
                            'dense': dense_count,
                            'rnn': rnn_count,
                            'attention': attention_count
                        }
            except (json.JSONDecodeError, ValueError):
                pass

        return keras_metadata

    def _extract_datasets_info(self, h5f) -> Dict[str, Any]:
        """
        Extract information about datasets in the HDF5 file.

        Args:
            h5f: Open HDF5 file object

        Returns:
            Dictionary with dataset information
        """
        datasets_info = {}

        # Function to process each dataset (we'll limit to avoid huge outputs)
        dataset_count = 0
        max_datasets = 50

        def process_dataset(name, dataset):
            nonlocal dataset_count
            if dataset_count >= max_datasets:
                return

            if isinstance(dataset, h5py.Dataset):
                dataset_count += 1
                datasets_info[name] = {
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype)
                }

        # Recursively visit datasets
        h5f.visititems(process_dataset)

        # Calculate total parameter count
        total_params = 0
        for info in datasets_info.values():
            shape = info['shape']
            if shape:
                # Calculate product of dimensions
                params = 1
                for dim in shape:
                    params *= dim
                total_params += params

        # If we hit the limit, indicate there are more datasets
        if dataset_count >= max_datasets:
            datasets_info['_meta'] = {
                'limited': True,
                'message': f'Showing {max_datasets} of {dataset_count}+ datasets'
            }

        datasets_info['_stats'] = {
            'total_parameters': total_params,
            'parameter_count_millions': round(total_params / 1000000, 2)
        }

        return datasets_info

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # First check if it's a Keras model
        if metadata.get('is_keras_model'):
            # If we have layer counts, we can determine the model architecture
            if 'layer_type_counts' in metadata:
                counts = metadata['layer_type_counts']

                # Check for CNN
                if counts.get('conv', 0) > 5:
                    return 'Keras-CNN', ModelConfidence.HIGH

                # Check for RNN
                if counts.get('rnn', 0) > 0:
                    return 'Keras-RNN', ModelConfidence.HIGH

                # Check for Transformer/Attention models
                if counts.get('attention', 0) > 0:
                    return 'Keras-Transformer', ModelConfidence.HIGH

                # Default to plain neural network
                return 'Keras-NN', ModelConfidence.MEDIUM

            # If we have model class, use that
            if 'model_class' in metadata:
                return f"Keras-{metadata['model_class']}", ModelConfidence.HIGH

            # Just Keras without more details
            return 'Keras', ModelConfidence.MEDIUM

        # Check architecture hints
        if 'architecture_hints' in metadata and metadata['architecture_hints']:
            return metadata['architecture_hints'][0], ModelConfidence.MEDIUM

        # Check layer types for non-Keras models
        if 'layer_types' in metadata and metadata['layer_types']:
            layer_types = metadata['layer_types']

            # Check for specific layer combinations
            if 'Conv2D' in layer_types:
                return 'ConvolutionalNetwork', ModelConfidence.MEDIUM

            if 'LSTM' in layer_types or 'GRU' in layer_types:
                return 'RecurrentNetwork', ModelConfidence.MEDIUM

            if 'Attention' in layer_types or 'MultiHeadAttention' in layer_types:
                return 'AttentionModel', ModelConfidence.MEDIUM

        # Check total parameter count for size classification
        if 'datasets' in metadata and '_stats' in metadata['datasets']:
            param_count = metadata['datasets']['_stats'].get('parameter_count_millions', 0)

            if param_count > 100:
                return 'LargeModel', ModelConfidence.LOW
            elif param_count > 10:
                return 'MediumModel', ModelConfidence.LOW
            else:
                return 'SmallModel', ModelConfidence.LOW

        # Default fallback
        return 'HDF5-Model', ModelConfidence.LOW

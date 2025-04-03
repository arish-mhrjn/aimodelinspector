from typing import Dict, Any, Tuple, List, Optional, Set
import logging
import os
import re
import json
import struct
from pathlib import Path

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class TensorFlowAnalyzer(BaseAnalyzer):
    """
    Analyzer for TensorFlow model files (.pb) and SavedModel directories.

    This analyzer can identify various TensorFlow models including frozen graphs,
    GraphDef protocol buffers, and SavedModel directories, extracting metadata
    about model architecture and layers.
    """

    # Model architecture signature patterns in the binary data
    MODEL_SIGNATURES = {
        b'tensorflow/serving': ('TensorFlow-Serving', ModelConfidence.HIGH),
        b'keras_model': ('Keras', ModelConfidence.HIGH),
        b'saved_model.pb': ('SavedModel', ModelConfidence.HIGH),
        b'efficientnet': ('EfficientNet', ModelConfidence.HIGH),
        b'inception': ('Inception', ModelConfidence.HIGH),
        b'resnet': ('ResNet', ModelConfidence.HIGH),
        b'mobilenet': ('MobileNet', ModelConfidence.HIGH),
        b'densenet': ('DenseNet', ModelConfidence.HIGH),
        b'bert': ('BERT', ModelConfidence.HIGH),
        b'transformer': ('TransformerModel', ModelConfidence.MEDIUM),
        b'attention': ('AttentionModel', ModelConfidence.LOW),
    }

    # Operation patterns that suggest model types
    OPERATION_PATTERNS = {
        b'Conv2D': ('ConvolutionalNetwork', ModelConfidence.MEDIUM),
        b'MatMul': ('NeuralNetwork', ModelConfidence.LOW),
        b'BiasAdd': ('NeuralNetwork', ModelConfidence.LOW),
        b'LSTM': ('RNNModel', ModelConfidence.HIGH),
        b'GRU': ('RNNModel', ModelConfidence.HIGH),
        b'Softmax': ('Classifier', ModelConfidence.LOW),
        b'ConcatV2': ('FeatureFusion', ModelConfidence.LOW),
    }

    def __init__(self):
        """Initialize the TensorFlow analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.pb'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        TensorFlow .pb files might contain arbitrary code in certain cases.

        Args:
            file_path: Path to the file

        Returns:
            False as TensorFlow files might not be safe
        """
        # TensorFlow files might contain unsafe operations
        return False

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a TensorFlow model file or directory.

        Args:
            file_path: Path to the TensorFlow model file or directory

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
            # Check if it's a directory (SavedModel format)
            if path.is_dir():
                return self._analyze_savedmodel_dir(file_path)

            # Otherwise analyze as .pb file
            metadata = self._extract_metadata_from_pb(file_path)
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing TensorFlow file {file_path}: {e}")
            raise

    def _analyze_savedmodel_dir(self, dir_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a SavedModel directory.

        Args:
            dir_path: Path to the SavedModel directory

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        path = Path(dir_path)
        metadata = {'format': 'SavedModel'}

        # Check for expected files in a SavedModel
        saved_model_pb = path / 'saved_model.pb'
        if not saved_model_pb.exists():
            self.logger.warning(f"Directory does not contain saved_model.pb: {dir_path}")

        # Extract structure information
        structure = {'files': [], 'directories': []}

        for item in path.iterdir():
            if item.is_file():
                structure['files'].append(item.name)
            elif item.is_dir():
                structure['directories'].append(item.name)

        metadata['structure'] = structure

        # Look for variables directory
        variables_dir = path / 'variables'
        if variables_dir.exists() and variables_dir.is_dir():
            var_files = [f.name for f in variables_dir.iterdir() if f.is_file()]
            metadata['variables'] = var_files

        # Look for assets directory
        assets_dir = path / 'assets'
        if assets_dir.exists() and assets_dir.is_dir():
            asset_files = [f.name for f in assets_dir.iterdir() if f.is_file()]
            metadata['assets'] = asset_files

        # Check for signature file
        signature_file = path / 'signature.json'
        if signature_file.exists():
            try:
                with open(signature_file, 'r') as f:
                    signature = json.load(f)
                    metadata['signature'] = signature
            except (json.JSONDecodeError, IOError):
                pass

        # If saved_model.pb exists, try to extract some metadata from it
        if saved_model_pb.exists():
            pb_metadata = self._extract_metadata_from_pb(str(saved_model_pb))
            metadata.update({f"pb_{k}": v for k, v in pb_metadata.items()})

        # Determine model type
        model_type = "SavedModel"
        confidence = ModelConfidence.MEDIUM

        # Try to get more specific model type
        if 'signature' in metadata and 'name' in metadata['signature']:
            model_name = metadata['signature']['name']
            model_type = f"SavedModel-{model_name}"
            confidence = ModelConfidence.HIGH
        elif saved_model_pb.exists():
            # Get model type from the .pb file analysis
            if 'pb_model_type' in metadata and metadata['pb_model_type'] != 'TensorFlow':
                model_type = metadata['pb_model_type']
                confidence = metadata.get('pb_confidence', ModelConfidence.MEDIUM)

        return model_type, confidence, metadata

    def _extract_metadata_from_pb(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a .pb file using binary analysis.

        Args:
            file_path: Path to the .pb file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'TensorFlow-pb'
        }

        # Read the file in binary mode for pattern matching
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

                # Extract operations by searching for common TF op signatures
                operations = set()
                for op_pattern in self.OPERATION_PATTERNS:
                    if op_pattern in content:
                        operations.add(op_pattern.decode('utf-8', errors='ignore'))

                if operations:
                    metadata['operations'] = list(operations)

                # Look for model signatures
                for signature, (model_type, _) in self.MODEL_SIGNATURES.items():
                    if signature in content:
                        metadata['detected_signatures'] = metadata.get('detected_signatures', [])
                        metadata['detected_signatures'].append(model_type)

                # Try to find tensor shapes (very crude approach)
                shapes = re.findall(rb'shape\[[\d,]+\]', content)
                if shapes:
                    metadata['tensor_shapes'] = [s.decode('utf-8', errors='ignore') for s in shapes[:10]]  # Limit to 10

                # Try to find tensor names
                tensor_names = re.findall(rb'[a-zA-Z0-9_]+:0', content)
                if tensor_names:
                    metadata['tensor_names'] = [t.decode('utf-8', errors='ignore') for t in tensor_names[:10]]

        except Exception as e:
            self.logger.warning(f"Could not fully analyze TensorFlow pb file: {e}")

        return metadata

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # First check from detected signatures
        if 'detected_signatures' in metadata and metadata['detected_signatures']:
            # Return the first signature with highest confidence
            for signature in self.MODEL_SIGNATURES.values():
                model_type, confidence = signature
                if model_type in metadata['detected_signatures']:
                    return model_type, confidence

        # Next try from operations
        if 'operations' in metadata:
            operations = metadata['operations']

            # Check for specific combinations of operations
            if 'Conv2D' in operations and 'MaxPool' in operations:
                return 'ConvolutionalNetwork', ModelConfidence.HIGH

            if 'LSTM' in operations or 'GRU' in operations:
                return 'RecurrentNetwork', ModelConfidence.HIGH

            if 'ConcatV2' in operations and 'Conv2D' in operations:
                return 'FeatureFusionNetwork', ModelConfidence.MEDIUM

            # Look for individual key operations
            for op, (model_type, confidence) in self.OPERATION_PATTERNS.items():
                op_string = op.decode('utf-8', errors='ignore')
                if op_string in operations:
                    return model_type, confidence

        # Default to generic TensorFlow model
        return "TensorFlow", ModelConfidence.LOW

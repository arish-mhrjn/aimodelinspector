# model_inspector/analyzers/jax.py
from typing import Dict, Any, Tuple, List, Optional, Set
import os
import logging
import json
import struct
from pathlib import Path
import re
import msgpack
import numpy as np
from collections import defaultdict, Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class JAXAnalyzer(BaseAnalyzer):
    """
    Analyzer for JAX (.msgpack) model files.

    This analyzer can identify and extract metadata from JAX models serialized
    using MessagePack, including model architecture, parameters, and other attributes.
    """

    # Common model architecture patterns
    MODEL_PATTERNS = {
        r'transformer': ('Transformer', ModelConfidence.HIGH),
        r'bert': ('BERT', ModelConfidence.HIGH),
        r't5': ('T5', ModelConfidence.HIGH),
        r'vit': ('Vision Transformer', ModelConfidence.HIGH),
        r'mlp': ('MLP', ModelConfidence.HIGH),
        r'lstm': ('LSTM', ModelConfidence.HIGH),
        r'gpt': ('GPT', ModelConfidence.HIGH),
        r'flax': ('Flax', ModelConfidence.MEDIUM),
        r'embedding': ('Embedding', ModelConfidence.MEDIUM),
    }

    # JAX-specific layer patterns
    JAX_PATTERNS = {
        r'Dense': 'Dense',
        r'Conv': 'Convolutional',
        r'Attention': 'Attention',
        r'LayerNorm': 'LayerNorm',
        r'Dropout': 'Dropout',
        r'Embed': 'Embedding',
    }

    def __init__(self):
        """Initialize the JAX analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.msgpack'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a JAX model file (.msgpack) to determine its type and extract metadata.

        Args:
            file_path: Path to the JAX model file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid MessagePack file
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            metadata = self._extract_msgpack_metadata(file_path)
            model_type, confidence = self._determine_model_type(metadata)
            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing JAX model file {file_path}: {e}")
            raise

    def _extract_msgpack_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a JAX MessagePack file.

        Args:
            file_path: Path to the JAX model file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'jax_msgpack'
        }

        # Try to parse the MessagePack file safely
        try:
            # Perform a safe, partial loading of the MessagePack file
            with open(file_path, 'rb') as f:
                # Load up to a reasonable size limit to prevent memory issues
                data = f.read(10 * 1024 * 1024)  # Read first 10MB

                # Try to unpack the MessagePack data
                unpacker = msgpack.Unpacker(max_buffer_size=10 * 1024 * 1024)
                unpacker.feed(data)

                # Extract the first few objects to analyze structure
                structure = {}
                param_shapes = {}
                param_count = 0
                param_types = Counter()
                layer_types = Counter()

                # Process a limited number of items to prevent DoS
                for i, item in enumerate(unpacker):
                    if i >= 1000:  # Limit number of items we process
                        break

                    if isinstance(item, dict):
                        # Process dictionary data
                        for key, value in item.items():
                            if isinstance(key, bytes):
                                key = key.decode('utf-8', errors='ignore')

                            # Extract module structure
                            if 'module' in str(key).lower():
                                structure[str(key)] = str(value)[:100]  # Limit string length

                            # Look for model configuration
                            if 'config' in str(key).lower() or 'params' in str(key).lower():
                                if isinstance(value, dict):
                                    metadata['config_keys'] = list(value.keys())[:50]  # Limit number of keys

                    # Process array data to get parameter statistics
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        # Check if this might be a parameter array
                        if isinstance(item[0], (int, float, np.number)):
                            param_count += 1
                            param_shapes[f"param_{param_count}"] = len(item)
                            param_types.update(['numeric'])
                        elif isinstance(item[0], (list, tuple)) and len(item[0]) > 0:
                            param_count += 1
                            param_shapes[f"param_{param_count}"] = (len(item), len(item[0]))
                            param_types.update(['nested_array'])

                    # Look for layer types in key names
                    if isinstance(item, dict):
                        for key in item.keys():
                            key_str = str(key)
                            for pattern in self.JAX_PATTERNS:
                                if re.search(pattern, key_str):
                                    layer_types.update([self.JAX_PATTERNS[pattern]])

                metadata['param_count'] = param_count
                if param_shapes:
                    metadata['param_shapes_sample'] = dict(list(param_shapes.items())[:20])
                if param_types:
                    metadata['param_types'] = dict(param_types)
                if layer_types:
                    metadata['layer_types'] = dict(layer_types)
                if structure:
                    metadata['structure_sample'] = structure

        except Exception as e:
            self.logger.warning(f"Error extracting detailed metadata: {e}")
            metadata['parser_error'] = str(e)

        # Check for companion files
        model_dir = Path(file_path).parent
        model_name = Path(file_path).stem

        # Look for config files
        config_candidates = [
            model_dir / f"{model_name}_config.json",
            model_dir / 'config.json',
            model_dir / 'model_config.json',
            model_dir / 'hyperparams.json',
        ]

        for config_file in config_candidates:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        metadata['config'] = config

                        # Extract key model parameters from config
                        if 'hidden_size' in config:
                            metadata['hidden_size'] = config['hidden_size']
                        if 'num_heads' in config:
                            metadata['num_heads'] = config['num_heads']
                        if 'num_layers' in config:
                            metadata['num_layers'] = config['num_layers']
                        if 'vocab_size' in config:
                            metadata['vocab_size'] = config['vocab_size']
                except (json.JSONDecodeError, IOError):
                    pass

        return metadata

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check if config contains architecture information
        if 'config' in metadata:
            config = metadata['config']

            # Look for specific architecture fields
            arch_key = None
            for key in ['model_type', 'architecture', 'name', 'type']:
                if key in config:
                    arch_key = config[key]
                    break

            if arch_key:
                # Check if it matches known patterns
                arch_key_lower = str(arch_key).lower()
                for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
                    if re.search(pattern, arch_key_lower):
                        return model_type, confidence

                # If not a known pattern but we have a name
                return arch_key, ModelConfidence.MEDIUM

        # Check for layer types that might indicate model architecture
        if 'layer_types' in metadata:
            layer_types = metadata['layer_types']

            # Check for transformer-specific layers
            if 'Attention' in layer_types and 'LayerNorm' in layer_types:
                return 'Transformer', ModelConfidence.MEDIUM

            # Check for CNN
            if 'Convolutional' in layer_types:
                return 'CNN', ModelConfidence.MEDIUM

            # Check for simple MLP
            if 'Dense' in layer_types and len(layer_types) < 3:
                return 'MLP', ModelConfidence.MEDIUM

        # Check structure sample for architectural hints
        if 'structure_sample' in metadata:
            structure = metadata['structure_sample']
            structure_str = json.dumps(structure).lower()

            for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
                if re.search(pattern, structure_str):
                    return model_type, ModelConfidence.LOW

        # Check config keys for hints
        if 'config_keys' in metadata:
            config_keys = ' '.join(metadata['config_keys']).lower()

            for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
                if re.search(pattern, config_keys):
                    return model_type, ModelConfidence.LOW

            # Special case for transformers
            if any(k in config_keys for k in ['attention', 'transformer', 'encoder', 'decoder']):
                return 'Transformer', ModelConfidence.LOW

        # Use basic heuristics based on parameter count
        if 'param_count' in metadata:
            param_count = metadata.get('param_count', 0)
            if param_count > 100:
                return 'JAX-Neural-Network', ModelConfidence.LOW

        # Default fallback
        return 'JAX-Model', ModelConfidence.UNKNOWN

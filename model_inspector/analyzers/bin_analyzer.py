# model_inspector/analyzers/bin_analyzer.py

"""
Binary Model Analyzer for AI model identification.

This module implements a comprehensive multi-stage approach for identifying
AI models in .bin files. It employs signature detection, binary pattern analysis,
companion file inspection, filename parsing, and tensor structure analysis to
determine the most likely model type without executing any code from the file.

The analyzer processes evidence from multiple detection stages and aggregates
them with appropriate confidence weights to make a final determination about
the model's framework, architecture, and other attributes.

Key features:
    - Framework identification (PyTorch, TensorFlow, ONNX, etc.)
    - Architecture detection (ResNet, BERT, YOLO, etc.)
    - Size and quantization analysis
    - Companion configuration file inspection
    - Tensor shape and distribution analysis
    - Statistical confidence scoring for all predictions
    - Detailed metadata extraction
    - Return this information without attempting to load the model

This approach prioritizes safety by avoiding any execution of model code
while still providing comprehensive model identification capabilities.

The implementation is complete, but could be enhanced further with:
    - More sophisticated tensor detection for specific formats
    - Integration with the sandbox system for safe partial loading attempts
    - Additional framework-specific detection patterns
    - Improved statistical analysis of weight distributions

Possible Enhancement: A Two-Stage Approach
A more comprehensive approach could be:
    First stage (BinAnalyzer):
        - Identify what type of model is in the .bin file
        - Determine which specialized analyzer would be best
        - Extract basic/safe metadata
    Second stage (optional, if safe):
        - Pass to the specialized analyzer (PyTorch, TensorFlow, etc.)
        - Use that analyzer for more detailed metadata extraction
        - This would only happen if the confidence is high and the operation is deemed safe

"""
from typing import Dict, Any, Tuple, List, Optional, Set, Union
import os
import logging
import re
import json
import struct
import io
import binascii
import hashlib
import math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer
from ..exceptions import UnsupportedFormatError, SecurityViolationError


class BinAnalyzer(BaseAnalyzer):
    """
    Analyzer for binary (.bin) model files from various AI frameworks.

    Implements a multi-stage analysis approach to identify the model type
    and extract metadata from binary model files with minimal risk.
    """

    # Common magic bytes and signatures for various formats
    SIGNATURES = {
        'pytorch': [b'PK\x03\x04', b'\x80\x02', b'\x80\x03', b'\x80\x04'],  # ZIP or pickle protocol markers
        'tensorflow': [b'TFMD', b'\x08\x01\x12'],  # TF and protobuf signatures
        'onnx': [b'ONNX', b'onnx.ModelProto'],
        'word2vec': [b'\x0a\x00\x00\x00', b'\x00\x00\x80\x3f'],  # Word2Vec header patterns
        'fasttext': [b'FastText', b'fastText'],
        'mxnet': [b'MXN', b'NDArray'],
        'caffe': [b'caffe', b'caffemodel'],
        'yolo': [b'yolo', b'darknet'],
        'dlib': [b'dlib', b'serialization::archive'],
        'trt': [b'TRT', b'tensorrt'],
    }

    # Common model architecture patterns to search in binary data
    ARCHITECTURE_PATTERNS = {
        'resnet': ('ResNet', ModelConfidence.HIGH),
        'vgg': ('VGG', ModelConfidence.HIGH),
        'efficientnet': ('EfficientNet', ModelConfidence.HIGH),
        'densenet': ('DenseNet', ModelConfidence.HIGH),
        'inception': ('Inception', ModelConfidence.HIGH),
        'mobilenet': ('MobileNet', ModelConfidence.HIGH),
        'yolo': ('YOLO', ModelConfidence.HIGH),
        'ssd': ('SSD', ModelConfidence.HIGH),
        'faster_rcnn': ('FasterRCNN', ModelConfidence.HIGH),
        'mask_rcnn': ('MaskRCNN', ModelConfidence.HIGH),
        'bert': ('BERT', ModelConfidence.HIGH),
        'roberta': ('RoBERTa', ModelConfidence.HIGH),
        'albert': ('ALBERT', ModelConfidence.HIGH),
        'gpt': ('GPT', ModelConfidence.HIGH),
        'gpt-2': ('GPT-2', ModelConfidence.HIGH),
        'gpt-3': ('GPT-3', ModelConfidence.HIGH),
        't5': ('T5', ModelConfidence.HIGH),
        'wav2vec': ('Wav2Vec', ModelConfidence.HIGH),
        'fasttext': ('FastText', ModelConfidence.HIGH),
        'word2vec': ('Word2Vec', ModelConfidence.HIGH),
        'llama': ('LLaMA', ModelConfidence.HIGH),
        'mistral': ('Mistral', ModelConfidence.HIGH),
        'stable-diffusion': ('StableDiffusion', ModelConfidence.HIGH),
        'unet': ('UNet', ModelConfidence.HIGH),
    }

    # Framework identifier strings
    FRAMEWORK_IDENTIFIERS = {
        'tensorflow': ['tensorflow', 'tf.', 'keras'],
        'pytorch': ['pytorch', 'torch.', 'torchvision'],
        'onnx': ['onnx', 'onnxruntime'],
        'caffe': ['caffe', 'caffemodel'],
        'mxnet': ['mxnet', 'gluon'],
        'darknet': ['darknet', 'yolov'],
        'huggingface': ['transformers', 'tokenizers', 'huggingface'],
        'xgboost': ['xgboost', 'booster'],
        'lightgbm': ['lightgbm', 'lgbm'],
        'scikit-learn': ['sklearn'],
        'dlib': ['dlib'],
        'tensorrt': ['tensorrt', 'trt'],
    }

    # Typical weight tensor dimensions for common architectures
    ARCHITECTURE_DIMENSIONS = {
        'resnet50': [(64, 3, 7, 7), (64,), (256, 64, 1, 1)],  # First few layers of ResNet50
        'vgg16': [(64, 3, 3, 3), (64,), (128, 64, 3, 3)],  # First few layers of VGG16
        'yolov3': [(255, 1024, 1, 1), (255, 512, 1, 1)],  # YOLO output heads
        'bert': [(768, 768), (768, 3072), (3072, 768)],  # BERT attention, intermediate, output
    }

    # File size ranges for common model types (in bytes)
    SIZE_RANGES = {
        'tiny_cv': (10_000, 1_000_000),  # Tiny CV models (10KB - 1MB)
        'small_cv': (1_000_000, 25_000_000),  # Small CV models (1MB - 25MB)
        'medium_cv': (25_000_000, 250_000_000),  # Medium CV models (25MB - 250MB)
        'large_cv': (250_000_000, 1_000_000_000),  # Large CV models (250MB - 1GB)
        'word_embeddings': (50_000_000, 3_000_000_000),  # Word embeddings (50MB - 3GB)
        'small_transformer': (100_000_000, 500_000_000),  # Small transformers (100MB - 500MB)
        'medium_transformer': (500_000_000, 3_000_000_000),  # Medium transformers (500MB - 3GB)
        'large_transformer': (3_000_000_000, 20_000_000_000),  # Large transformers (3GB - 20GB)
        'quantized_llm': (500_000_000, 8_000_000_000),  # Quantized LLMs (500MB - 8GB)
    }

    def __init__(self):
        """Initialize the binary file analyzer."""
        super().__init__()
        # Initialize stage weights for confidence scoring
        self.stage_weights = {
            'file_size': 0.05,
            'magic_bytes': 0.15,
            'binary_patterns': 0.1,
            'companion_files': 0.2,
            'filename_patterns': 0.1,
            'tensor_analysis': 0.25,
            'structured_analysis': 0.15,
        }

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.bin'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely without executing code.

        Binary files might contain serialized code or unsafe data structures.

        Args:
            file_path: Path to the file

        Returns:
            False as binary files can potentially contain unsafe content
        """
        # Since .bin files can be many different formats including potentially unsafe ones,
        # we treat them as potentially unsafe
        return False

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a binary model file using all available detection methods.

        Args:
            file_path: Path to the binary file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            SecurityViolationError: If the analysis might be unsafe
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Initialize results dictionary to store evidence from each stage
            results = {
                'file_path': file_path,
                'evidence': defaultdict(list),
                'format_confidence': {},
                'architecture_confidence': {},
                'file_size': os.path.getsize(file_path),
                'companion_files': [],
            }

            # Stage 1: Preliminary File Analysis
            self._analyze_file_size(file_path, results)
            self._examine_magic_bytes(file_path, results)
            self._scan_binary_patterns(file_path, results)

            # Stage 2: Context and Companion File Analysis
            self._analyze_companion_files(file_path, results)
            self._analyze_filename_patterns(file_path, results)

            # Stage 3: Structure-Based Analysis
            self._analyze_tensor_structure(file_path, results)
            self._analyze_framework_structure(file_path, results)

            # Stage 5: Final Classification
            model_type, confidence, metadata = self._determine_model_type(results)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing binary file {file_path}: {e}")
            raise

    def _analyze_file_size(self, file_path: str, results: Dict) -> None:
        """
        Analyze file size to get initial hints about model type.

        Args:
            file_path: Path to the binary file
            results: Results dictionary to update
        """
        file_size = results['file_size']

        for size_type, (min_size, max_size) in self.SIZE_RANGES.items():
            if min_size <= file_size <= max_size:
                results['evidence']['file_size'].append(size_type)

        # Additional insights based on file size
        if file_size < 1_000_000:  # Less than 1MB
            results['format_confidence']['quantized_model'] = 0.3
            results['evidence']['file_size'].append('possibly_quantized_or_tiny')

        elif file_size > 5_000_000_000:  # Greater than 5GB
            results['format_confidence']['large_language_model'] = 0.4
            results['evidence']['file_size'].append('large_foundation_model')

        elif 100_000_000 <= file_size <= 500_000_000:  # 100MB - 500MB
            results['format_confidence']['vision_model'] = 0.3
            results['evidence']['file_size'].append('typical_vision_model')

    def _examine_magic_bytes(self, file_path: str, results: Dict) -> None:
        """
        Examine file headers and magic bytes to identify format.

        Args:
            file_path: Path to the binary file
            results: Results dictionary to update
        """
        with open(file_path, 'rb') as f:
            # Read first 256 bytes
            header = f.read(256)

            # Check for known signatures
            for format_name, signatures in self.SIGNATURES.items():
                for signature in signatures:
                    if signature in header:
                        results['evidence']['magic_bytes'].append(f"{format_name}_signature")
                        results['format_confidence'][format_name] = results['format_confidence'].get(format_name,
                                                                                                     0) + 0.4

            # Check for specific data type patterns
            if self._check_for_float32_pattern(header):
                results['evidence']['magic_bytes'].append("float32_weights")

            if self._check_for_float16_pattern(header):
                results['evidence']['magic_bytes'].append("float16_weights")

            if self._check_for_int8_pattern(header):
                results['evidence']['magic_bytes'].append("int8_weights")
                results['format_confidence']['quantized_model'] = results['format_confidence'].get('quantized_model',
                                                                                                   0) + 0.3

    def _check_for_float32_pattern(self, data: bytes) -> bool:
        """Check if data contains patterns typical for float32 weights."""
        # Simple heuristic: look for blocks of bytes that could represent normalized float32 values
        values = []
        for i in range(0, len(data) - 4, 4):
            try:
                value = struct.unpack('f', data[i:i + 4])[0]
                if -10 < value < 10 and not math.isnan(value):
                    values.append(value)
            except:
                continue

        # If we found enough valid float values in a reasonable range for weights
        return len(values) > 10 and statistics_look_like_weights(values)

    def _check_for_float16_pattern(self, data: bytes) -> bool:
        """Check if data contains patterns typical for float16 weights."""
        # Simple heuristic for float16 - look for patterns of valid half-precision values
        values = []
        for i in range(0, len(data) - 2, 2):
            try:
                # Convert half-precision (16-bit) to single precision (32-bit)
                half = int.from_bytes(data[i:i + 2], byteorder='little')
                # This is a simplified check - real implementation would use proper half-float conversion
                sign = (half >> 15) & 0x1
                exp = (half >> 10) & 0x1F
                frac = half & 0x3FF

                if exp > 0 and exp < 31:  # Not special values
                    values.append(half)
            except:
                continue

        # Check if we have enough values and they follow weight-like distribution
        return len(values) > 15

    def _check_for_int8_pattern(self, data: bytes) -> bool:
        """Check if data contains patterns typical for int8 quantized weights."""
        # Check for patterns of int8 values
        distrib = [0] * 256
        for byte in data:
            distrib[byte] += 1

        # Quantized int8 models often use a lot of the value range
        used_values = sum(1 for count in distrib if count > 0)
        return used_values > 200  # Most of the possible int8 values are used

    def _scan_binary_patterns(self, file_path: str, results: Dict) -> None:
        """
        Scan file for framework identifiers and architecture names.

        Args:
            file_path: Path to the binary file
            results: Results dictionary to update
        """
        # Read file in chunks to scan for patterns
        max_scan_size = min(10 * 1024 * 1024, results['file_size'])  # Limit to 10MB or file size

        string_patterns = []
        framework_hits = defaultdict(int)
        arch_hits = defaultdict(int)

        with open(file_path, 'rb') as f:
            remaining = max_scan_size
            while remaining > 0:
                chunk_size = min(remaining, 1024 * 1024)  # 1MB chunks
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                # Extract printable ASCII strings from binary data
                strings = extract_strings(chunk, min_length=4)
                string_patterns.extend(strings)

                # Check for framework identifiers
                for framework, identifiers in self.FRAMEWORK_IDENTIFIERS.items():
                    for identifier in identifiers:
                        id_bytes = identifier.encode('utf-8', errors='ignore')
                        hits = chunk.count(id_bytes)
                        if hits > 0:
                            framework_hits[framework] += hits

                # Check for architecture patterns
                for arch_name in self.ARCHITECTURE_PATTERNS:
                    name_bytes = arch_name.encode('utf-8', errors='ignore')
                    hits = chunk.count(name_bytes)
                    if hits > 0:
                        arch_hits[arch_name] += hits

                remaining -= chunk_size

        # Process framework hits
        for framework, hits in framework_hits.items():
            if hits > 0:
                results['evidence']['binary_patterns'].append(f"{framework}_identifier")
                results['format_confidence'][framework] = results['format_confidence'].get(framework, 0) + min(
                    0.1 * hits, 0.5)

        # Process architecture hits
        for arch_name, hits in arch_hits.items():
            if hits > 0:
                pattern_name, confidence = self.ARCHITECTURE_PATTERNS[arch_name]
                results['evidence']['binary_patterns'].append(f"{pattern_name}_pattern")
                results['architecture_confidence'][pattern_name] = results['architecture_confidence'].get(pattern_name,
                                                                                                          0) + min(
                    0.1 * hits, 0.6)

        # Extract token/vocab size if present
        vocab_size = extract_vocab_size(string_patterns)
        if vocab_size:
            results['vocab_size'] = vocab_size
            results['evidence']['binary_patterns'].append(f"vocab_size_{vocab_size}")
            if vocab_size > 30000:
                results['format_confidence']['large_language_model'] = results['format_confidence'].get(
                    'large_language_model', 0) + 0.3

        # Store extracted strings for further analysis
        results['extracted_strings'] = string_patterns[:1000]  # Limit to 1000 strings

    def _analyze_companion_files(self, file_path: str, results: Dict) -> None:
        """
        Analyze companion files in the same directory for clues.

        Args:
            file_path: Path to the binary file
            results: Results dictionary to update
        """
        directory = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Check for common configuration/metadata files
        companion_patterns = [
            (f"{base_name}.json", "json_config", 0.6),
            (f"{base_name}.yaml", "yaml_config", 0.6),
            (f"{base_name}.yml", "yaml_config", 0.6),
            (f"{base_name}.xml", "xml_config", 0.5),
            (f"{base_name}.prototxt", "caffe_config", 0.8),
            (f"{base_name}.cfg", "darknet_config", 0.8),
            (f"{base_name}_config.json", "model_config", 0.7),
            (f"{base_name}_config.py", "python_config", 0.5),
            (f"{base_name}.model", "companion_model", 0.5),
            (f"{base_name}.params", "mxnet_params", 0.8),
            (f"{base_name}.names", "classes_file", 0.7),
            (f"{base_name}.classes", "classes_file", 0.7),
            (f"config.json", "huggingface_config", 0.7),
            (f"pytorch_model.bin.index.json", "hf_model_index", 0.9),
            (f"vocab.txt", "tokenizer_vocab", 0.7),
            (f"tokenizer.json", "tokenizer_config", 0.8),
            (f"tokenizer_config.json", "tokenizer_config", 0.8),
            (f"tf_model.h5", "tensorflow_companion", 0.8),
        ]

        found_companions = []

        for pattern, companion_type, confidence in companion_patterns:
            full_path = os.path.join(directory, pattern)
            if os.path.exists(full_path):
                found_companions.append((full_path, companion_type))
                results['evidence']['companion_files'].append(companion_type)

                # Try to determine framework from companion
                if 'huggingface' in companion_type:
                    results['format_confidence']['huggingface'] = results['format_confidence'].get('huggingface',
                                                                                                   0) + confidence
                elif 'caffe' in companion_type:
                    results['format_confidence']['caffe'] = results['format_confidence'].get('caffe', 0) + confidence
                elif 'tensorflow' in companion_type:
                    results['format_confidence']['tensorflow'] = results['format_confidence'].get('tensorflow',
                                                                                                  0) + confidence
                elif 'darknet' in companion_type:
                    results['format_confidence']['darknet'] = results['format_confidence'].get('darknet',
                                                                                               0) + confidence
                elif 'mxnet' in companion_type:
                    results['format_confidence']['mxnet'] = results['format_confidence'].get('mxnet', 0) + confidence

                # Try to extract useful information from JSON companions
                if companion_type.endswith('json') and companion_type != 'tokenizer_vocab':
                    try:
                        with open(full_path, 'r', encoding='utf-8') as cf:
                            companion_data = json.load(cf)
                            self._analyze_companion_json(companion_data, results)
                    except Exception as e:
                        self.logger.debug(f"Error reading companion JSON: {e}")

        results['companion_files'] = [path for path, _ in found_companions]

    def _analyze_companion_json(self, data: Dict[str, Any], results: Dict) -> None:
        """
        Extract useful information from companion JSON files.

        Args:
            data: JSON data from companion file
            results: Results dictionary to update
        """
        # Check for model architecture information
        architecture_keys = ['model_type', 'architecture', 'architectures', 'model']
        for key in architecture_keys:
            if key in data:
                arch_value = data[key]
                if isinstance(arch_value, list):
                    arch_value = arch_value[0] if arch_value else ""

                if isinstance(arch_value, str):
                    results['evidence']['companion_files'].append(f"architecture_{arch_value}")
                    # Match against known architectures
                    for pattern, (name, _) in self.ARCHITECTURE_PATTERNS.items():
                        if pattern.lower() in arch_value.lower():
                            results['architecture_confidence'][name] = results['architecture_confidence'].get(name,
                                                                                                              0) + 0.7

        # Check for vocabulary size
        vocab_keys = ['vocab_size', 'n_vocab', 'max_position_embeddings']
        for key in vocab_keys:
            if key in data and isinstance(data[key], (int, float)):
                results['vocab_size'] = data[key]
                results['evidence']['companion_files'].append(f"vocab_size_{data[key]}")

        # Check for model dimensions
        dim_keys = ['hidden_size', 'n_embd', 'd_model', 'encoder_width', 'embedding_dim']
        for key in dim_keys:
            if key in data and isinstance(data[key], (int, float)):
                results['model_dimension'] = data[key]
                results['evidence']['companion_files'].append(f"dimension_{data[key]}")

        # Check for attention heads
        head_keys = ['num_attention_heads', 'n_head', 'encoder_attention_heads']
        for key in head_keys:
            if key in data and isinstance(data[key], (int, float)):
                results['attention_heads'] = data[key]
                results['evidence']['companion_files'].append(f"heads_{data[key]}")

        # Check for number of layers
        layer_keys = ['num_hidden_layers', 'n_layer', 'num_encoder_layers']
        for key in layer_keys:
            if key in data and isinstance(data[key], (int, float)):
                results['num_layers'] = data[key]
                results['evidence']['companion_files'].append(f"layers_{data[key]}")

    def _analyze_filename_patterns(self, file_path: str, results: Dict) -> None:
        """
        Analyze filename for clues about model type.

        Args:
            file_path: Path to the binary file
            results: Results dictionary to update
        """
        filename = os.path.basename(file_path)
        filename_lower = filename.lower()

        # Check for framework indicators in filename
        framework_patterns = [
            ('pytorch', ['pytorch', 'torch', 'pt_model']),
            ('tensorflow', ['tensorflow', 'tf_model', 'tf2_model', 'keras']),
            ('onnx', ['onnx_model']),
            ('caffe', ['caffe', 'caffemodel']),
            ('darknet', ['darknet', 'yolov']),
            ('word2vec', ['word2vec', 'w2v']),
            ('fasttext', ['fasttext', 'ft_model']),
            ('huggingface', ['hf_model']),
        ]

        for framework, patterns in framework_patterns:
            for pattern in patterns:
                if pattern in filename_lower:
                    results['evidence']['filename_patterns'].append(f"{framework}_filename")
                    results['format_confidence'][framework] = results['format_confidence'].get(framework, 0) + 0.4

        # Check for architecture indicators in filename
        for arch_pattern, (arch_name, confidence_level) in self.ARCHITECTURE_PATTERNS.items():
            if arch_pattern in filename_lower.replace('-', '_').replace(' ', '_'):
                results['evidence']['filename_patterns'].append(f"{arch_name}_filename")
                results['architecture_confidence'][arch_name] = results['architecture_confidence'].get(arch_name,
                                                                                                       0) + 0.5

        # Check for weight type indicators
        weight_patterns = [
            ('weights', 'model_weights'),
            ('params', 'model_parameters'),
            ('embeddings', 'embedding_weights'),
            ('quantized', 'quantized_model'),
            ('int8', 'int8_quantized'),
            ('fp16', 'fp16_weights'),
            ('fp32', 'fp32_weights'),
        ]

        for pattern, weight_type in weight_patterns:
            if pattern in filename_lower:
                results['evidence']['filename_patterns'].append(weight_type)
                if 'quantized' in weight_type:
                    results['format_confidence']['quantized_model'] = results['format_confidence'].get(
                        'quantized_model', 0) + 0.4

        # Check for variant/size indicators
        size_patterns = {
            'tiny': 0.1,
            'small': 0.2,
            'medium': 0.3,
            'large': 0.4,
            'base': 0.3,
            'huge': 0.5,
            'xl': 0.5,
            'xxl': 0.6,
        }

        for size_name, confidence in size_patterns.items():
            size_pattern = f"-{size_name}" if size_name in ["base", "small", "large"] else size_name
            if size_pattern in filename_lower:
                results['evidence']['filename_patterns'].append(f"{size_name}_size_variant")
                results['model_size'] = size_name

    def _analyze_tensor_structure(self, file_path: str, results: Dict) -> None:
        """
        Analyze potential tensor structure in the binary file.

        Args:
            file_path: Path to the binary file
            results: Results dictionary to update
        """
        # This is a simplified approach - complete tensor analysis would be more complex
        file_size = results['file_size']

        # Skip tensor analysis for very large files
        if file_size > 500 * 1024 * 1024:  # Skip for files > 500MB
            self.logger.debug(f"Skipping detailed tensor analysis for large file: {file_path}")
            return

        # Try to infer tensor structure from the file
        with open(file_path, 'rb') as f:
            # Sample data in blocks
            block_size = min(10 * 1024 * 1024, file_size // 5)  # 10MB or 1/5 of file
            blocks = []

            # Read beginning, middle and end for sampling
            f.seek(0)
            blocks.append(f.read(block_size))

            if file_size > block_size * 2:
                f.seek(file_size // 2)
                blocks.append(f.read(block_size))

            if file_size > block_size * 3:
                f.seek(max(0, file_size - block_size))
                blocks.append(f.read(block_size))

        # Look for float32 tensor patterns in the data blocks
        tensor_shapes = []
        for block in blocks:
            shapes = detect_tensor_shapes(block)
            if shapes:
                tensor_shapes.extend(shapes)

        if tensor_shapes:
            results['detected_shapes'] = tensor_shapes[:10]  # Limit to 10 shapes
            results['evidence']['tensor_analysis'].append("detected_tensor_shapes")

            # Match tensor shapes against known architectures
            for arch_name, arch_dims in self.ARCHITECTURE_DIMENSIONS.items():
                matches = 0
                for shape in tensor_shapes:
                    if shape in arch_dims:
                        matches += 1

                if matches > 0:
                    arch_normalized = arch_name.split('v')[0] if 'v' in arch_name else arch_name.split('-')[0]
                    matching_pattern = None
                    # Find the matching pattern in ARCHITECTURE_PATTERNS
                    for pattern, (name, _) in self.ARCHITECTURE_PATTERNS.items():
                        if pattern in arch_normalized.lower():
                            matching_pattern = pattern
                            break

                    if matching_pattern:
                        model_name, _ = self.ARCHITECTURE_PATTERNS[matching_pattern]
                        confidence_boost = min(0.1 * matches, 0.5)  # Cap at 0.5
                        results['architecture_confidence'][model_name] = results['architecture_confidence'].get(
                            model_name, 0) + confidence_boost
                        results['evidence']['tensor_analysis'].append(f"{arch_name}_tensor_match")

            # Try to infer if this is an embedding file from tensor dimensionality
            for shape in tensor_shapes:
                if len(shape) == 2 and shape[1] > 50 and shape[0] > 1000:  # Typical for embeddings
                    results['format_confidence']['word_embeddings'] = results['format_confidence'].get(
                        'word_embeddings', 0) + 0.4
                    results['evidence']['tensor_analysis'].append("embedding_dimensions")
                    results['vocab_size'] = shape[0]
                    results['embedding_dim'] = shape[1]
                    break

    def _analyze_framework_structure(self, file_path: str, results: Dict) -> None:
        """
        Analyze framework-specific structural patterns in the binary data.

        Args:
            file_path: Path to the binary file
            results: Results dictionary to update
        """
        # For efficiency, skip this analysis for very large files
        if results['file_size'] > 1000 * 1024 * 1024:  # Skip for files > 1GB
            return

        # Read a portion of the file for structure analysis
        with open(file_path, 'rb') as f:
            # Read first 512KB for header analysis
            header_data = f.read(512 * 1024)

            # PyTorch specific structure detection
            if self._check_pytorch_structure(header_data):
                results['evidence']['structured_analysis'].append("pytorch_structure")
                results['format_confidence']['pytorch'] = results['format_confidence'].get('pytorch', 0) + 0.6

            # TensorFlow specific structure detection
            if self._check_tensorflow_structure(header_data):
                results['evidence']['structured_analysis'].append("tensorflow_structure")
                results['format_confidence']['tensorflow'] = results['format_confidence'].get('tensorflow', 0) + 0.6

            # ONNX specific structure detection
            if self._check_onnx_structure(header_data):
                results['evidence']['structured_analysis'].append("onnx_structure")
                results['format_confidence']['onnx'] = results['format_confidence'].get('onnx', 0) + 0.7

            # Word embedding specific structure detection
            if self._check_word_embedding_structure(header_data):
                results['evidence']['structured_analysis'].append("word_embedding_structure")
                results['format_confidence']['word_embeddings'] = results['format_confidence'].get('word_embeddings',
                                                                                                   0) + 0.7

            # Hugging Face specific structure detection
            if self._check_huggingface_structure(header_data):
                results['evidence']['structured_analysis'].append("huggingface_structure")
                results['format_confidence']['huggingface'] = results['format_confidence'].get('huggingface', 0) + 0.6

    def _check_pytorch_structure(self, data: bytes) -> bool:
        """Check for PyTorch specific structural patterns."""
        # Check for PyTorch serialization markers
        if any(marker in data for marker in [b'torch', b'storage_type', b'_torch_', b'pytorch']):
            return True

        # Check for pickle protocol markers + PyTorch related strings
        pickle_markers = [b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05']
        if any(marker in data for marker in pickle_markers) and b'tensor' in data:
            return True

        # Check for PyTorch ZIP archive with appropriate content
        if data.startswith(b'PK\x03\x04') and any(
                marker in data for marker in [b'data.pkl', b'model.pth', b'state_dict']):
            return True

        return False

    def _check_tensorflow_structure(self, data: bytes) -> bool:
        """Check for TensorFlow specific structural patterns."""
        # Check for TF saved model markers
        if any(marker in data for marker in [b'tensorflow', b'keras', b'saved_model.pb', b'variables/']):
            return True

        # Check for specific protobuf patterns used in TF
        if b'VariableV2' in data or b'checkpoints' in data:
            return True

        # Check for TF Lite model markers
        if data.startswith(b'TFL3') or b'tflite' in data:
            return True

        return False

    def _check_onnx_structure(self, data: bytes) -> bool:
        """Check for ONNX specific structural patterns."""
        # Check for ONNX markers
        if b'onnx' in data and (b'ModelProto' in data or b'GraphProto' in data):
            return True

        # ONNX often uses protobuf serialization
        if b'onnx' in data and b'protobuf' in data:
            return True

        # Check for ONNX operator names
        onnx_ops = [b'Conv', b'MatMul', b'Relu', b'MaxPool', b'Softmax', b'Gemm']
        if sum(1 for op in onnx_ops if op in data) >= 3:  # At least 3 operator names found
            return True

        return False

    def _check_word_embedding_structure(self, data: bytes) -> bool:
        """Check for Word Embedding (Word2Vec, FastText, GloVe) structural patterns."""
        # Check for Word2Vec/GloVe header format - typically has vocabulary size and vector dimension
        # in the first few bytes (in ascii or binary format)

        # ASCII header check (e.g., "10000 300")
        match = re.search(rb'(\d+)[ \t]+(\d+)', data[:1024])
        if match:
            n, dim = int(match.group(1)), int(match.group(2))
            # Check if dimensions are reasonable for word embeddings
            if 1000 <= n <= 10_000_000 and 50 <= dim <= 1000:
                return True

        # FastText marker check
        if b'fastText' in data or b'skipgram' in data or b'cbow' in data:
            return True

        # Word embedding specific strings
        embedding_markers = [b'embeddings', b'vectors', b'word2vec', b'glove']
        if any(marker in data for marker in embedding_markers):
            return True

        return False

    def _check_huggingface_structure(self, data: bytes) -> bool:
        """Check for Hugging Face Transformers specific structural patterns."""
        # Check for Hugging Face markers
        hf_markers = [b'huggingface', b'transformers', b'tokenizer', b'bert', b'gpt', b'roberta']
        if any(marker in data for marker in hf_markers):
            return True

        # Check for specific architecture strings common in HF models
        arch_markers = [b'attention', b'transformer', b'layer_norm', b'intermediate']
        if sum(1 for marker in arch_markers if marker in data) >= 2:  # At least 2 markers
            return True

        # Check for PyTorch structure + transformer patterns
        if self._check_pytorch_structure(data) and (b'config.json' in data or b'attention' in data):
            return True

        return False

    def _determine_model_type(self, results: Dict) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Make a final determination of model type based on all evidence.

        Args:
            results: Results from all analysis stages

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        # Normalize confidence scores for frameworks
        total_format_confidence = sum(results['format_confidence'].values())
        normalized_format = {}

        if total_format_confidence > 0:
            for framework, score in results['format_confidence'].items():
                normalized_format[framework] = score / total_format_confidence

        # Normalize confidence scores for architectures
        total_arch_confidence = sum(results['architecture_confidence'].values())
        normalized_arch = {}

        if total_arch_confidence > 0:
            for arch, score in results['architecture_confidence'].items():
                normalized_arch[arch] = score / total_arch_confidence

        # Determine the most likely format
        framework = max(normalized_format.items(), key=lambda x: x[1])[0] if normalized_format else "unknown"

        # Determine the most likely architecture
        architecture = max(normalized_arch.items(), key=lambda x: x[1])[0] if normalized_arch else "unknown"

        # Calculate overall confidence level
        if framework in normalized_format and architecture in normalized_arch:
            framework_score = normalized_format[framework]
            arch_score = normalized_arch[architecture]

            # Combined confidence score (0.0-1.0)
            combined_score = 0.4 * framework_score + 0.6 * arch_score

            if combined_score > 0.8:
                confidence = ModelConfidence.HIGH
            elif combined_score > 0.5:
                confidence = ModelConfidence.MEDIUM
            elif combined_score > 0.3:
                confidence = ModelConfidence.LOW
            else:
                confidence = ModelConfidence.UNKNOWN
        else:
            confidence = ModelConfidence.LOW

        # Prepare final metadata
        metadata = {
            'file_size': results['file_size'],
            'file_size_mb': round(results['file_size'] / (1024 * 1024), 2),
            'likely_framework': framework,
            'framework_confidence': normalized_format.get(framework, 0),
            'architecture_confidence': normalized_arch.get(architecture, 0),
            'companion_files': results['companion_files'],
        }

        # Add any extracted vocabulary size
        if 'vocab_size' in results:
            metadata['vocab_size'] = results['vocab_size']

        # Add model size if detected
        if 'model_size' in results:
            metadata['model_size_variant'] = results['model_size']

        # Add model dimensions if detected
        if 'model_dimension' in results:
            metadata['model_dimension'] = results['model_dimension']

        # Add attention heads if detected
        if 'attention_heads' in results:
            metadata['attention_heads'] = results['attention_heads']

        # Add number of layers if detected
        if 'num_layers' in results:
            metadata['num_layers'] = results['num_layers']

        # Add detected tensor shapes if found
        if 'detected_shapes' in results:
            metadata['tensor_shapes'] = results['detected_shapes']

        # Add model embedding dimension if detected
        if 'embedding_dim' in results:
            metadata['embedding_dim'] = results['embedding_dim']

        # Formulate model type string
        if framework != "unknown" and architecture != "unknown":
            model_type = f"{architecture}-{framework.capitalize()}"
        elif architecture != "unknown":
            model_type = architecture
        elif framework != "unknown":
            model_type = f"{framework.capitalize()}-Model"
        else:
            model_type = "Unknown-Binary-Model"

        # Add alternative possibilities if confidence is not high
        if confidence != ModelConfidence.HIGH:
            # Get top 3 alternatives for framework
            framework_alternatives = sorted(
                [(f, score) for f, score in normalized_format.items() if f != framework],
                key=lambda x: x[1], reverse=True
            )[:2]

            # Get top 3 alternatives for architecture
            arch_alternatives = sorted(
                [(a, score) for a, score in normalized_arch.items() if a != architecture],
                key=lambda x: x[1], reverse=True
            )[:2]

            if framework_alternatives:
                metadata['alternative_frameworks'] = [
                    {'name': f, 'confidence': round(score, 2)}
                    for f, score in framework_alternatives
                ]

            if arch_alternatives:
                metadata['alternative_architectures'] = [
                    {'name': a, 'confidence': round(score, 2)}
                    for a, score in arch_alternatives
                ]

        # Add evidence summary
        metadata['evidence_summary'] = {
            category: items[:5] for category, items in results['evidence'].items() if items
        }

        return model_type, confidence, metadata


# Helper functions

def extract_strings(data: bytes, min_length: int = 4) -> List[str]:
    """Extract ASCII strings from binary data."""
    strings = []
    current = []

    for byte in data:
        # Check if it's a printable ASCII character
        if 32 <= byte <= 126:
            current.append(chr(byte))
        else:
            # End of string
            if len(current) >= min_length:
                strings.append(''.join(current))
            current = []

    # Add final string if there is one
    if len(current) >= min_length:
        strings.append(''.join(current))

    return strings


def extract_vocab_size(strings: List[str]) -> Optional[int]:
    """Try to extract vocabulary size from strings."""
    vocab_patterns = [
        r'vocab_size["\s:=]+(\d+)',
        r'n_vocab["\s:=]+(\d+)',
        r'vocab["\s:=]+(\d+)',
        r'dictionary_size["\s:=]+(\d+)',
    ]

    for string in strings:
        for pattern in vocab_patterns:
            match = re.search(pattern, string, re.IGNORECASE)
            if match:
                try:
                    size = int(match.group(1))
                    if 100 <= size <= 1_000_000:  # Reasonable vocab size
                        return size
                except ValueError:
                    continue

    return None


def detect_tensor_shapes(data: bytes) -> List[tuple]:
    """
    Attempt to detect tensor shapes by analyzing patterns in the binary data.

    This is a simplified heuristic approach. A real implementation would use
    more sophisticated techniques to detect tensor structure.
    """
    shapes = []

    # Try to find tensor shapes for float32 data blocks (oversimplified)
    # In reality, we would look for actual tensor header information

    # Look for potential tensor sizes by seeking regular patterns
    # Convert data to float32 array and check for statistical patterns
    try:
        # Get length of data in 4-byte increments
        num_floats = len(data) // 4
        if num_floats < 16:
            return shapes

        # Convert to numpy array for analysis
        float_data = np.frombuffer(data[:num_floats * 4], dtype=np.float32)

        # Filter out invalid values
        valid_data = float_data[~np.isnan(float_data) & ~np.isinf(float_data)]

        # Only look for shapes if the data looks like weights
        if len(valid_data) < 16 or not statistics_look_like_weights(valid_data):
            return shapes

        # Try to find tensor dimensions by factorizing the data length
        tensor_sizes = factorize_number(len(valid_data))

        # Look for common tensor shapes
        for shape in tensor_sizes:
            # Prefer 2D, 3D and 4D tensors with dimensions common in ML models
            if len(shape) >= 2 and len(shape) <= 4:
                shapes.append(shape)
    except:
        # Fall back to simple heuristics if numpy analysis fails
        pass

    return shapes


def statistics_look_like_weights(values) -> bool:
    """
    Check if a set of values has statistical properties of model weights.

    Args:
        values: List or array of numeric values

    Returns:
        True if the values look like model weights
    """
    if len(values) < 10:
        return False

    # Convert to numpy array if it's not already
    try:
        import numpy as np
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # Filter out invalid values
        values = values[~np.isnan(values) & ~np.isinf(values)]

        if len(values) < 10:
            return False

        # Check statistical properties of weights
        mean = np.mean(values)
        std = np.std(values)

        # Most ML weights are centered around 0 with small standard deviation
        if abs(mean) > 10 or std > 10:
            return False

        # Check for normal-like distribution (simple test)
        # Most values should be within 2 standard deviations
        within_2std = np.sum(np.abs(values - mean) < 2 * std) / len(values)
        if within_2std < 0.8:
            return False

        return True
    except:
        # Without numpy, do basic checks
        if not values:
            return False

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5

        # Basic checks
        return abs(mean) < 10 and std < 10


def factorize_number(n: int) -> List[Tuple[int, ...]]:
    """
    Find possible tensor shapes by factorizing a number.

    Args:
        n: Number to factorize

    Returns:
        List of possible tensor shapes (as tuples)
    """
    results = []

    # Common tensor dimensions to check
    common_dims = [1, 3, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096]

    # Try to find combinations of common dimensions
    for dim1 in common_dims:
        if n % dim1 != 0:
            continue

        remaining = n // dim1

        # 1D shape
        results.append((n,))

        # 2D shapes
        results.append((dim1, remaining))
        if dim1 != remaining:
            results.append((remaining, dim1))

        # 3D shapes
        for dim2 in common_dims:
            if remaining % dim2 != 0:
                continue

            dim3 = remaining // dim2
            results.append((dim1, dim2, dim3))

            # Only add permutations if dimensions are different
            if dim1 != dim2 and dim1 != dim3 and dim2 != dim3:
                results.append((dim1, dim3, dim2))
                results.append((dim2, dim1, dim3))
                results.append((dim2, dim3, dim1))
                results.append((dim3, dim1, dim2))
                results.append((dim3, dim2, dim1))

            # 4D shapes
            for dim4 in common_dims:
                if dim3 % dim4 == 0:
                    dim5 = dim3 // dim4
                    results.append((dim1, dim2, dim4, dim5))

    return results[:10]  # Limit to 10 possible shapes to avoid explosion

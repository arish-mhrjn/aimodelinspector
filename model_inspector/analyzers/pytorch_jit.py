# model_inspector/analyzers/pytorch_jit.py
from typing import Dict, Any, Tuple, Optional, List, Set
import logging
import os
import re
import zipfile
import json
import io
import struct
from pathlib import Path
from collections import defaultdict, Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class PyTorchJITAnalyzer(BaseAnalyzer):
    """
    Analyzer for PyTorch JIT (TorchScript) models.

    These models are serialized PyTorch models that can be run without Python
    dependencies, commonly used for production deployments.
    """

    # PyTorch JIT model signatures
    PT_ZIP_MAGIC = b'PK\x03\x04'  # ZIP header used by PyTorch
    TORCH_VERSION_KEY = b'version'  # Key for version info in PyTorch models

    # Common files in TorchScript archives
    JIT_FILES = {
        'constants.pkl',
        'data.pkl',
        'code/__torch__.py',
        'model.json',
        'attributes.pkl'
    }

    # Model architecture patterns
    MODEL_PATTERNS = {
        r'resnet': ('ResNet', ModelConfidence.HIGH),
        r'vgg': ('VGG', ModelConfidence.HIGH),
        r'bert': ('BERT', ModelConfidence.HIGH),
        r'gpt': ('GPT', ModelConfidence.HIGH),
        r'transformer': ('Transformer', ModelConfidence.MEDIUM),
        r'efficientnet': ('EfficientNet', ModelConfidence.HIGH),
        r'inception': ('Inception', ModelConfidence.HIGH),
        r'densenet': ('DenseNet', ModelConfidence.HIGH),
        r'yolo': ('YOLO', ModelConfidence.HIGH),
        r'ssd': ('SSD', ModelConfidence.HIGH),
        r'faster_rcnn': ('FasterRCNN', ModelConfidence.HIGH),
        r'mask_rcnn': ('MaskRCNN', ModelConfidence.HIGH),
        r'unet': ('UNet', ModelConfidence.HIGH),
        r'lstm': ('LSTM', ModelConfidence.MEDIUM),
        r'gru': ('GRU', ModelConfidence.MEDIUM),
        r'rnn': ('RNN', ModelConfidence.LOW),
    }

    def __init__(self):
        """Initialize the PyTorch JIT analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.pt', '.pth'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a PyTorch JIT model file.

        Args:
            file_path: Path to the PyTorch JIT model file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid PyTorch JIT model
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # First determine if it's a valid PyTorch JIT model
            is_jit, model_type = self._check_jit_model(file_path)

            if is_jit:
                # Extract metadata from JIT model
                metadata = self._extract_jit_metadata(file_path)

                # Determine model type and confidence
                model_type, confidence = self._determine_model_type(metadata, model_type)

                return model_type, confidence, metadata
            else:
                # Not a JIT model, could be a regular PyTorch save
                metadata = self._extract_regular_torch_metadata(file_path)
                model_type, confidence = self._determine_model_type(metadata, None)

                return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing PyTorch JIT file {file_path}: {e}")
            raise

    def _check_jit_model(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the file is a PyTorch JIT model.

        Args:
            file_path: Path to the model file

        Returns:
            Tuple of (is_jit_model, model_type_hint)
        """
        # Check if it's a zip file (PyTorch models are zip archives)
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)

                if header != self.PT_ZIP_MAGIC:
                    return False, None

            # Check for JIT model specific files
            with zipfile.ZipFile(file_path) as zf:
                contents = set(zf.namelist())

                # Check for JIT-specific files
                jit_indicators = [
                    'constants.pkl',
                    'code/',
                    'model.json'
                ]

                is_jit = any(indicator in contents for indicator in jit_indicators)

                # Try to get a hint about the model type from code files
                model_type_hint = None
                for filename in contents:
                    if filename.startswith('code/') and filename.endswith('.py'):
                        try:
                            with zf.open(filename) as code_file:
                                code_content = code_file.read().decode('utf-8', errors='ignore')

                                # Check for common model architecture patterns
                                for pattern, (arch_name, _) in self.MODEL_PATTERNS.items():
                                    if re.search(pattern, code_content, re.IGNORECASE):
                                        model_type_hint = arch_name
                                        break

                                if model_type_hint:
                                    break
                        except Exception:
                            continue

                return is_jit, model_type_hint

        except (zipfile.BadZipFile, IOError):
            return False, None

    def _extract_jit_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PyTorch JIT model.

        Args:
            file_path: Path to the JIT model

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'pytorch_jit'
        }

        try:
            with zipfile.ZipFile(file_path) as zf:
                # Get list of all files in the archive
                metadata['contained_files'] = zf.namelist()

                # Extract model.json if it exists
                if 'model.json' in zf.namelist():
                    try:
                        with zf.open('model.json') as f:
                            model_info = json.load(f)
                            metadata['model_info'] = model_info

                            # Extract useful properties from model_info
                            if 'torch_version' in model_info:
                                metadata['torch_version'] = model_info['torch_version']

                            # Process producer name and version if available
                            if 'producer_name' in model_info:
                                metadata['producer'] = model_info['producer_name']
                                if 'producer_version' in model_info:
                                    metadata['producer_version'] = model_info['producer_version']
                    except json.JSONDecodeError:
                        pass

                # Look for module hierarchy in code directory
                modules = [name for name in zf.namelist() if name.startswith('code/') and name.endswith('.py')]
                if modules:
                    metadata['module_hierarchy'] = modules

                    # Extract module structure to help identify model architecture
                    module_structure = {}
                    for module_path in modules:
                        try:
                            with zf.open(module_path) as f:
                                content = f.read().decode('utf-8', errors='ignore')
                                # Look for class definitions and function signatures
                                classes = re.findall(r'class\s+(\w+)', content)
                                functions = re.findall(r'def\s+(\w+)', content)

                                if classes or functions:
                                    module_structure[module_path] = {
                                        'classes': classes,
                                        'functions': functions
                                    }

                                    # Check for forward method, a key indicator of model architecture
                                    if 'forward' in functions:
                                        forward_content = re.search(r'def\s+forward\s*\([^)]*\)\s*:([^@]+)',
                                                                    content, re.DOTALL)
                                        if forward_content:
                                            # Extract the forward method body
                                            module_structure[module_path]['forward_method'] = forward_content.group(
                                                1).strip()
                        except Exception:
                            continue

                    if module_structure:
                        metadata['module_structure'] = module_structure

                # Check for ops.yaml which contains operator info
                if 'ops.yaml' in zf.namelist() or 'ops.json' in zf.namelist():
                    operators_file = 'ops.yaml' if 'ops.yaml' in zf.namelist() else 'ops.json'
                    try:
                        with zf.open(operators_file) as f:
                            content = f.read().decode('utf-8')
                            # Simple extraction without full YAML parsing
                            ops = re.findall(r'op: "(.*?)"', content)
                            if ops:
                                metadata['operators'] = ops
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning(f"Error extracting JIT metadata: {e}")

        return metadata

    def _extract_regular_torch_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a regular PyTorch save file.

        Args:
            file_path: Path to the PyTorch file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'pytorch_regular'
        }

        try:
            # Check for zip format
            with open(file_path, 'rb') as f:
                header = f.read(4)

                if header == self.PT_ZIP_MAGIC:
                    # It's a zip file
                    with zipfile.ZipFile(file_path) as zf:
                        metadata['contained_files'] = zf.namelist()

                        # Check for common PyTorch save patterns
                        if 'archive/data.pkl' in zf.namelist():
                            metadata['format_details'] = 'torch_save_with_pickle'

                        # Look for version info
                        if 'archive/version' in zf.namelist():
                            try:
                                with zf.open('archive/version') as f:
                                    version = f.read().strip()
                                    metadata['torch_version'] = version.decode('utf-8')
                            except Exception:
                                pass
                else:
                    # Not a zip file, check for pickle format
                    f.seek(0)
                    check_bytes = f.read(10)  # Read a bit more to check for pickle header

                    if check_bytes.startswith(b'\x80\x02') or check_bytes.startswith(b'\x80\x03'):
                        metadata['format_details'] = 'pickle_format'

        except Exception as e:
            self.logger.warning(f"Error extracting regular PyTorch metadata: {e}")

        return metadata

    def _determine_model_type(self, metadata: Dict[str, Any], model_hint: Optional[str]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata
            model_hint: Optional hint about model type from initial analysis

        Returns:
            Tuple of (model_type, confidence)
        """
        # Start with provided hint if available
        if model_hint:
            return model_hint, ModelConfidence.MEDIUM

        # Check module structure for architecture clues
        if 'module_structure' in metadata:
            # Search for architecture patterns in module names and class names
            for module_path, structure in metadata['module_structure'].items():
                # Check class names
                for class_name in structure.get('classes', []):
                    class_name_lower = class_name.lower()
                    for pattern, (arch_name, confidence) in self.MODEL_PATTERNS.items():
                        if re.search(pattern, class_name_lower):
                            return arch_name, confidence

                # Check forward method content if available
                if 'forward_method' in structure:
                    forward_content = structure['forward_method'].lower()
                    for pattern, (arch_name, confidence) in self.MODEL_PATTERNS.items():
                        if re.search(pattern, forward_content):
                            return arch_name, confidence

        # Check model_info for hints
        if 'model_info' in metadata:
            # Some models store their architecture info in model.json
            model_info = metadata['model_info']

            # Check various fields that might contain architecture info
            for field in ['model_type', 'architecture', 'name', 'model_name']:
                if field in model_info:
                    value = str(model_info[field]).lower()
                    for pattern, (arch_name, confidence) in self.MODEL_PATTERNS.items():
                        if re.search(pattern, value):
                            return arch_name, confidence

        # Check for operators that might indicate model type
        if 'operators' in metadata:
            ops = metadata['operators']

            # Count operation types for heuristic classification
            op_counter = Counter(ops)

            # Check for transformer-specific operations
            transformer_ops = ['attention', 'matmul', 'softmax', 'layernorm']
            has_transformer_ops = any(op for op in ops if any(t_op in op.lower() for t_op in transformer_ops))

            # Check for CNN operations
            cnn_ops = ['conv2d', 'maxpool', 'avgpool', 'batchnorm']
            has_cnn_ops = any(op for op in ops if any(c_op in op.lower() for c_op in cnn_ops))

            # Check for RNN operations
            rnn_ops = ['lstm', 'gru', 'rnn']
            has_rnn_ops = any(op for op in ops if any(r_op in op.lower() for r_op in rnn_ops))

            if has_transformer_ops:
                return 'Transformer', ModelConfidence.MEDIUM
            elif has_cnn_ops:
                return 'CNN', ModelConfidence.MEDIUM
            elif has_rnn_ops:
                return 'RNN', ModelConfidence.MEDIUM

        # Check filename for hints (common naming pattern)
        model_name = Path(metadata.get('file_path', '')).stem.lower()
        for pattern, (arch_name, confidence) in self.MODEL_PATTERNS.items():
            if re.search(pattern, model_name):
                return arch_name, ModelConfidence.LOW

        # Default based on format
        if metadata.get('format') == 'pytorch_jit':
            return 'PyTorch-JIT', ModelConfidence.LOW
        else:
            return 'PyTorch-Model', ModelConfidence.LOW

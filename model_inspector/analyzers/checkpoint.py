from typing import Dict, Any, Tuple, List, Optional, Set
import logging
import os
import re
import pickle
import json
import struct
import io
from pathlib import Path
from collections import defaultdict, Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class CheckpointAnalyzer(BaseAnalyzer):
    """
    Analyzer for checkpoint (.ckpt) files from various ML frameworks.

    This analyzer can identify checkpoint files from TensorFlow, PyTorch, and
    other frameworks, extracting metadata about model architecture and weights.
    """

    # Common signatures and magic bytes
    TF_CKPT_MAGIC = b'model'  # Common pattern in TF checkpoints
    TORCH_MAGIC = b'PK\x03\x04'  # ZIP header used by PyTorch

    # Model architecture name patterns
    MODEL_PATTERNS = {
        r'resnet': ('ResNet', ModelConfidence.HIGH),
        r'vgg': ('VGG', ModelConfidence.HIGH),
        r'bert': ('BERT', ModelConfidence.HIGH),
        r'gpt': ('GPT', ModelConfidence.HIGH),
        r't5': ('T5', ModelConfidence.HIGH),
        r'efficientnet': ('EfficientNet', ModelConfidence.HIGH),
        r'inception': ('Inception', ModelConfidence.HIGH),
        r'llama': ('LLaMA', ModelConfidence.HIGH),
        r'yolo': ('YOLO', ModelConfidence.HIGH),
        r'stable.?diffusion': ('StableDiffusion', ModelConfidence.HIGH),
    }

    # Stable Diffusion component patterns
    SD_COMPONENT_PATTERNS = {
        r'first_stage_model': ('VAE', ModelConfidence.HIGH),
        r'model\.diffusion_model': ('UNet', ModelConfidence.HIGH),
        r'cond_stage_model': ('TextEncoder', ModelConfidence.HIGH),
    }

    def __init__(self):
        """Initialize the Checkpoint analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.ckpt'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        Checkpoint files might contain arbitrary code in pickled objects.

        Args:
            file_path: Path to the file

        Returns:
            False as checkpoint files can contain arbitrary code
        """
        # Checkpoint files often contain pickled objects which can execute code
        return False

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a checkpoint file.

        Args:
            file_path: Path to the checkpoint file

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
            # Attempt to determine checkpoint type
            ckpt_type = self._determine_checkpoint_type(file_path)

            # Extract metadata based on type
            if ckpt_type == 'tensorflow':
                metadata = self._extract_tensorflow_metadata(file_path)
            elif ckpt_type == 'pytorch':
                metadata = self._extract_pytorch_metadata(file_path)
            elif ckpt_type == 'stable_diffusion':
                metadata = self._extract_sd_metadata(file_path)
            else:
                # Generic checkpoint analysis
                metadata = self._extract_generic_metadata(file_path)

            # Add checkpoint type to metadata
            metadata['checkpoint_type'] = ckpt_type

            # Determine model type
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing checkpoint file {file_path}: {e}")
            raise

    def _determine_checkpoint_type(self, file_path: str) -> str:
        """
        Determine the type of checkpoint file.

        Args:
            file_path: Path to the checkpoint file

        Returns:
            Type of checkpoint ('tensorflow', 'pytorch', 'stable_diffusion', or 'unknown')
        """
        # Check file signature
        with open(file_path, 'rb') as f:
            header = f.read(256)  # Read first 256 bytes for identification

            # Check for PyTorch ZIP signature
            if header.startswith(self.TORCH_MAGIC):
                # Further check for stable diffusion
                has_sd_markers = False
                for marker in [b'first_stage_model', b'cond_stage_model', b'diffusion_model']:
                    if marker in header:
                        has_sd_markers = True
                        break

                return 'stable_diffusion' if has_sd_markers else 'pytorch'

            # Check for TensorFlow markers
            if self.TF_CKPT_MAGIC in header or b'tensorflow' in header:
                return 'tensorflow'

        # Try deeper inspection for files without clear signatures
        try:
            # This is a minimal implementation
            # In a real-world scenario, we'd use more sophisticated checks
            # But for safety reasons, we don't want to load the full checkpoint

            # Check for SD-specific file sizes
            size = os.path.getsize(file_path)
            # SD 1.5 checkpoint is typically around 4-7GB
            if 4_000_000_000 <= size <= 7_500_000_000:
                return 'stable_diffusion'

            # Smaller checkpoints might be components
            if 2_000_000_000 <= size <= 3_500_000_000:
                return 'stable_diffusion'

        except Exception:
            pass

        # Default to unknown
        return 'unknown'

    def _extract_tensorflow_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a TensorFlow checkpoint.

        Args:
            file_path: Path to the checkpoint file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'tensorflow_checkpoint'
        }

        # Check for accompanying metadata files
        checkpoint_dir = Path(file_path).parent
        model_name = Path(file_path).stem

        # Check for index file
        index_file = checkpoint_dir / f"{model_name}.index"
        if index_file.exists():
            metadata['has_index_file'] = True

        # Check for data files
        data_files = list(checkpoint_dir.glob(f"{model_name}.data-*"))
        if data_files:
            metadata['data_files'] = [f.name for f in data_files]

        # Check for config files
        config_candidates = [
            checkpoint_dir / 'checkpoint',
            checkpoint_dir / 'model.json',
            checkpoint_dir / 'config.json',
            checkpoint_dir / f"{model_name}_config.json"
        ]

        for config_file in config_candidates:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        if config_file.name == 'checkpoint':
                            # Parse TensorFlow checkpoint file
                            checkpoint_config = {}
                            for line in f:
                                if ':' in line:
                                    key, value = line.split(':', 1)
                                    checkpoint_config[key.strip()] = value.strip()
                            metadata['checkpoint_config'] = checkpoint_config
                        else:
                            # Parse JSON config file
                            config = json.load(f)
                            metadata['config'] = config
                except (json.JSONDecodeError, IOError):
                    pass

        return metadata

    def _extract_pytorch_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PyTorch checkpoint.

        Args:
            file_path: Path to the checkpoint file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'pytorch_checkpoint'
        }

        # We need to be careful not to load the checkpoint directly for security
        # But we can extract some basic information about it

        # Check if it's a zip file
        try:
            import zipfile
            with zipfile.ZipFile(file_path) as zf:
                metadata['contained_files'] = zf.namelist()

                # Look for metadata files in the archive
                for meta_file in ['metadata.json', 'config.json', 'args.json']:
                    if meta_file in zf.namelist():
                        try:
                            with zf.open(meta_file) as f:
                                meta_content = json.load(f)
                                metadata[meta_file.split('.')[0]] = meta_content
                        except json.JSONDecodeError:
                            pass

            return metadata

        except zipfile.BadZipFile:
            # Not a zip file, continue with basic file analysis
            pass

        # Check for binary signatures without loading the whole file
        try:
            with open(file_path, 'rb') as f:
                # Read the first chunk to check for PyTorch pickle protocol
                header = f.read(256)

                if header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03'):
                    metadata['format_details'] = 'pytorch_pickle'

                    # Look for common keys
                    common_keys = [b"model_state_dict", b"state_dict", b"model", b"weights"]
                    for key in common_keys:
                        if key in header:
                            metadata['detected_keys'] = metadata.get('detected_keys', [])
                            metadata['detected_keys'].append(key.decode('utf-8'))

        except IOError:
            pass

        # Check for companion files
        checkpoint_dir = Path(file_path).parent
        model_name = Path(file_path).stem

        # Check for config files
        config_candidates = [
            checkpoint_dir / 'config.json',
            checkpoint_dir / f"{model_name}_config.json",
            checkpoint_dir / 'args.json',
            checkpoint_dir / 'params.json',
        ]

        for config_file in config_candidates:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        metadata['config'] = config
                except (json.JSONDecodeError, IOError):
                    pass

        return metadata

    def _extract_sd_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a Stable Diffusion checkpoint.

        Args:
            file_path: Path to the checkpoint file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'stable_diffusion_checkpoint'
        }

        # Try to determine SD version from file size
        size_gb = metadata['file_size'] / (1024 ** 3)  # Size in GB

        if 1.5 <= size_gb <= 2.5:
            metadata['likely_sd_version'] = '1.x-2.0'
        elif 4.0 <= size_gb <= 7.5:
            metadata['likely_sd_version'] = '1.5 or XL'
        elif size_gb < 1.5:
            metadata['likely_sd_version'] = 'Component or pruned model'

        # We will not load the full checkpoint for security reasons
        # But we can try to extract some metadata from file headers

        try:
            # Look for common Stable Diffusion keys in the file
            with open(file_path, 'rb') as f:
                # Read chunks to scan for key patterns
                chunk_size = 1024 * 1024  # 1MB chunks
                total_read = 0
                max_read = 50 * 1024 * 1024  # Read at most 50MB

                # Components we've found
                components = set()

                while total_read < max_read:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    for pattern in [b'model.diffusion_model', b'first_stage_model', b'cond_stage_model']:
                        if pattern in chunk and pattern.decode('utf-8', errors='ignore') not in components:
                            components.add(pattern.decode('utf-8', errors='ignore'))

                    total_read += len(chunk)

                    # If we found all main components, we can stop
                    if len(components) >= 3:
                        break

                if components:
                    metadata['detected_components'] = list(components)

        except IOError:
            pass

        # Check for companion files
        checkpoint_dir = Path(file_path).parent
        model_name = Path(file_path).stem

        # Look for yaml or json config files
        config_candidates = [
            checkpoint_dir / f"{model_name}.yaml",
            checkpoint_dir / 'model.yaml',
            checkpoint_dir / 'config.yaml',
            checkpoint_dir / 'inference.yaml',
        ]

        for config_file in config_candidates:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        # Try to identify key parameters without full YAML parsing
                        content = f.read()

                        # Look for model dimensions
                        h_match = re.search(r'target_size:\s*\[?(\d+)', content)
                        if h_match:
                            metadata['height'] = int(h_match.group(1))

                        w_match = re.search(r'target_size:\s*\[?\d+,\s*(\d+)', content)
                        if w_match:
                            metadata['width'] = int(w_match.group(1))

                        # Check for model version info
                        v_match = re.search(r'version:\s*[\'"]?([^\'"]+)', content)
                        if v_match:
                            metadata['version'] = v_match.group(1)
                except IOError:
                    pass

        return metadata

    def _extract_generic_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract generic metadata from an unknown checkpoint format.

        Args:
            file_path: Path to the checkpoint file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'generic_checkpoint'
        }

        # Try to gather some information from file headers
        with open(file_path, 'rb') as f:
            header = f.read(1024)  # Read first 1KB

            # Check for common binary signatures
            if header.startswith(b'PK\x03\x04'):
                metadata['format_details'] = 'zip_archive'
            elif header.startswith(b'\x89HDF'):
                metadata['format_details'] = 'hdf5_format'
            elif header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03'):
                metadata['format_details'] = 'pickle_format'

            # Look for architecture hints
            for arch_pattern, (model_type, _) in self.MODEL_PATTERNS.items():
                if arch_pattern.encode() in header:
                    metadata['architecture_hint'] = model_type
                    break

        # Check for companion files
        checkpoint_dir = Path(file_path).parent
        model_name = Path(file_path).stem

        # Look for config files
        for ext in ['.json', '.yaml', '.yml', '.cfg', '.config']:
            config_file = checkpoint_dir / f"{model_name}{ext}"
            if config_file.exists():
                metadata['config_file'] = str(config_file.name)
                break

        return metadata

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check for Stable Diffusion
        if metadata.get('checkpoint_type') == 'stable_diffusion':
            version = metadata.get('likely_sd_version', '').lower()
            if 'xl' in version:
                return 'StableDiffusion-XL', ModelConfidence.HIGH
            elif '2.0' in version:
                return 'StableDiffusion-2.0', ModelConfidence.HIGH
            else:
                return 'StableDiffusion', ModelConfidence.HIGH

        # Check for architecture hints
        if 'architecture_hint' in metadata:
            return metadata['architecture_hint'], ModelConfidence.MEDIUM

        # Check config for architecture info
        if 'config' in metadata:
            config = metadata['config']

            # Look for architecture or model type fields
            arch_key = None
            for key in ['architecture', 'model_type', 'model_name', 'name']:
                if key in config:
                    arch_key = config[key]
                    break

            if arch_key:
                # Check if it matches known patterns
                arch_key_lower = arch_key.lower()
                for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
                    if re.search(pattern, arch_key_lower):
                        return model_type, confidence

                # If not a known pattern but we have a name
                return arch_key, ModelConfidence.MEDIUM

        # Check detected components for SD
        if 'detected_components' in metadata and metadata['format'] == 'stable_diffusion_checkpoint':
            if 'model.diffusion_model' in metadata['detected_components']:
                return 'StableDiffusion-UNet', ModelConfidence.HIGH
            elif 'first_stage_model' in metadata['detected_components']:
                return 'StableDiffusion-VAE', ModelConfidence.HIGH
            elif 'cond_stage_model' in metadata['detected_components']:
                return 'StableDiffusion-TextEncoder', ModelConfidence.HIGH

        # If we have a format, use it
        if 'format' in metadata:
            format_type = metadata['format']
            if 'tensorflow' in format_type:
                return 'TensorFlow-Checkpoint', ModelConfidence.HIGH
            elif 'pytorch' in format_type:
                return 'PyTorch-Checkpoint', ModelConfidence.HIGH

        # Default fallback
        return 'Unknown-Checkpoint', ModelConfidence.LOW

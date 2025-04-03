from typing import Dict, Any, Tuple, List, Optional, Set
import logging
import os
import io
import re
import zipfile
import json
from pathlib import Path

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class PyTorchAnalyzer(BaseAnalyzer):
    """
    Analyzer for PyTorch model files.

    This analyzer can identify various PyTorch model types including computer vision
    models, language models, and custom architectures based on their structure.
    """

    # Patterns in state_dict keys that identify model types
    MODEL_PATTERNS = {
        # General architecture patterns
        r'bert\.': ('BERT', ModelConfidence.HIGH),
        r'gpt\.': ('GPT', ModelConfidence.HIGH),
        r'llama\.': ('LLaMA', ModelConfidence.HIGH),
        r'resnet': ('ResNet', ModelConfidence.HIGH),
        r'vgg': ('VGG', ModelConfidence.HIGH),
        r'densenet': ('DenseNet', ModelConfidence.HIGH),
        r'efficientnet': ('EfficientNet', ModelConfidence.HIGH),
        r'unet\.': ('UNet', ModelConfidence.MEDIUM),

        # Transformers patterns
        r'encoder\.layer\.\d+\.attention': ('Transformer-Encoder', ModelConfidence.HIGH),
        r'decoder\.block\.\d+\.layer\.\d+\.SelfAttention': ('Transformer-Decoder', ModelConfidence.HIGH),
        r'self_attn\.': ('Transformer', ModelConfidence.MEDIUM),
        r'encoder\.layers\.\d+\.self_attn': ('Transformer-Encoder', ModelConfidence.HIGH),

        # Stable Diffusion patterns
        r'model\.diffusion_model': ('StableDiffusion-UNet', ModelConfidence.HIGH),
        r'first_stage_model': ('StableDiffusion-VAE', ModelConfidence.HIGH),
        r'cond_stage_model': ('StableDiffusion-TextEncoder', ModelConfidence.HIGH),

        # GAN patterns
        r'generator\.': ('GAN-Generator', ModelConfidence.HIGH),
        r'discriminator\.': ('GAN-Discriminator', ModelConfidence.HIGH),

        # RL model patterns
        r'policy_net\.': ('PolicyNetwork', ModelConfidence.HIGH),
        r'q_net\.': ('QNetwork', ModelConfidence.HIGH),
        r'actor\.': ('ActorNetwork', ModelConfidence.MEDIUM),
        r'critic\.': ('CriticNetwork', ModelConfidence.MEDIUM),
    }

    # Common dimension sizes for specific model architectures
    EMBEDDING_DIMS = {
        768: ['BERT-Base', 'RoBERTa-Base', 'DistilBERT'],
        512: ['GPT2-Small', 'CLIP'],
        1024: ['BERT-Large', 'RoBERTa-Large', 'GPT2-Medium'],
        1280: ['GPT2-Large', 'CLIP-Large'],
        1536: ['GPT2-XL'],
        2048: ['GPT-NeoX', 'CLIP-Huge'],
        4096: ['GPT-J', 'LLaMA-7B'],
        5120: ['LLaMA-13B'],
        8192: ['LLaMA-65B', 'LLaMA-70B'],
    }

    # Known model architectures based on files within the archive
    ARCHITECTURE_FILES = {
        'config.json': 'HuggingFace',
        'pytorch_model.bin.index.json': 'HuggingFace-Sharded',
        'vocab.txt': 'BERT-family',
        'tokenizer.json': 'Tokenizer-Included',
        'merges.txt': 'GPT-family',
    }

    def __init__(self):
        """Initialize the PyTorch analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.pt', '.pth'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        PyTorch files can execute arbitrary code through pickled objects.

        Args:
            file_path: Path to the file

        Returns:
            False since PyTorch files can contain arbitrary code
        """
        # PyTorch files can contain arbitrary code in pickled format
        return False

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a PyTorch model file.

        Args:
            file_path: Path to the PyTorch model file

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
            # Check if it's a zip file (common for HuggingFace models)
            if self._is_zip_file(file_path):
                return self._analyze_zip_archive(file_path)

            # Check if it's a directory (SavedModel format)
            if path.is_dir():
                return self._analyze_saved_model_dir(file_path)

            # Standard PyTorch file analysis using metadata extraction
            metadata = self._extract_metadata_from_file(file_path)
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing PyTorch file {file_path}: {e}")
            raise

    def _is_zip_file(self, file_path: str) -> bool:
        """
        Check if a file is a zip archive.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is a zip archive
        """
        try:
            with zipfile.ZipFile(file_path) as _:
                return True
        except zipfile.BadZipFile:
            return False

    def _analyze_zip_archive(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a zip archive containing model files.

        Args:
            file_path: Path to the zip file

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        metadata = {'format': 'zip_archive'}
        file_list = []

        with zipfile.ZipFile(file_path) as zip_file:
            file_list = zip_file.namelist()

            # Try to find and parse config files
            for config_file in ['config.json', 'model_config.json', 'model_args.json']:
                if config_file in file_list:
                    with zip_file.open(config_file) as f:
                        try:
                            config = json.load(f)
                            metadata['config'] = config
                            break
                        except json.JSONDecodeError:
                            pass

        metadata['contained_files'] = file_list

        # Determine model type based on files in the archive
        model_type, confidence = self._determine_archive_model_type(file_list, metadata)

        return model_type, confidence, metadata

    def _determine_archive_model_type(
            self,
            file_list: List[str],
            metadata: Dict[str, Any]
    ) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from archive contents.

        Args:
            file_list: List of files in the archive
            metadata: Metadata including config if available

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check for known architecture files
        for file, arch in self.ARCHITECTURE_FILES.items():
            if any(f.endswith(file) for f in file_list):
                return arch, ModelConfidence.MEDIUM

        # Check config for architecture info
        if 'config' in metadata:
            config = metadata['config']

            # Common HuggingFace config fields
            if 'architectures' in config:
                arch = config['architectures'][0]
                return arch, ModelConfidence.HIGH

            if 'model_type' in config:
                model_type = config['model_type'].upper()
                return model_type, ModelConfidence.HIGH

            # Extract dimensional info for architecture guessing
            if 'hidden_size' in config:
                hidden_size = config['hidden_size']
                for dim, models in self.EMBEDDING_DIMS.items():
                    if hidden_size == dim:
                        # Return the first matched model as a guess
                        return models[0], ModelConfidence.MEDIUM

        # If bin files exist, it's likely a HuggingFace model
        bin_files = [f for f in file_list if f.endswith('.bin')]
        if bin_files:
            return "HuggingFace", ModelConfidence.LOW

        # Generic PyTorch model
        return "PyTorch", ModelConfidence.LOW

    def _analyze_saved_model_dir(self, dir_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a SavedModel directory.

        Args:
            dir_path: Path to the SavedModel directory

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        path = Path(dir_path)
        metadata = {'format': 'saved_model_dir'}

        # List files in the directory
        file_list = []
        for p in path.glob('**/*'):
            if p.is_file():
                file_list.append(str(p.relative_to(path)))

        metadata['contained_files'] = file_list

        # Look for config files
        config_files = [
            path / 'config.json',
            path / 'model_config.json',
            path / 'model_args.json'
        ]

        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        metadata['config'] = config
                        break
                except (json.JSONDecodeError, IOError):
                    pass

        # Determine model type from directory contents
        return self._determine_archive_model_type(file_list, metadata)

    def _extract_metadata_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PyTorch model file without full loading.

        Args:
            file_path: Path to the model file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'pytorch_model'
        }

        # Try to safely analyze file header without loading the model
        try:
            # Read the first few bytes to check for PyTorch magic numbers and version
            with open(file_path, 'rb') as f:
                header = f.read(24)  # Read enough for PyTorch header

                # Check for PyTorch magic number (zip files and torch files start differently)
                if header[:2] == b'\x50\x4b':  # PK header (zip)
                    metadata['file_type'] = 'pytorch_zip'
                elif header[:2] == b'\x80\x02':  # Pickle protocol 2
                    metadata['file_type'] = 'pytorch_pickle'

                # Try to extract more info without executing code
                # This requires a deeper analysis that's too complex for this example
        except Exception as e:
            self.logger.warning(f"Could not extract PyTorch file header: {e}")

        # For a real implementation, we'd need to safely extract state dict structure
        # without executing potentially malicious code

        return metadata

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Here we would normally analyze state_dict keys against MODEL_PATTERNS
        # But since we're not loading the model for safety, we use limited info

        if 'config' in metadata:
            config = metadata['config']

            if 'architectures' in config:
                arch = config['architectures'][0]
                return arch, ModelConfidence.HIGH

            if 'model_type' in config:
                model_type = config['model_type'].upper()
                return model_type, ModelConfidence.HIGH

        # Based on file type
        file_type = metadata.get('file_type', '')
        if 'pytorch_zip' in file_type:
            return "HuggingFace", ModelConfidence.LOW

        # Default fallback
        return "PyTorch", ModelConfidence.LOW

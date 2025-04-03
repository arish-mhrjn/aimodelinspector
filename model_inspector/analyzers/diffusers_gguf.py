from typing import Dict, Any, Tuple, Optional, List, Set
import re
import logging
from pathlib import Path

from ..models.confidence import ModelConfidence
from .base_diffusion_analyzer import BaseDiffusionAnalyzer
from .gguf import GGUFAnalyzer


class DiffusersGGUFAnalyzer(BaseDiffusionAnalyzer):
    """
    Analyzer for diffusion models in GGUF format.

    This specializes the generic GGUF analyzer to detect and extract
    metadata specific to diffusion models stored in GGUF format.
    """

    def __init__(self):
        """Initialize the GGUF diffusion model analyzer."""
        super().__init__()
        self.gguf_analyzer = GGUFAnalyzer()

    def get_supported_extensions(self) -> set:
        """Get file extensions supported by this analyzer."""
        return {'.gguf'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a GGUF file to determine if it's a diffusion model.

        Args:
            file_path: Path to the GGUF file

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        # Use the base GGUF analyzer first
        base_type, base_confidence, metadata = self.gguf_analyzer.analyze(file_path)

        # Create a tensor_shapes dict for compatibility with base methods
        tensor_info = self._extract_tensor_info(metadata)
        metadata['tensor_shapes'] = tensor_info

        path = Path(file_path)

        # Extract Diffusers metadata from file path and context
        is_diffusers, diffusers_metadata = self.extract_diffusers_metadata(path, metadata)
        metadata.update(diffusers_metadata)

        # Check if this is a LoRA adapter
        if self.is_lora_adapter(metadata, tensor_info):
            model_type, lora_info = self.analyze_lora(metadata, tensor_info)
            metadata.update(lora_info)
            return model_type, ModelConfidence.HIGH, metadata

        # Check if this is a textual embedding
        if self.is_sd_textual_embedding(metadata, tensor_info):
            embedding_meta = self.extract_sd_embedding_metadata(metadata, tensor_info)
            metadata.update(embedding_meta)
            model_type = f"SD_TextualEmbedding"
            if 'model_compatibility' in embedding_meta:
                model_type = f"{embedding_meta['model_compatibility']}_TextualEmbedding"
            return model_type, ModelConfidence.HIGH, metadata

        # Check if this is a diffusion model
        if is_diffusers:
            # Determine the specific diffusion model type
            model_type, confidence = self.determine_diffusers_model_type(metadata, tensor_info)
            return model_type, confidence, metadata

        # Now check if this is actually a diffusion model
        is_diffusion, diffusion_type, confidence = self._check_diffusion_model(metadata)

        if is_diffusion:
            # Add diffusion-specific metadata
            diffusion_metadata = self._extract_diffusion_metadata(metadata)
            metadata.update(diffusion_metadata)
            return diffusion_type, confidence, metadata

        # If not a diffusion model, return the base analysis
        return base_type, base_confidence, metadata

    def _extract_tensor_info(self, metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract tensor information from GGUF metadata into a format compatible with other analyzers.

        Args:
            metadata: GGUF metadata dictionary

        Returns:
            Dictionary of tensor information
        """
        tensor_info = {}

        # Extract tensor names and shapes from GGUF metadata
        tensor_count = metadata.get('tensor_count', 0)

        for i in range(tensor_count):
            tensor_name_key = f'tensor.{i}.name'
            tensor_shape_key = f'tensor.{i}.shape'

            if tensor_name_key in metadata and tensor_shape_key in metadata:
                name = metadata[tensor_name_key]
                shape = metadata[tensor_shape_key]

                tensor_info[name] = {
                    'shape': shape,
                    'dtype': metadata.get(f'tensor.{i}.dtype', 'unknown')
                }

        return tensor_info

    def _check_diffusion_model(self, metadata: Dict[str, Any]) -> Tuple[bool, str, ModelConfidence]:
        """
        Check if the GGUF file is a diffusion model.

        Args:
            metadata: Metadata from base GGUF analysis

        Returns:
            Tuple of (is_diffusion, model_type, confidence)
        """
        # Check for diffusion model indicators in metadata
        if 'general.architecture' in metadata:
            arch = metadata['general.architecture'].lower()
            for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
                if re.search(pattern, arch):
                    return True, model_type, confidence

        # Look for evidence in tensor names if available
        if 'tensor_count' in metadata and metadata['tensor_count'] > 0:
            for key in metadata.keys():
                if key.startswith('tensor.'):
                    tensor_name = key.split('.')[1]
                    for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
                        if re.search(pattern, tensor_name.lower()):
                            return True, model_type, confidence

        # Check for common diffusion model configuration keys
        diffusion_config_keys = [
            'unet_config', 'vae_config', 'text_encoder_config',
            'scheduler_config', 'tokenizer_config', 'safety_checker_config'
        ]

        for key in diffusion_config_keys:
            if key in metadata:
                return True, "StableDiffusion", ModelConfidence.MEDIUM

        return False, "GGUF_Model", ModelConfidence.LOW

    def _extract_diffusion_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract diffusion model specific metadata.

        Args:
            metadata: Base metadata from GGUF analysis

        Returns:
            Diffusion-specific metadata
        """
        # Extract relevant diffusion model properties
        diffusion_metadata = {
            "is_diffusion_model": True,
        }

        # Look for specific diffusion parameters
        unet_params = {}
        vae_params = {}
        text_encoder_params = {}

        # Extract UNet parameters
        for key in metadata:
            if key.startswith('unet.') or key.startswith('diffusion_model.'):
                param_name = key.split('.', 1)[1]
                unet_params[param_name] = metadata[key]
                diffusion_metadata['has_unet'] = True

            # Extract VAE parameters
            elif key.startswith('vae.') or key.startswith('first_stage_model.'):
                param_name = key.split('.', 1)[1]
                vae_params[param_name] = metadata[key]
                diffusion_metadata['has_vae'] = True

            # Extract text encoder parameters
            elif key.startswith('text_encoder.') or key.startswith('cond_stage_model.'):
                param_name = key.split('.', 1)[1]
                text_encoder_params[param_name] = metadata[key]
                diffusion_metadata['has_text_encoder'] = True

        # Store component parameters if found
        if unet_params:
            diffusion_metadata['unet_params'] = unet_params

        if vae_params:
            diffusion_metadata['vae_params'] = vae_params

        if text_encoder_params:
            diffusion_metadata['text_encoder_params'] = text_encoder_params

        # Look for model type indicators
        if 'general.architecture' in metadata:
            arch_name = metadata['general.architecture'].lower()

            # Identify model family
            if 'sdxl' in arch_name:
                diffusion_metadata['model_family'] = 'SDXL'
            elif 'sd2' in arch_name:
                diffusion_metadata['model_family'] = 'SD2.x'
            elif 'sd1' in arch_name:
                diffusion_metadata['model_family'] = 'SD1.x'
            elif 'stable-cascade' in arch_name or 'cascade' in arch_name:
                diffusion_metadata['model_family'] = 'StableCascade'
            elif 'flux' in arch_name:
                diffusion_metadata['model_family'] = 'Flux'

        # Extract pipeline information if available
        if 'pipeline' in metadata:
            diffusion_metadata['pipeline'] = metadata['pipeline']

        # Extract scheduler information
        if 'scheduler.type' in metadata:
            diffusion_metadata['scheduler_type'] = metadata['scheduler.type']

        return diffusion_metadata

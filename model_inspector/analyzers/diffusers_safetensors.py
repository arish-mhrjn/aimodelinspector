from typing import Dict, Any, Tuple, Optional, List, Set
import logging
from pathlib import Path

from ..models.confidence import ModelConfidence
from .base_diffusion_analyzer import BaseDiffusionAnalyzer
from .safetensors import SafetensorsAnalyzer


class DiffusersSafetensorsAnalyzer(BaseDiffusionAnalyzer):
    """
    Analyzer for Diffusers safetensors models.

    This analyzer identifies and extracts metadata from Hugging Face Diffusers
    models stored in the safetensors format, including text-to-image models,
    image-to-image models, and other generative AI architectures.
    """

    def __init__(self):
        """Initialize the Diffusers safetensors analyzer."""
        super().__init__()
        self.safetensors_analyzer = SafetensorsAnalyzer()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.safetensors'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a Diffusers safetensors file to determine its model type and metadata.

        Args:
            file_path: Path to the safetensors file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid safetensors file
            Exception: If there is an error during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # First use the SafetensorsAnalyzer to get basic metadata
            _, _, safetensors_metadata = self.safetensors_analyzer.analyze(file_path)

            # Check if we need to set the format to safetensors if it's wrong
            if safetensors_metadata.get('format', '') != 'safetensors':
                safetensors_metadata['format'] = 'safetensors'

            # Check if this is a Diffusers model
            is_diffusers, diffusers_metadata = self.extract_diffusers_metadata(path, safetensors_metadata)

            # Merge both metadata sets
            combined_metadata = {**safetensors_metadata, **diffusers_metadata}

            # Add format identifier
            combined_metadata['format'] = 'safetensors'

            # First check for LoRA adapter specifically - these are most common
            if self.is_lora_adapter(combined_metadata, safetensors_metadata.get('tensor_shapes', {})):
                model_type, lora_info = self.analyze_lora(combined_metadata, safetensors_metadata)
                combined_metadata.update(lora_info)
                return model_type, ModelConfidence.HIGH, combined_metadata

            # Check for textual embeddings
            if self.is_sd_textual_embedding(combined_metadata, safetensors_metadata.get('tensor_shapes', {})):
                embedding_meta = self.extract_sd_embedding_metadata(combined_metadata,
                                                                   safetensors_metadata.get('tensor_shapes', {}))
                combined_metadata.update(embedding_meta)
                model_type = f"SD_TextualEmbedding"
                if 'model_compatibility' in embedding_meta:
                    model_type = f"{embedding_meta['model_compatibility']}_TextualEmbedding"
                return model_type, ModelConfidence.HIGH, combined_metadata

            # Now check if it's a Diffusers model component
            if is_diffusers:
                # It's a Diffusers model, determine specific type
                model_type, confidence = self.determine_diffusers_model_type(
                    combined_metadata,
                    safetensors_metadata.get('tensor_shapes', {})
                )

                return model_type, confidence, combined_metadata
            else:
                # Just a regular safetensors file, not a Diffusers model
                # But still try to identify some common patterns
                model_type = "Unknown"
                confidence = ModelConfidence.LOW

                # Check for common shapes to identify without config
                common_shapes = safetensors_metadata.get('common_shapes', {})
                for shape_tuple, count in common_shapes.items():
                    # Check for SDXL LoRA patterns (common shape patterns)
                    if isinstance(shape_tuple, tuple) and len(shape_tuple) == 2:
                        # SDXL LoRA pattern with 1280 dimension
                        if (shape_tuple[0] == 32 and shape_tuple[1] == 1280) or (
                                shape_tuple[0] == 1280 and shape_tuple[1] == 32):
                            return "SDXL_LoRA", ModelConfidence.HIGH, combined_metadata
                        # SD1.5 LoRA pattern with 768 dimension
                        elif (shape_tuple[0] == 32 and shape_tuple[1] == 768) or (
                                shape_tuple[0] == 768 and shape_tuple[1] == 32):
                            return "SD15_LoRA", ModelConfidence.HIGH, combined_metadata
                        # SD2.1 LoRA pattern with 1024 dimension
                        elif (shape_tuple[0] == 32 and shape_tuple[1] == 1024) or (
                                shape_tuple[0] == 1024 and shape_tuple[1] == 32):
                            return "SD21_LoRA", ModelConfidence.HIGH, combined_metadata

                    # Check for embedding pattern (small number of vectors × 768/1024/1280)
                    if isinstance(shape_tuple, tuple) and len(shape_tuple) == 2:
                        # Typical embedding dimensions
                        if shape_tuple[1] == 768 and shape_tuple[0] < 10:
                            return "SD15_TextualEmbedding", ModelConfidence.HIGH, combined_metadata
                        elif shape_tuple[1] == 1024 and shape_tuple[0] < 10:
                            return "SD21_TextualEmbedding", ModelConfidence.HIGH, combined_metadata
                        elif shape_tuple[1] == 1280 and shape_tuple[0] < 10:
                            return "SDXL_TextualEmbedding", ModelConfidence.HIGH, combined_metadata

                return model_type, confidence, combined_metadata

        except Exception as e:
            self.logger.error(f"Error analyzing Diffusers file {file_path}: {e}")
            raise

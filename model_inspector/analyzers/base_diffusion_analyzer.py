from typing import Dict, Any, Tuple, Optional, List, Set
import re
import logging
from pathlib import Path
from collections import defaultdict

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer
from .diffusion_constants import (
    DIFFUSION_MODEL_PATTERNS,
    DIFFUSION_CONFIG_HINTS,
    LORA_PATTERNS,
    EMBEDDING_TRIGGER_PATTERNS,
    TRIGGER_WORD_FIELDS,
    CONFIG_FIELDS
)


class BaseDiffusionAnalyzer(BaseAnalyzer):
    """
    Base class for all diffusion model analyzers.

    This class provides common utilities and methods for analyzing
    diffusion models regardless of their specific format.
    """

    # Use shared constants from diffusion_constants.py
    MODEL_PATTERNS = DIFFUSION_MODEL_PATTERNS
    CONFIG_HINTS = DIFFUSION_CONFIG_HINTS
    LORA_PATTERNS = LORA_PATTERNS
    CONFIG_FIELDS = CONFIG_FIELDS

    def __init__(self):
        """Initialize the base diffusion analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> Set[str]:
        """
        Get the file extensions supported by this analyzer.
        To be implemented by subclasses.

        Returns:
            Set of supported file extensions
        """
        raise NotImplementedError("Subclasses must implement get_supported_extensions")

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a diffusion model file.
        To be implemented by subclasses.

        Args:
            file_path: Path to the model file

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        raise NotImplementedError("Subclasses must implement analyze")

    def extract_trigger_words(self, metadata: Dict[str, Any], tensor_info: Dict[str, Any]) -> List[str]:
        """
        Extract trigger words from LoRA or Textual Inversion models.

        Args:
            metadata: Model metadata
            tensor_info: Tensor information

        Returns:
            List of extracted trigger words
        """
        trigger_words = []

        # Check metadata fields that might contain trigger words
        for field in TRIGGER_WORD_FIELDS:
            if field in metadata:
                value = metadata[field]
                if isinstance(value, str):
                    # For string values, add directly
                    trigger_words.append(value)
                elif isinstance(value, list):
                    # For lists, add each element
                    trigger_words.extend([item for item in value if isinstance(item, str)])
                elif isinstance(value, dict):
                    # For dictionaries, add values
                    trigger_words.extend([v for v in value.values() if isinstance(v, str)])

        # Look for embedding token patterns in tensor keys
        for key in tensor_info.keys():
            for pattern in EMBEDDING_TRIGGER_PATTERNS:
                match = re.search(pattern, key)
                if match and hasattr(match, 'groupdict') and match.groupdict().get('token'):
                    token = match.group('token')
                    if token not in trigger_words and token.strip():
                        trigger_words.append(token)
                        # Also add with angle brackets as it's often used that way
                        if not token.startswith('<'):
                            trigger_words.append(f"<{token}>")
            # Direct token match in the key
            if '<' in key and '>' in key:
                token_match = re.search(r'<([^>]+)>', key)
                if token_match:
                    token = token_match.group(0)  # with brackets
                    if token not in trigger_words and token.strip():
                        trigger_words.append(token)

        # Check for JSON-embedded token information
        if 'ss_metadata' in metadata and isinstance(metadata['ss_metadata'], dict):
            ss_meta = metadata['ss_metadata']
            for token_field in ['token', 'tokens', 'trigger', 'triggers', 'activation']:
                if token_field in ss_meta:
                    token_value = ss_meta[token_field]
                    if isinstance(token_value, str):
                        trigger_words.append(token_value)
                    elif isinstance(token_value, list):
                        trigger_words.extend([t for t in token_value if isinstance(t, str)])

        # Check if we have potential triggers from filename analysis
        if 'potential_trigger' in metadata and not trigger_words:
            trigger_words.append(metadata['potential_trigger'])

        if 'potential_token' in metadata and not trigger_words:
            trigger_words.append(metadata['potential_token'])

        # For LoRAs: Infer from filename if no other triggers found
        if not trigger_words and 'component_name' in metadata:
            name = metadata['component_name']

            # Extract candidate trigger from filename
            # Remove common prefixes like "lora_" and suffixes
            name = re.sub(r'^(lora_|embedding_)', '', name)
            # Replace underscores with spaces
            name = name.replace('_', ' ')
            # If it looks like a meaningful word (at least 3 chars), add it
            if len(name) >= 3:
                trigger_words.append(name)

        # For Textual Embeddings: check for common token patterns
        if not trigger_words and 'is_embedding' in metadata:
            # Extract token from metadata
            if 'token' in metadata:
                trigger_words.append(metadata['token'])
            elif 'token_name' in metadata:
                trigger_words.append(f"<{metadata['token_name']}>")

            # If still no token found, try to extract from filename
            if not trigger_words and 'component_name' in metadata:
                name = metadata['component_name']
                # Remove common prefixes
                name = re.sub(r'^(emb_|ti_|embedding_)', '', name)
                # If it looks like a word, add with angle brackets
                if len(name) >= 2:
                    trigger_words.append(f"<{name}>")

        # Remove duplicates and empty strings
        cleaned_triggers = [word.strip() for word in trigger_words if word and word.strip()]
        return list(set(cleaned_triggers))

    def is_lora_adapter(self, metadata: Dict[str, Any], tensor_info: Dict[str, Any]) -> bool:
        """
        Check if the model is a LoRA adapter based on metadata and tensor info.

        Args:
            metadata: Model metadata
            tensor_info: Tensor information

        Returns:
            True if it's a LoRA adapter, False otherwise
        """
        # Check for LoRA patterns in tensor names
        tensor_keys = list(tensor_info.keys())

        # If we have alpha parameters or lora patterns, it's LoRA
        for key in tensor_keys:
            for pattern in self.LORA_PATTERNS:
                if re.search(pattern, key):
                    return True

        # Check metadata for LoRA indicators
        if 'ss_network_dim' in metadata or 'network_dim' in metadata:
            return True

        if 'ss_network_alpha' in metadata or 'network_alpha' in metadata:
            return True

        if 'ss_network_module' in metadata and 'lora' in metadata['ss_network_module'].lower():
            return True

        # Check for typical LoRA shape patterns (pairs of rank matrices)
        shapes = metadata.get('common_shapes', {})
        for shape in shapes:
            # Many LoRAs use ranks like 4, 8, 16, 32, 64, etc.
            if isinstance(shape, tuple) and len(shape) == 2:
                if shape[1] in [4, 8, 16, 32, 64, 128, 256]:
                    matching_up_shape = None
                    for other_shape in shapes:
                        if isinstance(other_shape, tuple) and len(other_shape) == 2:
                            if other_shape[0] == shape[1]:  # Up matrix matches down matrix rank
                                matching_up_shape = other_shape
                                break

                    if matching_up_shape:
                        return True

        return False

    def is_sd_textual_embedding(self, metadata: Dict[str, Any], tensor_info: Dict[str, Any]) -> bool:
        """
        Check if the model is a Stable Diffusion textual embedding.

        Args:
            metadata: Model metadata
            tensor_info: Tensor information dictionary

        Returns:
            True if it's a SD textual embedding, False otherwise
        """
        # Check tensor keys for embedding patterns
        tensor_keys = list(tensor_info.keys())

        # Common patterns in SD textual embeddings
        embedding_patterns = [
            r'<.*>',  # Token identifiers like <concept>
            r'emb_params',
            r'string_to_param',
            r'name_to_idx',
            r'learned_embeds',
            r'clip_l',
            r'embedding'
        ]

        for key in tensor_keys:
            for pattern in embedding_patterns:
                if re.search(pattern, key):
                    return True

        # Check for typical embedding shapes
        # SD embeddings are typically small tensors with these dimensions:
        # - 768/1024 for SD1.x/SD2.x (last dimension)
        # - Very few tokens (first dimension, usually 1-5)
        for key, info in tensor_info.items():
            if 'shape' in info:
                shape = info['shape']
                if len(shape) == 2:
                    # Check if it matches embedding dimensions for SD models
                    if shape[1] in [768, 1024, 1280]:
                        # Check if first dimension is small (few tokens)
                        if shape[0] <= 10:
                            return True

        # Check metadata for embedding indicators
        if 'ss_output_name' in metadata:
            return True

        if 'ss_tag_frequency' in metadata:
            return True

        # Check for embedding vectors count
        if 'token_vectors' in metadata or 'string_to_token' in metadata:
            return True

        # Check for common metadata in textual inversion files
        embedding_meta_keys = [
            'train_params',
            'sd_checkpoint',
            'sd_checkpoint_name',
            'vectors',
            'token',
            'string_to_token',
            'name',
            'embedding_type'
        ]

        for key in metadata:
            if key in embedding_meta_keys:
                return True

        return False

    def extract_sd_embedding_metadata(self, metadata: Dict[str, Any], tensor_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information about SD textual embedding.

        Args:
            metadata: Model metadata
            tensor_info: Tensor information dictionary

        Returns:
            Dict with embedding-specific metadata
        """
        embedding_data = {
            'embedding_type': 'textual_inversion',
            'is_embedding': True
        }

        # Track if we've found a token
        found_token = False

        # Try to identify embedding token name from tensor keys
        for key in tensor_info:
            # Extract token names (patterns like <token>)
            token_match = re.search(r'<([^>]+)>', key)
            if token_match:
                embedding_data['token'] = token_match.group(0)  # With brackets
                embedding_data['token_name'] = token_match.group(1)  # Without brackets

                # Add this token to trigger_words
                embedding_data['trigger_words'] = [token_match.group(0)]
                found_token = True
                break

        # If we haven't found a token, check the filename
        if not found_token and 'component_name' in metadata:
            # Extract name without common prefixes
            name = metadata['component_name']
            clean_name = re.sub(r'^(emb_|ti_|embedding_|ZM_)', '', name)

            # For embedding, wrap in angle brackets if not already
            if not clean_name.startswith('<'):
                token = f"<{clean_name}>"
            else:
                token = clean_name

            embedding_data['token'] = token
            embedding_data['token_name'] = clean_name.strip('<>')
            embedding_data['trigger_words'] = [token]

        # Determine embedding dimensions
        for key, info in tensor_info.items():
            if 'shape' in info:
                shape = info['shape']
                if len(shape) == 2:
                    # Store vector count and dimension
                    embedding_data['vector_count'] = shape[0]
                    embedding_data['embedding_dim'] = shape[1]

                    # Determine SD version from embedding dimension
                    if shape[1] == 768:
                        embedding_data['model_compatibility'] = 'SD1.x'
                    elif shape[1] == 1024:
                        embedding_data['model_compatibility'] = 'SD2.x'
                    elif shape[1] == 1280:
                        embedding_data['model_compatibility'] = 'SDXL'
                    break

        # Extract training information if available
        if 'ss_training_comment' in metadata:
            embedding_data['training_comment'] = metadata['ss_training_comment']

        if 'train_params' in metadata:
            embedding_data['train_params'] = metadata['train_params']

        # Extract creator info
        if 'ss_creator' in metadata:
            embedding_data['creator'] = metadata['ss_creator']

        return embedding_data

    def analyze_lora(self, metadata: Dict[str, Any], tensor_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze a LoRA model and extract its information.

        Args:
            metadata: Model metadata
            tensor_info: Tensor information

        Returns:
            Tuple of (model_type, additional_metadata)
        """
        lora_info = {'is_lora': True}

        # Get LoRA rank/dimension
        if 'ss_network_dim' in metadata:
            lora_info['network_dim'] = metadata['ss_network_dim']
        elif 'network_dim' in metadata:
            lora_info['network_dim'] = metadata['network_dim']
        else:
            # Try to determine from common shapes
            shapes = metadata.get('common_shapes', {})
            for shape in shapes:
                if isinstance(shape, tuple) and len(shape) == 2:
                    for dim in [4, 8, 16, 32, 64, 128, 256]:
                        if shape[1] == dim:
                            lora_info['network_dim'] = dim
                            break
                    if 'network_dim' in lora_info:
                        break

        # Get LoRA alpha
        if 'ss_network_alpha' in metadata:
            lora_info['network_alpha'] = metadata['ss_network_alpha']
        elif 'network_alpha' in metadata:
            lora_info['network_alpha'] = metadata['network_alpha']

        # Add model name if available
        if 'ss_output_name' in metadata:
            lora_info['name'] = metadata['ss_output_name']
        elif 'component_name' in metadata:
            lora_info['name'] = metadata['component_name']

        # Extract trigger words
        trigger_words = self.extract_trigger_words(metadata, tensor_info)
        if trigger_words:
            lora_info['trigger_words'] = trigger_words

        # Determine SD version compatibility from tensor shapes
        shapes = metadata.get('common_shapes', {})
        if shapes:
            has_768_dim = False
            has_1024_dim = False
            has_1280_dim = False
            has_2048_dim = False

            for shape in shapes:
                if isinstance(shape, tuple) and len(shape) == 2:
                    if 768 in shape:
                        has_768_dim = True
                    if 1024 in shape:
                        has_1024_dim = True
                    if 1280 in shape:
                        has_1280_dim = True
                    if 2048 in shape:
                        has_2048_dim = True

            # Determine model type from shape dimensions
            if has_1280_dim or has_2048_dim:
                lora_info['model_compatibility'] = 'SDXL'
                return 'SDXL_LoRA', lora_info
            elif has_1024_dim and not has_768_dim:
                lora_info['model_compatibility'] = 'SD2.x'
                return 'SD2_LoRA', lora_info
            elif has_768_dim and not has_1024_dim:
                lora_info['model_compatibility'] = 'SD1.x'
                return 'SD1_LoRA', lora_info
            elif has_768_dim and has_1024_dim:
                lora_info['model_compatibility'] = 'SD1.x/SD2.x'
                return 'SD_LoRA', lora_info

        # Check name hints in filename if we still don't have a type
        if 'component_name' in metadata:
            name = metadata['component_name'].lower()
            if any(x in name for x in ['xl', 'sdxl', 'sd-xl']):
                lora_info['model_compatibility'] = 'SDXL'
                return 'SDXL_LoRA', lora_info
            if any(x in name for x in ['sd2', 'sd-2', 'sd 2']):
                lora_info['model_compatibility'] = 'SD2.x'
                return 'SD2_LoRA', lora_info
            if any(x in name for x in ['sd15', 'sd-15', 'sd1.5', 'sd1']):
                lora_info['model_compatibility'] = 'SD1.x'
                return 'SD1_LoRA', lora_info

        # Generic LoRA identification if we can't be more specific
        return 'LoRA', lora_info

    def determine_diffusers_model_type(
            self,
            metadata: Dict[str, Any],
            tensor_info: Dict[str, Any]
    ) -> Tuple[str, ModelConfidence]:
        """
        Determine the specific Diffusers model type from metadata.

        Args:
            metadata: Diffusers-specific metadata
            tensor_info: Tensor shapes dictionary

        Returns:
            Tuple of (model_type, confidence)
        """
        # Start with the highest confidence indicators
        # 1. Check pipeline_class_name first (from model_index.json)
        if 'pipeline_class_name' in metadata:
            pipeline = metadata['pipeline_class_name']
            if pipeline in self.CONFIG_HINTS['pipeline_class_name']:
                return self.CONFIG_HINTS['pipeline_class_name'][pipeline]

        # 2. Check architectures field directly
        if 'architectures' in metadata and metadata['architectures']:
            arch = metadata['architectures'][0]  # Use first architecture
            for arch_name, (model_type, confidence) in self.CONFIG_HINTS['architectures'].items():
                if arch_name in arch:
                    return model_type, confidence

        # 3. Special case for well-known component folders
        if 'parent_folder' in metadata:
            parent = metadata['parent_folder']

            # Check if we're in a known model subfolder
            if parent in ['controlnet', 'adapter']:
                # Check for specific controlnet/adapter types in filenames or config
                if 'preprocessor_config' in metadata:
                    processor = metadata['preprocessor_config']
                    if 'preprocessor_config' in processor:
                        name = processor['preprocessor_config'].get('_name_or_path', '').lower()
                        if 'canny' in name:
                            return 'ControlNet_Canny', ModelConfidence.HIGH
                        elif 'depth' in name:
                            return 'ControlNet_Depth', ModelConfidence.HIGH
                        elif 'pose' in name or 'openpose' in name:
                            return 'ControlNet_Pose', ModelConfidence.HIGH
                        elif 'seg' in name or 'segmentation' in name:
                            return 'ControlNet_Segmentation', ModelConfidence.HIGH
                        elif 'tile' in name:
                            return 'ControlNet_Tile', ModelConfidence.HIGH

                # Generic controlnet/adapter if specifics unknown
                if parent == 'controlnet':
                    return 'ControlNet', ModelConfidence.HIGH
                elif parent == 'adapter':
                    return 'T2IAdapter', ModelConfidence.HIGH

        # 4. Check component name (from filename and folder structure)
        # For model components, get more specific than just "UNet"
        if 'component_type' in metadata:
            component = metadata['component_type']

            # Special handling for known components
            if component == 'unet':
                # Check for SDXL vs. SD1.5/SD2.1
                if 'cross_attention_dim' in metadata:
                    cross_dim = metadata['cross_attention_dim']
                    if cross_dim == 2048 or cross_dim == 1280:
                        return 'SDXL_UNet', ModelConfidence.HIGH
                    elif cross_dim == 1024:
                        return 'SD2.x_UNet', ModelConfidence.HIGH
                    elif cross_dim == 768:
                        return 'SD1.x_UNet', ModelConfidence.HIGH

                # If it has transformer_layers_per_block, it's SDXL
                if 'transformer_layers_per_block' in metadata and metadata[
                    'transformer_layers_per_block'] == 2:
                    return 'SDXL_UNet', ModelConfidence.HIGH

                # Check for down and up block types to identify model
                if 'down_block_types' in metadata:
                    down_blocks = metadata['down_block_types']
                    # SDXL has CrossAttnDownBlock2D twice
                    if down_blocks.count('CrossAttnDownBlock2D') == 2:
                        return 'SDXL_UNet', ModelConfidence.HIGH

                    # Check for video models
                    if any('Temporal' in block for block in down_blocks):
                        return 'StableVideoDiffusion_UNet', ModelConfidence.HIGH

                # Default UNet
                return 'Diffusers_UNet', ModelConfidence.HIGH

            elif component == 'vae':
                # Check for SD version from in_channels
                if 'in_channels' in metadata:
                    if metadata['in_channels'] == 4:
                        # Standard SD VAE
                        return 'StableDiffusion_VAE', ModelConfidence.HIGH
                    elif metadata['in_channels'] == 8:
                        # Could be advanced VAE like SDXL
                        return 'SDXL_VAE', ModelConfidence.HIGH
                return 'Diffusers_VAE', ModelConfidence.HIGH

            elif component == 'text_encoder':
                # Check for CLIP vs. other encoders
                if '_class_name' in metadata.get('config', {}):
                    class_name = metadata['config']['_class_name']
                    if 'CLIPTextModel' in class_name:
                        if 'hidden_size' in metadata.get('config', {}):
                            hidden_size = metadata['config']['hidden_size']
                            if hidden_size == 768:
                                return 'SD1.x_TextEncoder', ModelConfidence.HIGH
                            elif hidden_size == 1024:
                                return 'SD2.x_TextEncoder', ModelConfidence.HIGH
                            elif hidden_size == 1280:
                                return 'SDXL_TextEncoder1', ModelConfidence.HIGH
                        return 'CLIP_TextEncoder', ModelConfidence.HIGH
                return 'Diffusers_TextEncoder', ModelConfidence.HIGH

            elif component == 'text_encoder_2':
                return 'SDXL_TextEncoder2', ModelConfidence.HIGH

            elif component == 'controlnet':
                # Already handled in the parent folder check
                return 'ControlNet', ModelConfidence.HIGH

            elif component == 'adapter':
                # Already handled in the parent folder check
                return 'T2IAdapter', ModelConfidence.HIGH

            elif component == 'prior':
                return 'DiffusionPrior', ModelConfidence.HIGH

            elif component == 'decoder':
                return 'DiffusionDecoder', ModelConfidence.HIGH

            elif component == 'image_encoder':
                return 'ImageEncoder', ModelConfidence.HIGH

            # Add other component types with their specific identifiers
            # Return format with component clearly marked
            return f"Diffusers_{component.capitalize()}", ModelConfidence.HIGH

        # 5. Check specific config fields
        for field, hints in self.CONFIG_HINTS.items():
            if field in metadata:
                value = metadata[field]
                if value in hints:
                    return hints[value]

        # 6. Check tensor names for model patterns
        tensor_keys = list(tensor_info.keys())
        for key in tensor_keys:
            for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
                # When detecting Flux models, ensure we return the correct type
                if model_type == 'Flux':
                    if re.search(pattern, key):
                        return model_type, confidence  # Return 'Flux' directly, not prefixed
                if re.search(pattern, key):
                    return model_type, confidence

        # 7. Fallback: Return generic Diffusers component if we have component info
        if 'component_type' in metadata:
            return f"Diffusers_{metadata['component_type'].capitalize()}", ModelConfidence.MEDIUM

        # 8. Last resort: check file and folder names for hints
        if 'component_name' in metadata:
            name = metadata['component_name'].lower()

            # Check for model name hints in filename
            if 'flux' in name:
                return 'Flux', ModelConfidence.MEDIUM
            elif 'sdxl' in name or 'sd-xl' in name:
                return 'SDXL_Component', ModelConfidence.MEDIUM
            elif 'sd' in name or 'stable' in name:
                return 'StableDiffusion_Component', ModelConfidence.MEDIUM
            elif 'cascade' in name:
                return 'StableCascade_Component', ModelConfidence.MEDIUM
            elif 'controlnet' in name:
                return 'ControlNet', ModelConfidence.MEDIUM

            for pattern, (model_type, _) in self.MODEL_PATTERNS.items():
                # Convert regex pattern to simple string for checking
                simple_pattern = pattern.replace(r'\.', '.').replace(r'\d+', '')
                if simple_pattern.strip('\\.') in name:
                    return model_type, ModelConfidence.LOW

        # 9. Ultimate fallback
        return 'Diffusers', ModelConfidence.LOW

    def extract_diffusers_metadata(
            self,
            file_path: Path,
            tensor_metadata: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Extract Diffusers-specific metadata from file path and metadata.

        Args:
            file_path: Path to the model file
            tensor_metadata: Metadata from tensor analysis (like safetensors)

        Returns:
            Tuple of (is_diffusers, diffusers_metadata)
        """
        metadata = {}
        is_diffusers = False

        # Store the component name (filename) for later analysis
        metadata['component_name'] = file_path.stem

        # Check parent folder for model_index.json - definitive indicator of Diffusers model
        model_index = file_path.parent / "model_index.json"
        if model_index.exists():
            try:
                with open(model_index, 'r') as f:
                    import json
                    model_index_data = json.load(f)
                metadata['model_index'] = model_index_data
                is_diffusers = True

                # Extract model type from model_index.json
                if '_class_name' in model_index_data:
                    metadata['pipeline_class_name'] = model_index_data['_class_name']
            except Exception as e:
                self.logger.warning(f"Failed to parse model_index.json: {e}")

        # Check for config file in the same directory
        config_file = file_path.parent / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    import json
                    config = json.load(f)
                metadata['config'] = config

                # Extract key configuration fields to top level
                for field in self.CONFIG_FIELDS:
                    if field in config:
                        metadata[field] = config[field]

                # If it has the _diffusers_version field, it's definitely a Diffusers model
                if '_diffusers_version' in config:
                    is_diffusers = True
                    metadata['diffusers_version'] = config['_diffusers_version']

                # Check architectures field
                if 'architectures' in config:
                    is_diffusers = True
            except Exception as e:
                self.logger.warning(f"Failed to parse config.json: {e}")

        # Check for scheduler config
        scheduler_config = file_path.parent / "scheduler" / "scheduler_config.json"
        if scheduler_config.exists():
            try:
                with open(scheduler_config, 'r') as f:
                    import json
                    scheduler_data = json.load(f)
                metadata['scheduler_config'] = scheduler_data
                is_diffusers = True
            except Exception as e:
                self.logger.warning(f"Failed to parse scheduler_config.json: {e}")

        # Check for preprocessor config (common in adapters and controlnets)
        processor_config = file_path.parent / "preprocessor_config.json"
        if processor_config.exists():
            try:
                with open(processor_config, 'r') as f:
                    import json
                    processor_data = json.load(f)
                metadata['preprocessor_config'] = processor_data
                is_diffusers = True
            except Exception as e:
                self.logger.warning(f"Failed to parse preprocessor_config.json: {e}")

        # Check for tensor names in the Diffusers format
        tensor_keys = tensor_metadata.get('tensor_shapes', {}).keys()
        for key in tensor_keys:
            for pattern in self.MODEL_PATTERNS:
                if re.search(pattern, key):
                    is_diffusers = True
                    break

            if is_diffusers:
                break

        # Check for special case: is it organized like a Diffusers repo?
        # Look for parent directory structure (unet, text_encoder, vae folders)
        parent_dir = file_path.parent
        component_dirs = ['unet', 'text_encoder', 'vae', 'scheduler', 'tokenizer']
        found_components = [d.name for d in parent_dir.parent.iterdir()
                            if d.is_dir() and d.name in component_dirs]
        if len(found_components) >= 2:
            is_diffusers = True
            metadata['diffusers_components'] = found_components

        # Extract model component type based on filename and parent folder
        if is_diffusers:
            component_name = file_path.stem
            parent_folder = file_path.parent.name

            # Store both for analysis
            metadata['component_name'] = component_name
            metadata['parent_folder'] = parent_folder

            # Determine component type from folder structure
            if parent_folder in ['unet', 'text_encoder', 'vae', 'text_encoder_2',
                                 'controlnet', 'adapter', 'prior', 'decoder',
                                 'image_encoder', 'tokenizer', 'scheduler',
                                 'feature_extractor']:
                metadata['component_type'] = parent_folder
            # Or from filename if it's a direct component file
            elif component_name in ['unet', 'text_encoder', 'vae', 'text_encoder_2',
                                    'controlnet', 'adapter', 'prior', 'decoder']:
                metadata['component_type'] = component_name

        # Try to infer trigger words from filename for LoRAs and Embeddings
        filename = file_path.stem

        # For Textual Embeddings, try to infer a token from filename
        if any(pattern in filename.lower() for pattern in ['embedding', 'token', 'concept', 'textual']):
            # Extract a potential token (remove common prefixes)
            token_name = re.sub(r'^(emb_|ti_|embedding_)', '', filename)

            # If filename might contain a token
            if len(token_name) >= 2:
                metadata['potential_token'] = f"<{token_name}>"

        # For LoRAs, extract potential trigger word from filename
        elif any(pattern in filename.lower() for pattern in ['lora', 'adapter']):
            # Remove common prefixes
            trigger_candidate = re.sub(r'^(lora_|adapter_)', '', filename)
            # Replace underscores with spaces
            trigger_candidate = trigger_candidate.replace('_', ' ')

            if len(trigger_candidate) >= 3:
                metadata['potential_trigger'] = trigger_candidate

        return is_diffusers, metadata

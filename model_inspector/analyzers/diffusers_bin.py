# model_inspector/analyzers/diffusers_bin.py
from typing import Dict, Any, Tuple, Optional, List, Set
import logging
from pathlib import Path
import re
import os
import json

from ..models.confidence import ModelConfidence
from .base_diffusion_analyzer import BaseDiffusionAnalyzer
from .bin_analyzer import BinAnalyzer


class DiffusionBinAnalyzer(BaseDiffusionAnalyzer):
    """
    Safe analyzer for diffusion models in unsafe binary formats (.pt, .pth, .ckpt).

    This analyzer uses binary inspection techniques to identify diffusion models
    without loading them into memory.
    """

    def __init__(self):
        """Initialize the safe binary diffusion model analyzer."""
        super().__init__()
        self.bin_analyzer = BinAnalyzer()

    def get_supported_extensions(self) -> set:
        """Get file extensions supported by this analyzer."""
        return {'.pt', '.pth', '.ckpt', '.checkpoint'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """Check if the file can be analyzed safely without loading."""
        # We implement a safe analyzer, but the formats themselves are unsafe
        return False

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Safely analyze a binary diffusion model file.

        Args:
            file_path: Path to the binary file

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # First use the general BinAnalyzer (safe)
        base_type, base_confidence, metadata = self.bin_analyzer.analyze(file_path)

        # Now perform diffusion-specific binary analysis
        diffusion_metadata = self._extract_diffusion_metadata(file_path)
        metadata.update(diffusion_metadata)

        # Create an empty tensor_shapes dict for compatibility with base methods
        # Since we can't safely extract actual tensor shapes from binary files
        tensor_info = {}
        metadata['tensor_shapes'] = tensor_info

        # Check for Diffusers metadata from companion files
        is_diffusers, diffusers_metadata = self.extract_diffusers_metadata(path, metadata)
        if is_diffusers:
            metadata.update(diffusers_metadata)
            model_type, confidence = self.determine_diffusers_model_type(metadata, tensor_info)
            return model_type, confidence, metadata

        # Determine if this is a diffusion model
        is_diffusion, model_type, confidence = self._identify_diffusion_type(file_path, metadata)

        # Special case for LoRA
        if metadata.get("is_lora", False):
            lora_metadata = self._extract_lora_metadata(file_path, metadata)
            metadata.update(lora_metadata)
            return lora_metadata['lora_type'], ModelConfidence.HIGH, metadata

        # Special case for embeddings
        if metadata.get("is_embedding", False):
            embedding_metadata = self._extract_embedding_metadata(file_path, metadata)
            metadata.update(embedding_metadata)
            return embedding_metadata['embedding_type'], ModelConfidence.HIGH, metadata

        if is_diffusion:
            return model_type, confidence, metadata

        # If not clearly a diffusion model, return the base analysis
        return base_type, base_confidence, metadata

    def _extract_diffusion_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract diffusion model metadata from file and context.

        Args:
            file_path: Path to the binary file

        Returns:
            Diffusion-specific metadata
        """
        metadata = {
            "safe_analysis_only": True
        }

        # Check for companion config files
        self._check_companion_files(file_path, metadata)

        # Extract information from filename
        self._analyze_filename(file_path, metadata)

        # Look for binary markers without loading the model
        self._scan_binary_patterns(file_path, metadata)

        return metadata

    def _extract_lora_metadata(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract LoRA-specific metadata based on filename and context.

        Args:
            file_path: Path to the binary file
            metadata: Current metadata dictionary

        Returns:
            LoRA-specific metadata
        """
        lora_metadata = {
            'is_lora': True
        }

        filename = Path(file_path).name.lower()

        # Try to extract network dimension from filename patterns
        dim_match = re.search(r'rank[_-](\d+)|dim[_-](\d+)|r(\d+)[_-]', filename)
        if dim_match:
            # Get the first non-None group
            for group in dim_match.groups():
                if group is not None:
                    lora_metadata['network_dim'] = int(group)
                    break

        # Try to extract alpha value from filename patterns
        alpha_match = re.search(r'alpha[_-]([\d.]+)|a([\d.]+)[_-]', filename)
        if alpha_match:
            for group in alpha_match.groups():
                if group is not None:
                    lora_metadata['network_alpha'] = float(group)
                    break

        # Extract potential trigger words from filename
        name = Path(file_path).stem
        # Remove common prefixes
        trigger_candidate = re.sub(r'^(lora_|adapter_)', '', name)
        # Replace underscores with spaces
        trigger_candidate = trigger_candidate.replace('_', ' ')

        if len(trigger_candidate) >= 3:
            lora_metadata['trigger_words'] = [trigger_candidate]

        # Determine model compatibility from filename
        if "xl" in filename or "sdxl" in filename:
            lora_metadata['model_compatibility'] = 'SDXL'
            lora_metadata['lora_type'] = 'SDXL_LoRA'
        elif "sd2" in filename or "sd-2" in filename:
            lora_metadata['model_compatibility'] = 'SD2.x'
            lora_metadata['lora_type'] = 'SD2_LoRA'
        elif "sd1" in filename or "sd1.5" in filename or "sd-15" in filename:
            lora_metadata['model_compatibility'] = 'SD1.x'
            lora_metadata['lora_type'] = 'SD1_LoRA'
        else:
            lora_metadata['lora_type'] = 'LoRA'

        return lora_metadata

    def _extract_embedding_metadata(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract embedding-specific metadata based on filename and context.

        Args:
            file_path: Path to the binary file
            metadata: Current metadata dictionary

        Returns:
            Embedding-specific metadata
        """
        embedding_metadata = {
            'is_embedding': True,
            'embedding_type': 'textual_inversion',
        }

        filename = Path(file_path).stem
        # Extract token name from filename
        # Remove common prefixes
        token_name = re.sub(r'^(emb_|ti_|embedding_|ZM_)', '', filename)

        # For embedding, wrap in angle brackets if not already
        if not token_name.startswith('<'):
            token = f"<{token_name}>"
        else:
            token = token_name

        embedding_metadata['token'] = token
        embedding_metadata['token_name'] = token_name.strip('<>')
        embedding_metadata['trigger_words'] = [token]

        # Determine model compatibility from filename
        filename_lower = filename.lower()
        if "xl" in filename_lower or "sdxl" in filename_lower:
            embedding_metadata['model_compatibility'] = 'SDXL'
            embedding_metadata['embedding_type'] = 'SDXL_TextualEmbedding'
        elif "sd2" in filename_lower or "sd-2" in filename_lower:
            embedding_metadata['model_compatibility'] = 'SD2.x'
            embedding_metadata['embedding_type'] = 'SD2_TextualEmbedding'
        elif "sd1" in filename_lower or "sd1.5" in filename_lower or "sd-15" in filename_lower:
            embedding_metadata['model_compatibility'] = 'SD1.x'
            embedding_metadata['embedding_type'] = 'SD1_TextualEmbedding'
        else:
            embedding_metadata['embedding_type'] = 'TextualEmbedding'

        return embedding_metadata

    def _check_companion_files(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Check for companion files that may contain model configuration.

        Args:
            file_path: Path to the binary file
            metadata: Metadata dict to update
        """
        # Look for common companion files
        path = Path(file_path)
        dir_path = path.parent
        stem = path.stem

        # Config files to check
        config_files = [
            dir_path / f"{stem}.yaml",
            dir_path / f"{stem}.json",
            dir_path / "config.yaml",
            dir_path / "config.json",
            dir_path / "model_index.json"
        ]

        for config_file in config_files:
            if config_file.exists():
                metadata["has_config_file"] = True
                metadata["config_path"] = str(config_file)

                # Try to extract info from JSON config
                if config_file.suffix == '.json':
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            self._analyze_config_json(config_data, metadata)
                    except Exception as e:
                        self.logger.debug(f"Error parsing config JSON: {e}")
                break

    def _analyze_config_json(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """
        Extract useful information from config JSON.

        Args:
            config: Config data
            metadata: Metadata dict to update
        """
        # Check for SD-specific config keys
        sd_keys = [
            "model.diffusion_model", "cond_stage_model", "first_stage_model",
            "unet", "vae", "text_encoder"
        ]

        for key in sd_keys:
            if key in config or any(k.startswith(key) for k in config.keys()):
                metadata["is_diffusion_model"] = True
                metadata["model_family"] = "StableDiffusion"
                break

        # Extract configuration fields defined in the constants
        for field in self.CONFIG_FIELDS:
            if field in config:
                metadata[field] = config[field]

        # Check architecture hints
        if "architectures" in config and config["architectures"]:
            metadata["architectures"] = config["architectures"]
            arch = config["architectures"][0]  # Use first architecture
            for arch_name, (model_type, _) in self.CONFIG_HINTS['architectures'].items():
                if arch_name in arch:
                    metadata["likely_model_type"] = model_type
                    break

        # Check model_type hints
        if "model_type" in config:
            model_type = config["model_type"]
            if model_type in self.CONFIG_HINTS['model_type']:
                metadata["likely_model_type"] = self.CONFIG_HINTS['model_type'][model_type][0]

        # Check pipeline_class_name hints
        if "_class_name" in config:
            pipeline = config["_class_name"]
            if pipeline in self.CONFIG_HINTS['pipeline_class_name']:
                metadata["likely_model_type"] = self.CONFIG_HINTS['pipeline_class_name'][pipeline][0]

        # Check cross-attention dimension
        if "cross_attention_dim" in config:
            cross_dim = config["cross_attention_dim"]
            if cross_dim in self.CONFIG_HINTS['cross_attention_dim']:
                metadata["likely_model_type"] = self.CONFIG_HINTS['cross_attention_dim'][cross_dim][0]

        # Check prediction_type
        if "prediction_type" in config:
            pred_type = config["prediction_type"]
            if pred_type in self.CONFIG_HINTS['prediction_type']:
                metadata["prediction_model_type"] = self.CONFIG_HINTS['prediction_type'][pred_type][0]

    def _analyze_filename(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Extract clues from the filename.

        Args:
            file_path: Path to the binary file
            metadata: Metadata dict to update
        """
        filename = Path(file_path).name.lower()

        # Store component name for later analysis
        metadata['component_name'] = Path(file_path).stem

        # Try matching patterns from MODEL_PATTERNS on filename
        for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
            if re.search(pattern, filename):
                metadata['likely_model_type'] = model_type
                metadata['model_confidence'] = confidence
                metadata['is_diffusion_model'] = True
                break

        # Check for SD version indicators
        if "sd-v1" in filename or "sd_v1" in filename or "sd1" in filename:
            metadata["sd_version"] = "1.x"
            metadata["model_family"] = "StableDiffusion"

        elif "sd-v2" in filename or "sd_v2" in filename or "sd2" in filename:
            metadata["sd_version"] = "2.x"
            metadata["model_family"] = "StableDiffusion"

        elif "sdxl" in filename or "sd-xl" in filename or "sd_xl" in filename:
            metadata["sd_version"] = "XL"
            metadata["model_family"] = "StableDiffusion"

        elif "sd3" in filename or "sd-3" in filename:
            metadata["sd_version"] = "3"
            metadata["model_family"] = "StableDiffusion3"

        # Check for component indicators
        if "unet" in filename:
            metadata["component_type"] = "unet"

        elif "vae" in filename:
            metadata["component_type"] = "vae"

        elif "text_encoder" in filename or "clip" in filename:
            metadata["component_type"] = "text_encoder"

        # Check for LoRA indicators
        if "lora" in filename:
            metadata["is_lora"] = True

        # Check for embedding indicators
        if "embedding" in filename or "textual" in filename or "ti_" in filename:
            metadata["is_embedding"] = True

    def _scan_binary_patterns(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Scan for diffusion model markers in the binary data.

        Args:
            file_path: Path to the binary file
            metadata: Metadata dict to update
        """
        # Only scan a portion of the file for efficiency
        max_scan_size = 10 * 1024 * 1024  # 10MB

        # Convert regex patterns to binary search patterns where possible
        binary_patterns = {}

        # Generate binary patterns from MODEL_PATTERNS
        for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
            # Convert simple string patterns to binary
            if not any(c in pattern for c in ['.', '*', '+', '?', '|', '(', '[', '{', '\\']):
                binary_patterns[pattern.encode('utf-8')] = (model_type, confidence)

        with open(file_path, 'rb') as f:
            # Read first chunk for scanning
            data = f.read(max_scan_size)

            # Scan for diffusion model markers
            for pattern, (model_type, confidence) in binary_patterns.items():
                if pattern in data:
                    metadata["is_diffusion_model"] = True
                    metadata["diffusion_marker_detected"] = True
                    metadata["diffusion_marker"] = pattern.decode('utf-8', errors='ignore')
                    metadata["likely_model_type"] = model_type
                    metadata["model_confidence"] = confidence
                    break

    def _identify_diffusion_type(self, file_path: str, metadata: Dict[str, Any]) -> Tuple[bool, str, ModelConfidence]:
        """
        Determine if the model is a diffusion model and its type.

        Args:
            file_path: Path to the binary file
            metadata: Collected metadata

        Returns:
            Tuple of (is_diffusion, model_type, confidence)
        """
        is_diffusion = False
        model_type = "Unknown"
        confidence = ModelConfidence.LOW

        # Check for explicit diffusion indicators
        if metadata.get("is_diffusion_model", False):
            is_diffusion = True

            # If we already determined a likely model type with confidence, use it
            if 'likely_model_type' in metadata and 'model_confidence' in metadata:
                return True, metadata['likely_model_type'], metadata['model_confidence']

            # Determine more specific type if available
            if "model_family" in metadata:
                family = metadata["model_family"]

                if family == "StableDiffusion":
                    # Add version if known
                    if "sd_version" in metadata:
                        model_type = f"StableDiffusion_{metadata['sd_version']}"
                        confidence = ModelConfidence.HIGH
                    else:
                        model_type = "StableDiffusion"
                        confidence = ModelConfidence.MEDIUM

                    # Add component if known
                    if "component_type" in metadata:
                        model_type = f"{model_type}_{metadata['component_type'].capitalize()}"

                elif family == "StableDiffusion3":
                    model_type = "StableDiffusion3"
                    confidence = ModelConfidence.HIGH

                else:
                    model_type = family
                    confidence = ModelConfidence.MEDIUM

            # If no family but we have a likely type
            elif "likely_model_type" in metadata:
                model_type = metadata["likely_model_type"]
                confidence = ModelConfidence.MEDIUM

            else:
                model_type = "DiffusionModel"
                confidence = ModelConfidence.LOW

        return is_diffusion, model_type, confidence

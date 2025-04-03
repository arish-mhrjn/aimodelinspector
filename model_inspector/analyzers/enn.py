from typing import Dict, Any, Tuple, Optional, List, Set
import struct
import logging
import json
from pathlib import Path
import re
from collections import defaultdict, Counter
import os

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class ENNAnalyzer(BaseAnalyzer):
    """
    Analyzer for EPUB Neural Codec (.enn) format files.

    The ENN format is used for neural audio codecs, which are deep learning models
    designed for high-quality audio compression and reconstruction. These models
    typically include encoders, decoders, and quantizers specialized for audio signals.

    Possible improvements:
    - Add support for extracting model hyperparameters specific to audio processing
    - Include analysis of supported audio formats and sample rates
    - Extract codec-specific parameters like bitrate, latency, and compression ratio
    - Add detection of specific neural codec architectures (EnCodec, SoundStream, etc.)
    - Deploy frequency analysis to categorize models by audio bandwidth capability
    - Add extraction of any training dataset information if available
    - Support detection of specialized audio codecs (music, speech, environment)
    - Add version compatibility checks for the ENN format as it evolves
    """

    # ENN magic number and constants
    ENN_MAGIC = b'ENNC'  # Magic bytes for the ENN format

    # Model architecture types
    ARCHITECTURES = {
        'encodec': 'EnCodec',
        'soundstream': 'SoundStream',
        'hificodec': 'HiFi-Codec',
        'audiocraft': 'AudioCraft',
        'dac': 'DAC',
        'museenc': 'MuseEnc',
        'univnet': 'UnivNet'
    }

    # Known codec configurations
    CODEC_CONFIGS = {
        'mono': 'Mono Audio',
        'stereo': 'Stereo Audio',
        'multichannel': 'Multichannel Audio',
        'speech': 'Speech Optimized',
        'music': 'Music Optimized',
        'general': 'General Purpose',
        'lq': 'Low Quality',
        'mq': 'Medium Quality',
        'hq': 'High Quality',
    }

    # Common sampling rates
    SAMPLING_RATES = [8000, 16000, 24000, 32000, 44100, 48000, 88200, 96000]

    def __init__(self):
        """Initialize the ENN analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.enn'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze an ENN file to determine its model type and metadata.

        Args:
            file_path: Path to the ENN file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid ENN file
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)

                if magic == self.ENN_MAGIC:
                    # It's an ENN file
                    f.seek(0)  # Reset position
                    metadata = self._parse_enn(f)
                else:
                    raise ValueError(f"Not a valid ENN file. Expected magic {self.ENN_MAGIC!r}, got {magic!r}")

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing ENN file {file_path}: {e}")
            raise

    def _parse_enn(self, f) -> Dict[str, Any]:
        """
        Parse an ENN format file.

        Args:
            f: Open file handle positioned at start

        Returns:
            Extracted metadata

        Raises:
            ValueError: If the format is invalid
        """
        metadata = {}

        # Read magic number
        magic = f.read(4)
        if magic != self.ENN_MAGIC:
            raise ValueError("Not a valid ENN file")

        # Read version
        version = struct.unpack('<I', f.read(4))[0]
        metadata["format_version"] = version

        # Read header size
        header_size = struct.unpack('<I', f.read(4))[0]
        metadata["header_size"] = header_size

        # Read model info
        model_info_size = struct.unpack('<I', f.read(4))[0]
        metadata["model_info_size"] = model_info_size

        # Read model configuration if available
        if model_info_size > 0:
            model_info_bytes = f.read(model_info_size)
            try:
                # Attempt to parse as JSON (common format for model info)
                model_info = json.loads(model_info_bytes.decode('utf-8'))
                metadata["model_info"] = model_info

                # Extract important fields from model_info
                self._extract_model_info(metadata, model_info)
            except json.JSONDecodeError:
                # If not JSON, store raw bytes length
                metadata["model_info_bytes_length"] = len(model_info_bytes)

        # Read number of sections
        num_sections = struct.unpack('<I', f.read(4))[0]
        metadata["num_sections"] = num_sections

        # Parse sections
        sections = []
        for i in range(num_sections):
            section = self._read_section_header(f)
            sections.append(section)

        metadata["sections"] = sections

        # Read file size information
        current_pos = f.tell()
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        metadata["file_size_bytes"] = file_size

        # Compute model weights size (file size minus header)
        weights_size = file_size - current_pos
        metadata["weights_size_bytes"] = weights_size

        # Add computed fields
        self._add_computed_fields(metadata)

        return metadata

    def _read_section_header(self, f) -> Dict[str, Any]:
        """
        Read a section header from the ENN file.

        Args:
            f: Open file handle positioned at a section header

        Returns:
            Dictionary containing section information
        """
        section = {}

        # Read section type
        section_type = struct.unpack('<I', f.read(4))[0]
        section["type"] = section_type

        # Read section name length
        name_length = struct.unpack('<I', f.read(4))[0]

        # Read section name
        if name_length > 0:
            name_bytes = f.read(name_length)
            section["name"] = name_bytes.decode('utf-8')
        else:
            section["name"] = ""

        # Read section size
        section_size = struct.unpack('<Q', f.read(8))[0]
        section["size_bytes"] = section_size

        # Map section types to human-readable names
        section_types = {
            0: "Header",
            1: "Encoder",
            2: "Decoder",
            3: "Quantizer",
            4: "Configuration",
            5: "Auxiliary Data",
            6: "Weights"
        }
        section["type_name"] = section_types.get(section_type, f"Unknown ({section_type})")

        return section

    def _extract_model_info(self, metadata: Dict[str, Any], model_info: Dict[str, Any]) -> None:
        """
        Extract relevant fields from the model info section.

        Args:
            metadata: Main metadata dictionary to update
            model_info: Parsed model info section
        """
        # Extract architecture
        if "architecture" in model_info:
            metadata["architecture"] = model_info["architecture"]

        # Extract version
        if "version" in model_info:
            metadata["model_version"] = model_info["version"]

        # Extract channels
        if "channels" in model_info:
            metadata["audio_channels"] = model_info["channels"]

        # Extract sampling rate
        if "sample_rate" in model_info:
            metadata["sample_rate"] = model_info["sample_rate"]

        # Extract bitrate information
        if "bitrate" in model_info:
            metadata["bitrate"] = model_info["bitrate"]

        # Extract codec type
        if "codec_type" in model_info:
            metadata["codec_type"] = model_info["codec_type"]

        # Extract any additional fields that might be useful
        for key in ["latency", "bandwidth", "complexity", "author", "description"]:
            if key in model_info:
                metadata[key] = model_info[key]

    def _add_computed_fields(self, metadata: Dict[str, Any]) -> None:
        """
        Add computed fields to metadata.

        Args:
            metadata: Metadata dict to augment
        """
        # Determine audio type
        if "audio_channels" in metadata:
            channels = metadata["audio_channels"]
            if channels == 1:
                metadata["audio_type"] = "Mono"
            elif channels == 2:
                metadata["audio_type"] = "Stereo"
            else:
                metadata["audio_type"] = f"Multichannel ({channels} channels)"

        # Calculate compression ratio if we have enough information
        if "file_size_bytes" in metadata:
            original_size = metadata.get("original_audio_size_bytes", 0)
            if original_size > 0:
                compression_ratio = original_size / metadata["file_size_bytes"]
                metadata["compression_ratio"] = compression_ratio

        # Determine model size category
        weights_size = metadata.get("weights_size_bytes", 0)
        if weights_size < 5 * 1024 * 1024:  # 5 MB
            metadata["model_size_category"] = "Small"
        elif weights_size < 50 * 1024 * 1024:  # 50 MB
            metadata["model_size_category"] = "Medium"
        elif weights_size < 500 * 1024 * 1024:  # 500 MB
            metadata["model_size_category"] = "Large"
        else:
            metadata["model_size_category"] = "Very Large"

        # Count sections by type
        sections = metadata.get("sections", [])
        section_counts = Counter(section["type_name"] for section in sections)
        metadata["section_counts"] = dict(section_counts)

        # Enhance with human-readable section sizes
        for section in sections:
            size_bytes = section.get("size_bytes", 0)
            if size_bytes < 1024:
                section["human_size"] = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                section["human_size"] = f"{size_bytes / 1024:.2f} KB"
            else:
                section["human_size"] = f"{size_bytes / (1024 * 1024):.2f} MB"

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Look for architecture information
        architecture = metadata.get("architecture", "")

        if architecture:
            # Clean architecture string
            arch_lower = architecture.lower()

            # Match known architectures
            for key, name in self.ARCHITECTURES.items():
                if key in arch_lower:
                    # Enhance with additional metadata if available
                    if "sample_rate" in metadata:
                        return f"{name} Neural Codec ({metadata['sample_rate']}Hz)", ModelConfidence.HIGH

                    # If we have channels
                    if "audio_channels" in metadata:
                        channels = metadata["audio_channels"]
                        if channels == 1:
                            return f"{name} Neural Codec (Mono)", ModelConfidence.HIGH
                        elif channels == 2:
                            return f"{name} Neural Codec (Stereo)", ModelConfidence.HIGH
                        else:
                            return f"{name} Neural Codec (Multichannel)", ModelConfidence.HIGH

                    # Basic name
                    return f"{name} Neural Codec", ModelConfidence.HIGH

            # If we have an architecture but couldn't match a known one
            return f"Neural Codec ({architecture})", ModelConfidence.MEDIUM

        # Check for codec_type information
        codec_type = metadata.get("codec_type", "")
        if codec_type:
            codec_lower = codec_type.lower()
            for key, name in self.CODEC_CONFIGS.items():
                if key in codec_lower:
                    return f"Neural Codec ({name})", ModelConfidence.MEDIUM

        # Check for sections to see if we have sufficient model components
        sections = metadata.get("sections", [])
        has_encoder = any(s.get("type_name") == "Encoder" for s in sections)
        has_decoder = any(s.get("type_name") == "Decoder" for s in sections)

        if has_encoder and has_decoder:
            return "Neural Codec (Encoder-Decoder)", ModelConfidence.MEDIUM
        elif has_encoder:
            return "Neural Codec (Encoder)", ModelConfidence.MEDIUM
        elif has_decoder:
            return "Neural Codec (Decoder)", ModelConfidence.MEDIUM

        # If we have some sections but couldn't categorize
        if sections:
            return "Neural Codec", ModelConfidence.LOW

        # Fall back to basic type
        return "Unknown Neural Codec", ModelConfidence.UNKNOWN

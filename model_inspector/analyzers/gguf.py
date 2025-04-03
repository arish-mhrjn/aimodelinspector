from typing import Dict, Any, Tuple, Optional, List, Set
import struct
import logging
import json
from pathlib import Path
import re
from collections import defaultdict, Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class GGUFAnalyzer(BaseAnalyzer):
    """
    Analyzer for GGUF and GGML format files.

    These formats are commonly used for quantized language models in projects
    like llama.cpp, supporting models like LLaMA, Mistral, Falcon, and others.
    """

    # GGUF magic number and constants
    GGUF_MAGIC = b'GGUF'

    # GGUF value types
    GGUF_VALUE_TYPES = {
        0: "uint8",
        1: "int8",
        2: "uint16",
        3: "int16",
        4: "uint32",
        5: "int32",
        6: "float32",
        7: "bool",
        8: "string",
        9: "array",
        10: "uint64",
        11: "int64",
        12: "float64",
    }

    # Known model architecture tokens
    MODEL_ARCHITECTURES = {
        'llama': 'LLaMA',
        'mistral': 'Mistral',
        'falcon': 'Falcon',
        'mpt': 'MPT',
        'gpt_neox': 'GPT-NeoX',
        'rwkv': 'RWKV',
        'gptj': 'GPT-J',
        'bloom': 'BLOOM',
        'qwen': 'Qwen',
        'gemma': 'Gemma',
        'phi': 'Phi',
        'stablelm': 'StableLM',
        'internlm': 'InternLM',
        'baichuan': 'Baichuan',
        'yi': 'Yi',
        'mixtral': 'Mixtral',
    }

    # Quantization methods and their readable names
    QUANT_METHODS = {
        0: "FP32",
        1: "FP16",
        2: "Q4_0",
        3: "Q4_1",
        4: "Q5_0",
        5: "Q5_1",
        6: "Q8_0",
        7: "Q8_1",
        8: "Q2_K",
        9: "Q3_K",
        10: "Q4_K",
        11: "Q5_K",
        12: "Q6_K",
        13: "Q8_K",
        14: "I8",
        15: "I16",
        16: "I32",
    }

    def __init__(self):
        """Initialize the GGUF analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.gguf', '.ggml'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a GGUF/GGML file to determine its model type and metadata.

        Args:
            file_path: Path to the GGUF file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid GGUF file
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Determine if it's GGUF or legacy GGML
            with open(file_path, 'rb') as f:
                magic = f.read(4)

                if magic == self.GGUF_MAGIC:
                    # It's a GGUF file
                    f.seek(0)  # Reset position
                    metadata = self._parse_gguf(f)
                else:
                    # Try as legacy GGML
                    f.seek(0)  # Reset position
                    metadata = self._parse_legacy_ggml(f)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing GGUF file {file_path}: {e}")
            raise

    def _parse_gguf(self, f) -> Dict[str, Any]:
        """
        Parse a GGUF format file.

        Args:
            f: Open file handle positioned at start

        Returns:
            Extracted metadata
        """
        metadata = {}

        # Read magic number
        magic = f.read(4)
        if magic != self.GGUF_MAGIC:
            raise ValueError("Not a valid GGUF file")

        # Read version
        version = struct.unpack('<I', f.read(4))[0]
        metadata["format_version"] = version

        # Read tensor and metadata counts
        if version >= 3:
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        else:  # v1 or v2
            if version == 1:
                tensor_count = struct.unpack('<I', f.read(4))[0]
                metadata_kv_count = struct.unpack('<I', f.read(4))[0]
            else:  # v2
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

        metadata["tensor_count"] = tensor_count

        # Parse metadata key-value pairs
        for _ in range(metadata_kv_count):
            key = self._read_string(f)
            value_type = struct.unpack('<I', f.read(4))[0]

            # Parse value based on type
            if value_type == 0:  # uint8
                value = struct.unpack('<B', f.read(1))[0]
            elif value_type == 1:  # int8
                value = struct.unpack('<b', f.read(1))[0]
            elif value_type == 2:  # uint16
                value = struct.unpack('<H', f.read(2))[0]
            elif value_type == 3:  # int16
                value = struct.unpack('<h', f.read(2))[0]
            elif value_type == 4:  # uint32
                value = struct.unpack('<I', f.read(4))[0]
            elif value_type == 5:  # int32
                value = struct.unpack('<i', f.read(4))[0]
            elif value_type == 6:  # float32
                value = struct.unpack('<f', f.read(4))[0]
            elif value_type == 7:  # bool
                value = bool(struct.unpack('<B', f.read(1))[0])
            elif value_type == 8:  # string
                value = self._read_string(f)
            elif value_type == 9:  # array
                array_type = struct.unpack('<I', f.read(4))[0]
                array_len = struct.unpack('<Q', f.read(8))[0]
                value = self._read_array(f, array_type, array_len)
            elif value_type == 10:  # uint64
                value = struct.unpack('<Q', f.read(8))[0]
            elif value_type == 11:  # int64
                value = struct.unpack('<q', f.read(8))[0]
            elif value_type == 12:  # float64
                value = struct.unpack('<d', f.read(8))[0]
            else:
                self.logger.warning(f"Unknown value type: {value_type}")
                continue

            metadata[key] = value

        # Add some computed fields
        self._add_computed_fields(metadata)

        return metadata

    def _read_string(self, f) -> str:
        """
        Read a length-prefixed string.

        Args:
            f: Open file handle

        Returns:
            Decoded string
        """
        length = struct.unpack('<Q', f.read(8))[0]
        string_bytes = f.read(length)
        return string_bytes.decode('utf-8')

    def _read_array(self, f, array_type: int, length: int) -> List:
        """
        Read an array of values.

        Args:
            f: Open file handle
            array_type: Type of array elements
            length: Length of the array

        Returns:
            List of values
        """
        result = []

        for _ in range(length):
            if array_type == 0:  # uint8
                value = struct.unpack('<B', f.read(1))[0]
            elif array_type == 1:  # int8
                value = struct.unpack('<b', f.read(1))[0]
            elif array_type == 2:  # uint16
                value = struct.unpack('<H', f.read(2))[0]
            elif array_type == 3:  # int16
                value = struct.unpack('<h', f.read(2))[0]
            elif array_type == 4:  # uint32
                value = struct.unpack('<I', f.read(4))[0]
            elif array_type == 5:  # int32
                value = struct.unpack('<i', f.read(4))[0]
            elif array_type == 6:  # float32
                value = struct.unpack('<f', f.read(4))[0]
            elif array_type == 7:  # bool
                value = bool(struct.unpack('<B', f.read(1))[0])
            elif array_type == 8:  # string
                value = self._read_string(f)
            else:
                break

            result.append(value)

        return result

    def _parse_legacy_ggml(self, f) -> Dict[str, Any]:
        """
        Parse a legacy GGML format file.

        Args:
            f: Open file handle positioned at start

        Returns:
            Extracted metadata (limited for legacy format)
        """
        metadata = {"format": "legacy_ggml"}

        # Legacy GGML has very limited header information
        # Try to extract what we can, but it's minimal
        try:
            # Skip magic number (32-bit)
            f.seek(4)

            # Read file version
            file_version = struct.unpack('<i', f.read(4))[0]
            metadata["file_version"] = file_version

            # Attempt to read common headers in legacy files
            # This varies by model type

            # Read a bit more to try to identify the model type
            header_data = f.read(128)

            # Look for architecture clues in binary data
            for arch_key, arch_name in self.MODEL_ARCHITECTURES.items():
                if arch_key.encode() in header_data:
                    metadata["detected_architecture"] = arch_name
                    break

        except Exception as e:
            self.logger.warning(f"Error parsing legacy GGML: {e}")

        return metadata

    def _add_computed_fields(self, metadata: Dict[str, Any]) -> None:
        """
        Add computed fields to metadata.

        Args:
            metadata: Metadata dict to augment
        """
        # Add readable names for quantization
        if 'general.quantization_version' in metadata:
            quant_version = metadata['general.quantization_version']
            if quant_version in self.QUANT_METHODS:
                metadata['quantization_method'] = self.QUANT_METHODS[quant_version]

        # Try to compute token/vocab size
        if 'tokenizer.model.vocab_size' in metadata:
            metadata['vocab_size'] = metadata['tokenizer.model.vocab_size']

        # Get context length if available
        if 'general.context_length' in metadata:
            metadata['context_length'] = metadata['general.context_length']

        # Extract embedding dimension if available
        if 'llama.embedding_length' in metadata:
            metadata['embedding_dim'] = metadata['llama.embedding_length']
        elif 'general.embedding_length' in metadata:
            metadata['embedding_dim'] = metadata['general.embedding_length']

        # Organize metadata by category
        organized = defaultdict(dict)

        for key, value in list(metadata.items()):
            if '.' in key:
                category, subkey = key.split('.', 1)
                organized[category][subkey] = value

        # Add the organized metadata back
        for category, values in organized.items():
            metadata[f"{category}_params"] = values

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Look for architecture information
        architecture = metadata.get('general.architecture')

        if architecture:
            # Clean architecture string
            arch_lower = architecture.lower()

            # Match known architectures
            for key, name in self.MODEL_ARCHITECTURES.items():
                if key in arch_lower:
                    # Check for specific variants
                    if 'context_length' in metadata:
                        context = metadata.get('context_length')
                        if context:
                            return f"{name}-{context}", ModelConfidence.HIGH

                    # Check for quantization info
                    if 'quantization_method' in metadata:
                        quant = metadata.get('quantization_method')
                        return f"{name}-{quant}", ModelConfidence.HIGH

                    # Check for model size hints
                    if 'embedding_dim' in metadata:
                        dim = metadata.get('embedding_dim')
                        if dim:
                            # Estimate model size from embedding dim
                            if dim >= 8192:
                                return f"{name}-65B+", ModelConfidence.MEDIUM
                            elif dim >= 5120:
                                return f"{name}-30B+", ModelConfidence.MEDIUM
                            elif dim >= 4096:
                                return f"{name}-7B+", ModelConfidence.MEDIUM

                    return name, ModelConfidence.HIGH

            # If we have an architecture but couldn't match a known one
            return f"Unknown-{architecture}", ModelConfidence.MEDIUM

        # Check for general tags indicating LLM type
        if 'general.name' in metadata:
            model_name = metadata['general.name']

            # Try to extract from model name
            model_name_lower = model_name.lower()
            for key, name in self.MODEL_ARCHITECTURES.items():
                if key in model_name_lower:
                    return name, ModelConfidence.MEDIUM

            # Mixtral special case (check for mixture of experts)
            if 'mixture' in model_name_lower or 'mixtral' in model_name_lower or 'moe' in model_name_lower:
                return "Mixtral-MoE", ModelConfidence.MEDIUM

            # Return the model name as the type
            return model_name, ModelConfidence.LOW

        # For legacy GGML
        if metadata.get("format") == "legacy_ggml" and "detected_architecture" in metadata:
            return metadata["detected_architecture"], ModelConfidence.MEDIUM

        # If we have tensors but can't determine type
        if metadata.get('tensor_count', 0) > 0:
            return "Unknown-LLM", ModelConfidence.LOW

        return "Unknown", ModelConfidence.UNKNOWN

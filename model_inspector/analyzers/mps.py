
from typing import Dict, Any, Tuple, List, Set
import struct
import logging
from pathlib import Path
import re
from collections import defaultdict
import json

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer

"""

Analyzer for Microsoft's Model Proto Serialization (MPS) format files.
MPS is a binary serialization format used by Microsoft for machine learning models,
especially those used with ONNX Runtime and DirectML. It stores model architecture,
weights, and metadata in a compact binary representation.

This analyzer extracts relevant model information including architecture type,
layer configurations, quantization settings, and other metadata to help identify
and catalog model files.

Potential Improvements for MPSAnalyzer:
    Expanded Format Coverage:
        - Enhance support for different MPS format versions as they may have variations in structure
        - Add handling for compressed or encrypted MPS variants
    Model Detection:
        - Build a more extensive database of model architectures and their signatures
        - Implement more sophisticated pattern matching to identify models from tensor structures
    Metadata Extraction:
        - Extract more detailed configuration parameters specific to different architectures
        - Add support for extracting attention mechanism details, activation functions, etc.
    Performance Optimization:
        - Implement selective parsing to avoid reading entire large files when only metadata is needed
        - Add caching mechanisms for repeated analyses of the same file
    Security Enhancements:
        - Add validation checks for potentially malformed MPS files
        - Implement better error handling for corrupted files
    Testing:
        - Create comprehensive test cases with real-world MPS files
        - Add unit tests for each parsing component
    Documentation:
        - Document the specific MPS format structure for reference
        - Add more detailed examples of expected outputs for different model types
    Integration:
        - Add compatibility with visualization tools to display model architecture
        - Implement export capabilities to common model description formats 

"""

# MPS file signature and format identifiers
MPS_SIGNATURE = b'MPKG'  # Common signature for MPS files

# Model architecture types that may be found in MPS files
MODEL_ARCHITECTURES = {
    'phi': 'Phi',
    'phi1_5': 'Phi-1.5',
    'phi2': 'Phi-2',
    'phi3': 'Phi-3',
    'llama': 'LLaMA',
    'mistral': 'Mistral',
    'gpt': 'GPT',
    'falcon': 'Falcon',
    'qwen': 'Qwen',
    'gemma': 'Gemma',
    'stable': 'Stable',
    'bert': 'BERT',
}

# Quantization methods potentially used
QUANT_METHODS = {
    'fp32': 'FP32',
    'fp16': 'FP16',
    'int8': 'INT8',
    'int4': 'INT4',
    'qint8': 'QINT8',
    'qint4': 'QINT4',
}
class MPSAnalyzer(BaseAnalyzer):

    def __init__(self):
        """Initialize the MPS analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> Set[str]:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.mps', '.mpk'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze an MPS file to determine its model type and metadata.

        Args:
            file_path: Path to the MPS file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid MPS file
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Check if it's an MPS file
            with open(file_path, 'rb') as f:
                magic = f.read(4)

                if magic == self.MPS_SIGNATURE:
                    # It's an MPS file
                    f.seek(0)  # Reset position
                    metadata = self._parse_mps(f)
                else:
                    raise ValueError(f"Not a valid MPS file: {file_path}")

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing MPS file {file_path}: {e}")
            raise

    def _parse_mps(self, f) -> Dict[str, Any]:
        """
        Parse an MPS format file.

        Args:
            f: Open file handle positioned at start

        Returns:
            Extracted metadata
        """
        metadata = {}

        # Read magic number
        magic = f.read(4)
        if magic != self.MPS_SIGNATURE:
            raise ValueError("Not a valid MPS file")

        # Read version
        version = struct.unpack('<I', f.read(4))[0]
        metadata["format_version"] = version

        # Parse header sections
        header_size = struct.unpack('<I', f.read(4))[0]
        metadata_section_count = struct.unpack('<I', f.read(4))[0]

        metadata["header_size"] = header_size

        # Read metadata sections
        for _ in range(metadata_section_count):
            section_id = struct.unpack('<I', f.read(4))[0]
            section_size = struct.unpack('<I', f.read(4))[0]

            # Process based on section ID
            if section_id == 1:  # Model info (common section ID)
                self._parse_model_info_section(f, section_size, metadata)
            elif section_id == 2:  # Architecture info
                self._parse_architecture_section(f, section_size, metadata)
            elif section_id == 3:  # Tokenizer info
                self._parse_tokenizer_section(f, section_size, metadata)
            else:
                # Skip unknown sections
                f.seek(section_size, 1)  # Seek relative to current position

        # Try to find tensor info
        try:
            # Seek to after header
            f.seek(header_size, 0)  # Seek from beginning
            tensor_count = struct.unpack('<I', f.read(4))[0]
            metadata["tensor_count"] = tensor_count

            # Extract tensor metadata without reading full tensors
            tensor_info = []
            for _ in range(tensor_count):
                tensor_metadata = self._read_tensor_metadata(f)
                if tensor_metadata:
                    tensor_info.append(tensor_metadata)

            if tensor_info:
                metadata["tensors"] = tensor_info

        except Exception as e:
            self.logger.debug(f"Could not extract tensor info: {e}")

        # Add some computed fields
        self._add_computed_fields(metadata)

        return metadata

    def _parse_model_info_section(self, f, section_size: int, metadata: Dict[str, Any]) -> None:
        """
        Parse the model info section.

        Args:
            f: Open file handle
            section_size: Size of the section in bytes
            metadata: Dictionary to update with parsed info
        """
        section_start = f.tell()
        section_end = section_start + section_size

        # Read model name length
        name_len = struct.unpack('<I', f.read(4))[0]
        model_name = f.read(name_len).decode('utf-8')
        metadata["model_name"] = model_name

        # Read attributes if there's more data
        if f.tell() < section_end:
            attr_count = struct.unpack('<I', f.read(4))[0]
            attributes = {}

            for _ in range(attr_count):
                key_len = struct.unpack('<I', f.read(4))[0]
                key = f.read(key_len).decode('utf-8')

                value_type = struct.unpack('<B', f.read(1))[0]
                if value_type == 0:  # string
                    val_len = struct.unpack('<I', f.read(4))[0]
                    value = f.read(val_len).decode('utf-8')
                elif value_type == 1:  # int
                    value = struct.unpack('<q', f.read(8))[0]
                elif value_type == 2:  # float
                    value = struct.unpack('<d', f.read(8))[0]
                elif value_type == 3:  # bool
                    value = bool(struct.unpack('<B', f.read(1))[0])
                else:
                    # Skip unknown value types
                    continue

                attributes[key] = value

            metadata["model_attributes"] = attributes

        # Ensure we're at the end of the section
        f.seek(section_end)

    def _parse_architecture_section(self, f, section_size: int, metadata: Dict[str, Any]) -> None:
        """
        Parse the architecture section.

        Args:
            f: Open file handle
            section_size: Size of the section in bytes
            metadata: Dictionary to update with parsed info
        """
        section_start = f.tell()
        section_end = section_start + section_size

        # Read architecture type
        arch_type_len = struct.unpack('<I', f.read(4))[0]
        arch_type = f.read(arch_type_len).decode('utf-8')
        metadata["architecture"] = arch_type

        # Read architecture parameters
        params_count = struct.unpack('<I', f.read(4))[0]
        arch_params = {}

        for _ in range(params_count):
            param_name_len = struct.unpack('<I', f.read(4))[0]
            param_name = f.read(param_name_len).decode('utf-8')

            value_type = struct.unpack('<B', f.read(1))[0]
            if value_type == 0:  # string
                val_len = struct.unpack('<I', f.read(4))[0]
                value = f.read(val_len).decode('utf-8')
            elif value_type == 1:  # int
                value = struct.unpack('<q', f.read(8))[0]
            elif value_type == 2:  # float
                value = struct.unpack('<d', f.read(8))[0]
            elif value_type == 3:  # bool
                value = bool(struct.unpack('<B', f.read(1))[0])
            else:
                # Skip unknown value types
                continue

            arch_params[param_name] = value

        metadata["architecture_params"] = arch_params

        # Ensure we're at the end of the section
        f.seek(section_end)

    def _parse_tokenizer_section(self, f, section_size: int, metadata: Dict[str, Any]) -> None:
        """
        Parse the tokenizer section.

        Args:
            f: Open file handle
            section_size: Size of the section in bytes
            metadata: Dictionary to update with parsed info
        """
        section_start = f.tell()
        section_end = section_start + section_size

        # Read tokenizer type
        tok_type_len = struct.unpack('<I', f.read(4))[0]
        tok_type = f.read(tok_type_len).decode('utf-8')
        metadata["tokenizer_type"] = tok_type

        # Read vocabulary size if present
        if f.tell() + 4 <= section_end:
            vocab_size = struct.unpack('<I', f.read(4))[0]
            metadata["vocab_size"] = vocab_size

            # Read additional tokenizer parameters if available
            if f.tell() + 4 <= section_end:
                params_count = struct.unpack('<I', f.read(4))[0]
                tok_params = {}

                for _ in range(params_count):
                    if f.tell() >= section_end:
                        break

                    param_name_len = struct.unpack('<I', f.read(4))[0]
                    param_name = f.read(param_name_len).decode('utf-8')

                    value_type = struct.unpack('<B', f.read(1))[0]
                    if value_type == 0:  # string
                        val_len = struct.unpack('<I', f.read(4))[0]
                        value = f.read(val_len).decode('utf-8')
                    elif value_type == 1:  # int
                        value = struct.unpack('<q', f.read(8))[0]
                    elif value_type == 2:  # float
                        value = struct.unpack('<d', f.read(8))[0]
                    elif value_type == 3:  # bool
                        value = bool(struct.unpack('<B', f.read(1))[0])
                    else:
                        # Skip unknown value types
                        continue

                    tok_params[param_name] = value

                metadata["tokenizer_params"] = tok_params

        # Ensure we're at the end of the section
        f.seek(section_end)

    def _read_tensor_metadata(self, f) -> Dict[str, Any]:
        """
        Read tensor metadata without reading the actual tensor data.

        Args:
            f: Open file handle

        Returns:
            Dictionary with tensor metadata
        """
        try:
            tensor_info = {}

            # Read tensor name
            name_len = struct.unpack('<I', f.read(4))[0]
            tensor_name = f.read(name_len).decode('utf-8')
            tensor_info["name"] = tensor_name

            # Read data type
            data_type = struct.unpack('<B', f.read(1))[0]
            tensor_info["data_type"] = data_type

            # Get readable data type
            if data_type == 0:
                tensor_info["data_type_name"] = "float32"
            elif data_type == 1:
                tensor_info["data_type_name"] = "float16"
            elif data_type == 2:
                tensor_info["data_type_name"] = "int32"
            elif data_type == 3:
                tensor_info["data_type_name"] = "int8"
            elif data_type == 4:
                tensor_info["data_type_name"] = "int4"
            else:
                tensor_info["data_type_name"] = f"unknown_{data_type}"

            # Read shape
            dims_count = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(dims_count):
                dim = struct.unpack('<Q', f.read(8))[0]
                shape.append(dim)

            tensor_info["shape"] = shape

            # Get data size and skip over data
            data_size = struct.unpack('<Q', f.read(8))[0]
            tensor_info["data_size"] = data_size

            # Skip over the actual tensor data
            f.seek(data_size, 1)

            return tensor_info

        except Exception as e:
            self.logger.debug(f"Error reading tensor metadata: {e}")
            return None

    def _add_computed_fields(self, metadata: Dict[str, Any]) -> None:
        """
        Add computed fields to metadata.

        Args:
            metadata: Metadata dict to augment
        """
        # Extract quantization info
        if "architecture_params" in metadata:
            arch_params = metadata["architecture_params"]

            # Look for quantization info
            for key, value in arch_params.items():
                if "quant" in key.lower() or "precision" in key.lower():
                    if isinstance(value, str):
                        quant_value = value.lower()
                        for quant_key, quant_name in self.QUANT_METHODS.items():
                            if quant_key in quant_value:
                                metadata["quantization_method"] = quant_name
                                break

            # Extract context length
            for key, value in arch_params.items():
                if "context" in key.lower() and "length" in key.lower():
                    metadata["context_length"] = value
                    break
                elif "max_seq_len" in key.lower() or "max_position" in key.lower():
                    metadata["context_length"] = value
                    break

            # Extract embedding dimension
            for key, value in arch_params.items():
                if "hidden" in key.lower() and "size" in key.lower():
                    metadata["embedding_dim"] = value
                    break
                elif "embedding" in key.lower() and "dim" in key.lower():
                    metadata["embedding_dim"] = value
                    break

            # Extract head information
            for key, value in arch_params.items():
                if "num_heads" in key.lower() or "head_count" in key.lower():
                    metadata["num_heads"] = value
                    break

        # Determine quantization from tensor info if not already found
        if "quantization_method" not in metadata and "tensors" in metadata:
            # Look at the most common tensor data type
            data_types = [tensor.get("data_type_name", "") for tensor in metadata["tensors"]]
            data_type_counts = {}
            for dt in data_types:
                if dt:
                    data_type_counts[dt] = data_type_counts.get(dt, 0) + 1

            if data_type_counts:
                most_common_type = max(data_type_counts.items(), key=lambda x: x[1])[0]
                if most_common_type == "float32":
                    metadata["quantization_method"] = "FP32"
                elif most_common_type == "float16":
                    metadata["quantization_method"] = "FP16"
                elif most_common_type == "int8":
                    metadata["quantization_method"] = "INT8"
                elif most_common_type == "int4":
                    metadata["quantization_method"] = "INT4"

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Look for architecture information
        architecture = metadata.get('architecture', '')
        model_name = metadata.get('model_name', '')

        # Check model name first
        if model_name:
            model_name_lower = model_name.lower()
            # Match known architectures in the model name
            for key, name in self.MODEL_ARCHITECTURES.items():
                if key in model_name_lower:
                    # Check for specific variants with context length
                    if 'context_length' in metadata:
                        context = metadata.get('context_length')
                        if context:
                            return f"{name}-{context}", ModelConfidence.HIGH

                    # Check for quantization info
                    if 'quantization_method' in metadata:
                        quant = metadata.get('quantization_method')
                        return f"{name}-{quant}", ModelConfidence.HIGH

                    # Return just the architecture name
                    return name, ModelConfidence.HIGH

        # Check architecture field
        if architecture:
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
                            if isinstance(dim, (int, float)):
                                if dim >= 8192:
                                    return f"{name}-65B+", ModelConfidence.MEDIUM
                                elif dim >= 5120:
                                    return f"{name}-30B+", ModelConfidence.MEDIUM
                                elif dim >= 4096:
                                    return f"{name}-7B+", ModelConfidence.MEDIUM

                    return name, ModelConfidence.HIGH

            # If we have an architecture but couldn't match a known one
            return f"Unknown-{architecture}", ModelConfidence.MEDIUM

        # Look for architecture in model attributes
        if "model_attributes" in metadata:
            attrs = metadata["model_attributes"]
            if "type" in attrs or "architecture" in attrs:
                arch_type = attrs.get("type", attrs.get("architecture", ""))
                arch_type_lower = arch_type.lower()

                for key, name in self.MODEL_ARCHITECTURES.items():
                    if key in arch_type_lower:
                        return name, ModelConfidence.MEDIUM

                return f"Unknown-{arch_type}", ModelConfidence.LOW

        # If we have tensors but can't determine type
        if metadata.get('tensor_count', 0) > 0:
            return "Unknown-LLM", ModelConfidence.LOW

        return "Unknown", ModelConfidence.UNKNOWN

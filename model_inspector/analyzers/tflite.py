from typing import Dict, Any, Tuple, List, Optional, Set
import os
import struct
import logging
from pathlib import Path
import re
import json
import tempfile

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class TFLiteAnalyzer(BaseAnalyzer):
    """
    Analyzer for TensorFlow Lite (.tflite) model files.

    TensorFlow Lite is TensorFlow's lightweight solution for mobile and
    embedded devices. This analyzer extracts metadata from .tflite files
    to identify model architecture, operations, and input/output details.
    """

    # TFLite file header signature (flatbuffers)
    TFLITE_MAGIC = b'TFL3'

    # Common model architectures used with TFLite
    COMMON_ARCHITECTURES = {
        "mobilenet": "MobileNet",
        "efficientnet": "EfficientNet",
        "inception": "Inception",
        "densenet": "DenseNet",
        "squeezenet": "SqueezeNet",
        "mnasnet": "MnasNet",
        "posenet": "PoseNet",
        "esrgan": "ESRGAN",
        "deeplabv3": "DeepLabV3",
        "ssd": "SSD",
        "yolo": "YOLO",
        "rnn": "RNN",
        "lstm": "LSTM",
        "transformer": "Transformer",
        "bert": "BERT",
    }

    # Common TFLite task types
    COMMON_TASKS = {
        "segmentation": "Segmentation",
        "detection": "Object Detection",
        "classification": "Classification",
        "recognition": "Recognition",
        "pose": "Pose Estimation",
        "style": "Style Transfer",
        "super_resolution": "Super Resolution",
        "text": "Text Processing",
        "nlp": "NLP",
    }

    # Common TFLite operations, useful for identifying model purpose
    SIGNIFICANT_OPS = {
        "CONV": "Convolutional",
        "DEPTHWISE_CONV": "Depthwise Convolutional",
        "TRANSPOSE_CONV": "Transpose Convolutional",
        "FULLY_CONNECTED": "Fully Connected",
        "LSTM": "LSTM",
        "UNIDIRECTIONAL_SEQUENCE_LSTM": "Unidirectional LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM": "Bidirectional LSTM",
        "TRANSFORMER": "Transformer",
        "SOFTMAX": "Softmax",
    }

    def __init__(self):
        """Initialize the TFLite analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.tflite'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a TensorFlow Lite model file to determine its type and extract metadata.

        Args:
            file_path: Path to the TFLite model file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid TFLite model
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Extract metadata from the TFLite model
            metadata = self._extract_tflite_metadata(file_path)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing TFLite file {file_path}: {e}")
            raise

    def _extract_tflite_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a TFLite model file.

        Args:
            file_path: Path to the TFLite model file

        Returns:
            Dictionary containing metadata

        Raises:
            ValueError: If the file is not a valid TFLite model
        """
        metadata = {
            "file_size": os.path.getsize(file_path),
            "format": "tflite",
        }

        # Check if file has valid TFLite magic number
        valid_model = self._validate_tflite_model(file_path)
        if not valid_model:
            metadata["valid_tflite"] = False
            return metadata

        metadata["valid_tflite"] = True

        # Extract binary info without dependencies on the TFLite interpreter
        self._extract_binary_metadata(file_path, metadata)

        # Look for associated metadata files
        self._find_associated_files(file_path, metadata)

        # Try to extract embedded metadata
        self._extract_embedded_metadata(file_path, metadata)

        return metadata

    def _validate_tflite_model(self, file_path: str) -> bool:
        """
        Validate if the file is a TFLite model by checking its magic number.

        Args:
            file_path: Path to the TFLite model file

        Returns:
            Boolean indicating if file appears to be a valid TFLite model
        """
        with open(file_path, 'rb') as f:
            # Read first 4 bytes
            magic = f.read(4)

            if magic == self.TFLITE_MAGIC:
                return True

            # Some TFLite models don't start with TFL3 but are still valid flatbuffers
            # Check for flatbuffer format (typically starts with 0x00, 0x00, 0x00)
            if magic.startswith(b'\x00\x00\x00'):
                # Try to read a bit further for common TFLite strings
                f.seek(0)
                header = f.read(256)
                for tfl_marker in [b'TensorFlowLite', b'tflite', b'TFLite']:
                    if tfl_marker in header:
                        return True

            return False

    def _extract_binary_metadata(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Extract metadata by examining the binary content of the TFLite file.

        Args:
            file_path: Path to the TFLite model file
            metadata: Dictionary to update with extracted information
        """
        # This is a simplified analysis without using TensorFlow libraries
        # For robust analysis, TFLite interpreter would be better but requires dependencies

        op_counts = {}
        tensor_types = set()

        try:
            with open(file_path, 'rb') as f:
                content = f.read()

                # Scan for key operation types
                for op_name in self.SIGNIFICANT_OPS:
                    count = content.count(op_name.encode())
                    if count > 0:
                        op_counts[op_name] = count

                # Look for tensor type indicators
                for tensor_type in [b'FLOAT32', b'UINT8', b'INT8', b'INT16',
                                    b'INT32', b'INT64', b'BOOL', b'STRING']:
                    if tensor_type in content:
                        tensor_types.add(tensor_type.decode('utf-8'))

                # Look for input/output tensor names and shapes
                # This is a heuristic approach rather than proper flatbuffer parsing
                # It might catch some but not all tensor information
                input_tensors = []
                output_tensors = []

                # Search for tensor names using regex on the binary data
                # Convert binary to string for regex, handling non-printable chars
                text = content.replace(b'\x00', b' ').decode('ascii', errors='replace')

                # Look for tensor names (typical format in TFLite models)
                tensor_names = re.findall(r'(?:input|Input|output|Output)_tensor(?:_\w+)*', text)
                if tensor_names:
                    for name in tensor_names:
                        if name.lower().startswith('input'):
                            input_tensors.append(name)
                        elif name.lower().startswith('output'):
                            output_tensors.append(name)

                if input_tensors:
                    metadata["input_tensors_detected"] = input_tensors
                if output_tensors:
                    metadata["output_tensors_detected"] = output_tensors

        except Exception as e:
            self.logger.debug(f"Error during binary metadata extraction: {e}")

        if op_counts:
            metadata["detected_operations"] = op_counts

            # Track operation categories
            if 'CONV' in op_counts or 'DEPTHWISE_CONV' in op_counts:
                metadata["has_convolutions"] = True

            if 'LSTM' in op_counts or 'UNIDIRECTIONAL_SEQUENCE_LSTM' in op_counts:
                metadata["has_lstm"] = True

            if 'TRANSFORMER' in op_counts:
                metadata["has_transformer"] = True

        if tensor_types:
            metadata["tensor_types"] = list(tensor_types)

            # Determine quantization
            if 'INT8' in tensor_types:
                metadata["quantization"] = "INT8"
            elif 'UINT8' in tensor_types and 'FLOAT32' not in tensor_types:
                metadata["quantization"] = "UINT8"
            elif 'FLOAT32' in tensor_types:
                metadata["quantization"] = "FLOAT32"

    def _find_associated_files(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Find and extract information from associated metadata files.

        Args:
            file_path: Path to the TFLite model file
            metadata: Dictionary to update with extracted information
        """
        model_dir = Path(file_path).parent
        model_name = Path(file_path).stem

        # Look for label files
        label_files = [
            model_dir / f"{model_name}_labels.txt",
            model_dir / "labels.txt",
            model_dir / "labelmap.txt",
            model_dir / f"{model_name}.labels",
        ]

        for label_file in label_files:
            if label_file.exists():
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        labels = [line.strip() for line in f if line.strip()]
                    metadata["labels"] = labels
                    metadata["label_count"] = len(labels)
                    metadata["label_file"] = str(label_file.name)
                    break
                except Exception as e:
                    self.logger.debug(f"Error reading label file {label_file}: {e}")

        # Look for metadata JSON files
        metadata_files = [
            model_dir / f"{model_name}_metadata.json",
            model_dir / "metadata.json",
            model_dir / f"{model_name}.json",
        ]

        for meta_file in metadata_files:
            if meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_content = json.load(f)
                    metadata["external_metadata"] = meta_content
                    break
                except Exception as e:
                    self.logger.debug(f"Error reading metadata file {meta_file}: {e}")

    def _extract_embedded_metadata(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Attempt to extract metadata embedded in the TFLite file.

        TFLite models can contain embedded metadata. This function extracts
        this metadata without requiring TensorFlow dependencies. It does this
        by looking for specific byte patterns in the file.

        Args:
            file_path: Path to the TFLite model file
            metadata: Dictionary to update with extracted information
        """
        # Look for metadata section markers in the file
        # Metadata in TFLite is stored in a TFLITE_METADATA section
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

                # Look for METADATA_BUFFER_INDEX marker
                metadata_idx = content.find(b'METADATA_BUFFER_INDEX')
                if metadata_idx > 0:
                    metadata["has_embedded_metadata"] = True

                    # Look for model description
                    desc_markers = [b"description", b"name", b"version", b"author"]
                    found_desc = []

                    for marker in desc_markers:
                        idx = content.find(marker)
                        if idx > 0:
                            # Extract string following marker (basic approach)
                            # A more robust approach would use flatbuffer parsing
                            end_idx = content.find(b'\x00', idx + len(marker))
                            if end_idx > 0:
                                found_string = content[idx:end_idx].decode('utf-8', errors='ignore')
                                found_string = re.sub(r'[^\w\s\.\-:]', '', found_string)
                                if re.search(r'[a-zA-Z]', found_string):  # Ensure it has letters
                                    found_desc.append(found_string)

                    if found_desc:
                        metadata["metadata_fragments"] = found_desc

                # Check for specific metadata schemas
                if b'ml_metadata' in content:
                    metadata["metadata_schema"] = "ml_metadata"
                elif b'TensorFlowLiteMetadata' in content:
                    metadata["metadata_schema"] = "TensorFlowLiteMetadata"

        except Exception as e:
            self.logger.debug(f"Error extracting embedded metadata: {e}")

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        if not metadata.get("valid_tflite", False):
            return "Invalid-TFLite", ModelConfidence.LOW

        model_type = "TFLite"
        confidence = ModelConfidence.MEDIUM

        # Check for model name in external metadata
        if "external_metadata" in metadata:
            ext_meta = metadata["external_metadata"]

            if "name" in ext_meta and ext_meta["name"]:
                model_name = ext_meta["name"]

                # Look for known architectures in the name
                for arch_key, arch_name in self.COMMON_ARCHITECTURES.items():
                    if arch_key.lower() in model_name.lower():
                        model_type = f"{arch_name}"
                        confidence = ModelConfidence.HIGH
                        break

                # If we didn't find a known architecture but have a name
                if confidence != ModelConfidence.HIGH:
                    model_type = model_name
                    confidence = ModelConfidence.MEDIUM

            # Check for task or model type in metadata
            for meta_key in ["task", "model_type", "type"]:
                if meta_key in ext_meta and ext_meta[meta_key]:
                    task_info = ext_meta[meta_key]

                    # Check for known task types
                    for task_key, task_name in self.COMMON_TASKS.items():
                        if task_key.lower() in task_info.lower():
                            if model_type != "TFLite":  # Already found an architecture
                                model_type += f"-{task_name}"
                            else:
                                model_type = f"TFLite-{task_name}"
                            confidence = ModelConfidence.HIGH
                            break

                    break  # Stop after finding the first task info

        # Check for labels to identify classification models
        if "labels" in metadata and model_type == "TFLite":
            label_count = metadata.get("label_count", 0)
            if label_count > 0:
                model_type = "TFLite-Classification"
                confidence = ModelConfidence.MEDIUM

                # Add label count to name
                model_type += f"-{label_count}classes"

        # Look for operation hints to determine model architecture
        if "detected_operations" in metadata and model_type == "TFLite":
            ops = metadata["detected_operations"]

            # Check model type based on operations
            if metadata.get("has_transformer", False):
                model_type = "TFLite-Transformer"
                confidence = ModelConfidence.MEDIUM

            elif metadata.get("has_lstm", False):
                model_type = "TFLite-LSTM"
                confidence = ModelConfidence.MEDIUM

            elif metadata.get("has_convolutions", False):
                model_type = "TFLite-CNN"
                confidence = ModelConfidence.MEDIUM

        # Add quantization info if available
        if "quantization" in metadata:
            model_type += f"-{metadata['quantization']}"

        # Return finalized model type and confidence
        return model_type, confidence

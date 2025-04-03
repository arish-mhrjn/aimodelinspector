"""
Module for analyzing OpenVINO IR model files (.xml/.bin pairs).

The OpenVINO Intermediate Representation (IR) format consists of two files:
1. An XML file (.xml) describing the model's topology, operations, and parameters
2. A binary file (.bin) containing the model's weights

This analyzer extracts metadata from both files to provide insights into
OpenVINO models, including input/output information, operation types,
and weights characteristics.

Potential improvements:
1. Add capability to estimate FLOPs/MACs for computational complexity analysis
2. Implement deeper layer analysis to detect specific network architectures
3. Add support for parsing precision information from IR v11+ models
4. Include model optimization hints based on the structure
5. Support for extracting custom layer information
6. Add visualization capabilities for the model graph
7. Implement performance estimation based on layer composition
"""

from typing import Dict, Any, Tuple, List, Set
import os
import struct
import logging
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from collections import Counter, defaultdict

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class OpenVINOIRAnalyzer(BaseAnalyzer):
    """
    Analyzer for OpenVINO Intermediate Representation (IR) files.

    This analyzer can process OpenVINO IR model files (.xml/.bin pairs) and extract
    important metadata including model structure, layer information, input/output
    specifications, and other relevant properties that help identify and
    characterize the model.
    """

    def __init__(self):
        """Initialize the OpenVINO IR analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.xml'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if file can be analyzed safely.

        OpenVINO IR files are considered safe for analysis.

        Args:
            file_path: Path to the OpenVINO IR XML file

        Returns:
            True as OpenVINO IR files don't pose security risks
        """
        return True

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze an OpenVINO IR model to determine its type and extract metadata.

        Args:
            file_path: Path to the OpenVINO IR XML file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the XML or BIN file doesn't exist
            ValueError: If the file is not a valid OpenVINO IR file
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"XML file not found: {file_path}")

        # Check for corresponding .bin file
        bin_path = path.with_suffix('.bin')
        if not bin_path.exists():
            raise FileNotFoundError(f"BIN file not found: {bin_path}")

        try:
            # Parse the XML file to extract metadata
            metadata = self._parse_xml(file_path)

            # Add binary file metadata
            metadata.update(self._parse_bin(str(bin_path)))

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except ET.ParseError:
            raise ValueError(f"Invalid XML file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error analyzing OpenVINO IR file {file_path}: {e}")
            raise

    def _parse_xml(self, file_path: str) -> Dict[str, Any]:
        """
        Parse an OpenVINO IR XML file to extract metadata.

        Args:
            file_path: Path to the XML file

        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "format": "openvino_ir",
            "file_path": file_path,
            "bin_file_path": str(Path(file_path).with_suffix('.bin')),
            "file_size_bytes": os.path.getsize(file_path),
            "layers": {},
            "edges": [],
            "inputs": [],
            "outputs": []
        }

        # Parse XML
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract model metadata
        if root.tag == 'net':
            # Extract model name
            if 'name' in root.attrib:
                metadata["model_name"] = root.attrib['name']

            # Extract version information
            if 'version' in root.attrib:
                metadata["ir_version"] = root.attrib['version']

            # Extract batch size if present
            if 'batch' in root.attrib:
                metadata["batch_size"] = int(root.attrib['batch'])

            # Extract precision information
            if 'precision' in root.attrib:
                metadata["precision"] = root.attrib['precision']

            # Count layer types
            layer_types = Counter()
            layers_count = 0

            # Process layers
            for layer in root.findall(".//layer"):
                layers_count += 1
                layer_id = layer.attrib.get('id', '')
                layer_name = layer.attrib.get('name', '')
                layer_type = layer.attrib.get('type', '')
                layer_precision = layer.attrib.get('precision', '')

                # Count layer type
                layer_types[layer_type] += 1

                # Store basic layer info
                metadata["layers"][layer_id] = {
                    "name": layer_name,
                    "type": layer_type,
                    "precision": layer_precision
                }

                # Extract input/output information for this layer
                for input_elem in layer.findall('./input/port'):
                    port_id = input_elem.attrib.get('id', '')

                    # Extract shape information if available
                    shape_dims = []
                    for dim in input_elem.findall('./dim'):
                        if dim.text:
                            shape_dims.append(int(dim.text))

                    if shape_dims:
                        metadata["layers"][layer_id]["input_shapes"] = metadata["layers"][layer_id].get("input_shapes",
                                                                                                        {})
                        metadata["layers"][layer_id]["input_shapes"][port_id] = shape_dims

                for output_elem in layer.findall('./output/port'):
                    port_id = output_elem.attrib.get('id', '')

                    # Extract shape information if available
                    shape_dims = []
                    for dim in output_elem.findall('./dim'):
                        if dim.text:
                            shape_dims.append(int(dim.text))

                    if shape_dims:
                        metadata["layers"][layer_id]["output_shapes"] = metadata["layers"][layer_id].get(
                            "output_shapes", {})
                        metadata["layers"][layer_id]["output_shapes"][port_id] = shape_dims

            # Store layer type statistics
            metadata["layer_types"] = dict(layer_types)
            metadata["total_layers"] = layers_count

            # Extract edges (connections between layers)
            for edge in root.findall(".//edge"):
                if 'from-layer' in edge.attrib and 'to-layer' in edge.attrib:
                    metadata["edges"].append((
                        edge.attrib.get('from-layer', ''),
                        edge.attrib.get('to-layer', '')
                    ))

            # Extract input layers
            for input_elem in root.findall(".//input"):
                if hasattr(input_elem, 'attrib') and 'name' in input_elem.attrib:
                    input_info = {
                        "name": input_elem.attrib.get('name', '')
                    }

                    # Try to extract shape information
                    shape_dims = []
                    for port in input_elem.findall('./port'):
                        for dim in port.findall('./dim'):
                            if dim.text:
                                shape_dims.append(int(dim.text))

                    if shape_dims:
                        input_info["shape"] = shape_dims

                    metadata["inputs"].append(input_info)

            # Find output layers
            for output_elem in root.findall(".//output"):
                for port in output_elem.findall('./port'):
                    if 'id' in port.attrib and 'precision' in port.attrib:
                        output_info = {
                            "id": port.attrib.get('id', ''),
                            "precision": port.attrib.get('precision', '')
                        }

                        # Extract shape if available
                        shape_dims = []
                        for dim in port.findall('./dim'):
                            if dim.text:
                                shape_dims.append(int(dim.text))

                        if shape_dims:
                            output_info["shape"] = shape_dims

                        metadata["outputs"].append(output_info)

        return metadata

    def _parse_bin(self, bin_path: str) -> Dict[str, Any]:
        """
        Extract basic metadata from the OpenVINO IR binary file.

        Args:
            bin_path: Path to the binary file

        Returns:
            Dictionary containing binary file metadata
        """
        bin_metadata = {
            "bin_file_size_bytes": os.path.getsize(bin_path),
        }

        # Try to sample a small part of the file to detect precision
        try:
            precision_hint = "unknown"
            with open(bin_path, 'rb') as f:
                # Read first 1000 values to analyze
                header = f.read(4000)  # Read 1000 potential float values

                # Check if appears to be mostly float16
                values_f16 = len(header) // 2
                values_f32 = len(header) // 4

                # Simple heuristic: if many values are very small when interpreted as float32,
                # it might be float16
                f32_values = []
                for i in range(0, min(1000, values_f32)):
                    if i * 4 + 4 <= len(header):
                        val = struct.unpack('f', header[i * 4:i * 4 + 4])[0]
                        if not (val == 0 or val == float('inf') or val == float(
                                '-inf') or val != val):  # last check is for NaN
                            f32_values.append(abs(val))

                if f32_values:
                    avg_val = sum(f32_values) / len(f32_values)
                    # Extremely small average suggests it might not be float32
                    if avg_val < 1e-30:
                        precision_hint = "likely_float16_or_int8"
                    # Reasonable average for neural network weights
                    elif 1e-4 < avg_val < 10:
                        precision_hint = "likely_float32"

            bin_metadata["weights_precision_hint"] = precision_hint

        except Exception as e:
            self.logger.warning(f"Error sampling binary file: {e}")

        return bin_metadata

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Start with model name if available
        model_name = metadata.get("model_name", "")
        layer_types = metadata.get("layer_types", {})

        # Common neural network architecture patterns
        architecture_patterns = {
            r"resnet|resnext": "ResNet",
            r"inception": "Inception",
            r"efficientnet": "EfficientNet",
            r"mobilenet": "MobileNet",
            r"squeezenet": "SqueezeNet",
            r"densenet": "DenseNet",
            r"yolo": "YOLO",
            r"ssd": "SSD",
            r"faster_rcnn|fast_rcnn|mask_rcnn": "R-CNN",
            r"vgg": "VGG",
            r"alexnet": "AlexNet",
            r"bert": "BERT",
            r"transformer": "Transformer",
            r"deeplabv3": "DeepLabV3",
            r"unet": "U-Net",
            r"facenet": "FaceNet",
            r"segnet": "SegNet",
            r"retinanet": "RetinaNet",
            r"centernet": "CenterNet",
            r"fcn": "FCN"
        }

        # Look for architecture clues in the model name
        if model_name:
            model_name_lower = model_name.lower()

            for pattern, architecture in architecture_patterns.items():
                if re.search(pattern, model_name_lower):
                    # Try to extract version numbers if present
                    version_match = re.search(r'(\d+)', model_name_lower)
                    version = version_match.group(1) if version_match else ""

                    if version:
                        return f"{architecture}-{version}", ModelConfidence.HIGH
                    else:
                        return architecture, ModelConfidence.HIGH

        # If model name didn't give us the architecture, try to infer from layers

        # Computer Vision model heuristics
        if layer_types.get("Convolution", 0) > 5:
            # Object detection models often have specific layers
            if any(lt in layer_types for lt in ["DetectionOutput", "RegionYolo", "ROIPooling"]):
                if "proposal" in model_name.lower() or "rpn" in model_name.lower() or "rcnn" in model_name.lower():
                    return "R-CNN Object Detector", ModelConfidence.MEDIUM
                elif "ssd" in model_name.lower():
                    return "SSD Object Detector", ModelConfidence.MEDIUM
                elif "yolo" in model_name.lower():
                    return "YOLO Object Detector", ModelConfidence.MEDIUM
                else:
                    return "Object Detection Model", ModelConfidence.MEDIUM

            # Segmentation models often have deconvolution/upsampling
            if layer_types.get("Deconvolution", 0) > 0 or layer_types.get("Interpolate", 0) > 0:
                return "Segmentation Model", ModelConfidence.MEDIUM

            # Check if it might be a classification model
            if layer_types.get("SoftMax", 0) > 0 and metadata.get("outputs", []):
                num_outputs = 0
                for output in metadata.get("outputs", []):
                    if "shape" in output and output["shape"]:
                        num_outputs = max(num_outputs, output["shape"][-1])

                if num_outputs > 0:
                    return f"Classification Model ({num_outputs} classes)", ModelConfidence.MEDIUM
                else:
                    return "Classification Model", ModelConfidence.MEDIUM

            # General CNN
            return "Convolutional Neural Network", ModelConfidence.MEDIUM

        # NLP model heuristics
        if layer_types.get("MatMul", 0) > 5 and layer_types.get("Add", 0) > 5 and layer_types.get("LSTM",
                                                                                                  0) + layer_types.get(
                "GRU", 0) > 0:
            return "RNN-based Language Model", ModelConfidence.MEDIUM

        if layer_types.get("MatMul", 0) > 10 and layer_types.get("Add", 0) > 10 and layer_types.get(
                "LayerNormalization", 0) > 0:
            return "Transformer-based Model", ModelConfidence.MEDIUM

        # If we have some layers but couldn't determine architecture
        if metadata.get("total_layers", 0) > 0:
            # Look at input shapes for clues
            if metadata.get("inputs"):
                # Try to determine based on input shape
                for input_info in metadata.get("inputs", []):
                    if "shape" in input_info and len(input_info["shape"]) >= 3:
                        # Typically image input with batch, height, width, channels
                        if len(input_info["shape"]) == 4:
                            return "Computer Vision Model", ModelConfidence.LOW

            return "Neural Network", ModelConfidence.LOW

        # If we couldn't determine anything specific
        return "OpenVINO IR Model", ModelConfidence.LOW

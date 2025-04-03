# model_inspector/analyzers/caffe.py
from typing import Dict, Any, Tuple, Optional, List, Set
import struct
import logging
from pathlib import Path
import os
import re
import numpy as np
from collections import defaultdict

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class CaffeModelAnalyzer(BaseAnalyzer):
    """
    Analyzer for Caffe model files (.caffemodel).

    This analyzer extracts metadata from Caffe binary model files, which are commonly
    used for computer vision and image processing models trained with the Caffe
    deep learning framework. It examines layer architecture, parameters, and other
    metadata to identify the model type and capabilities.

    Improvements:
    - Add support for parsing prototxt files when available alongside the caffemodel
    - Implement detection for common pre-trained Caffe architectures (AlexNet, VGG, etc.)
    - Add extraction of layer dimensions to better estimate model parameters
    - Improve model type detection with more signature-based identification
    """

    # Known Caffe model architectures
    MODEL_ARCHITECTURES = {
        'alexnet': 'AlexNet',
        'vgg': 'VGG',
        'googlenet': 'GoogLeNet',
        'inception': 'Inception',
        'resnet': 'ResNet',
        'squeezenet': 'SqueezeNet',
        'mobilenet': 'MobileNet',
        'fcn': 'FCN',
        'segnet': 'SegNet',
        'unet': 'U-Net',
        'faster_rcnn': 'Faster R-CNN',
        'ssd': 'SSD',
        'yolo': 'YOLO',
    }

    # Common layer types in Caffe models
    COMMON_LAYERS = {
        'conv': 'Convolution',
        'pool': 'Pooling',
        'fc': 'FullyConnected',
        'relu': 'ReLU',
        'bn': 'BatchNorm',
        'lrn': 'LRN',
        'dropout': 'Dropout',
        'softmax': 'Softmax',
        'concat': 'Concat',
        'slice': 'Slice',
        'eltwise': 'Eltwise',
        'sigmoid': 'Sigmoid',
        'tanh': 'Tanh',
    }

    def __init__(self):
        """Initialize the Caffe model analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.caffemodel'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a Caffe model file to determine its model type and metadata.

        Args:
            file_path: Path to the Caffe model file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid Caffe model
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Parse the Caffe model binary
            metadata = self._parse_caffemodel(file_path)

            # Check for associated prototxt file
            prototxt_path = self._find_prototxt(file_path)
            if prototxt_path:
                prototxt_metadata = self._parse_prototxt(prototxt_path)
                metadata.update(prototxt_metadata)

            # Add additional computed fields
            self._add_computed_fields(metadata)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing Caffe model file {file_path}: {e}")
            raise

    def _parse_caffemodel(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Caffe model binary file.

        Args:
            file_path: Path to the Caffe model file

        Returns:
            Extracted metadata
        """
        metadata = {
            "format": "caffemodel",
            "layer_count": 0,
            "layers": [],
            "parameter_count": 0,
            "blob_count": 0
        }

        try:
            # Caffe models use Google Protocol Buffers format
            # We'll extract basic statistics without deep parsing
            file_size = os.path.getsize(file_path)
            metadata["file_size"] = file_size

            # Count different types of layers by examining binary patterns
            with open(file_path, 'rb') as f:
                content = f.read()

                # Count layer signatures
                layer_counts = defaultdict(int)
                for layer_key in self.COMMON_LAYERS:
                    # Look for layer type strings in the binary
                    pattern = re.compile(rf'{layer_key.lower()}[\w_]*'.encode(), re.IGNORECASE)
                    matches = pattern.findall(content)
                    if matches:
                        layer_counts[self.COMMON_LAYERS[layer_key]] = len(matches)

                # Estimate number of parameters based on file size
                # Rough estimate: 4 bytes per parameter (float32)
                estimated_params = file_size / 4
                metadata["estimated_parameters"] = int(estimated_params)

                # Create layer list from counts
                layers = []
                for layer_type, count in layer_counts.items():
                    layers.append({
                        "type": layer_type,
                        "count": count
                    })

                metadata["layer_types"] = layers
                metadata["layer_count"] = sum(count for _, count in layer_counts.items())

                # Look for known architecture names
                for arch_key, arch_name in self.MODEL_ARCHITECTURES.items():
                    if arch_key.encode() in content:
                        metadata["architecture_hint"] = arch_name
                        break

        except Exception as e:
            self.logger.warning(f"Error extracting details from caffemodel: {e}")

        return metadata

    def _find_prototxt(self, model_path: str) -> Optional[str]:
        """
        Find associated prototxt file for the model.

        Args:
            model_path: Path to the Caffe model file

        Returns:
            Path to prototxt file if found, None otherwise
        """
        base_path = os.path.splitext(model_path)[0]

        # Try common prototxt naming patterns
        candidates = [
            f"{base_path}.prototxt",
            f"{base_path}_deploy.prototxt",
            f"{base_path}.proto",
            os.path.join(os.path.dirname(model_path), "deploy.prototxt")
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return None

    def _parse_prototxt(self, prototxt_path: str) -> Dict[str, Any]:
        """
        Parse a Caffe prototxt file.

        Args:
            prototxt_path: Path to the prototxt file

        Returns:
            Extracted metadata
        """
        metadata = {
            "has_prototxt": True,
            "prototxt_path": prototxt_path
        }

        try:
            # Simple text-based parsing of prototxt file
            with open(prototxt_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Extract model name if available
                name_match = re.search(r'name:\s*[\'"]([^\'"]+)[\'"]', content)
                if name_match:
                    metadata["model_name"] = name_match.group(1)

                # Count layers defined in prototxt
                layer_count = len(re.findall(r'\blayer\s*{', content))
                metadata["prototxt_layer_count"] = layer_count

                # Extract input dimensions if available
                input_dim_matches = re.findall(r'input_dim:\s*(\d+)', content)
                if len(input_dim_matches) >= 4:  # NCHW format
                    metadata["input_dimensions"] = {
                        "batch_size": int(input_dim_matches[0]),
                        "channels": int(input_dim_matches[1]),
                        "height": int(input_dim_matches[2]),
                        "width": int(input_dim_matches[3])
                    }

                # Check for task type hints
                if 'loss' in content.lower():
                    metadata["task_hint"] = "Training"
                else:
                    metadata["task_hint"] = "Inference"

                # Check for common task types
                if re.search(r'accuracy|softmax|SoftmaxWithLoss', content):
                    metadata["likely_task"] = "Classification"
                elif re.search(r'DetectionOutput|PriorBox|NormalizedBBox', content):
                    metadata["likely_task"] = "Object Detection"
                elif re.search(r'Deconv|Upsample|Interp', content):
                    metadata["likely_task"] = "Segmentation"

        except Exception as e:
            self.logger.warning(f"Error parsing prototxt file: {e}")

        return metadata

    def _add_computed_fields(self, metadata: Dict[str, Any]) -> None:
        """
        Add computed fields to metadata.

        Args:
            metadata: Metadata dict to augment
        """
        # Determine if the model is likely pre-trained or custom
        if "model_name" in metadata:
            for arch_key, arch_name in self.MODEL_ARCHITECTURES.items():
                if arch_key.lower() in metadata["model_name"].lower():
                    metadata["likely_architecture"] = arch_name
                    break

        # Estimate model complexity
        if "estimated_parameters" in metadata:
            params = metadata["estimated_parameters"]
            if params < 5_000_000:
                metadata["complexity"] = "Low"
            elif params < 50_000_000:
                metadata["complexity"] = "Medium"
            else:
                metadata["complexity"] = "High"

        # Determine if the model has common CNN architecture layers
        layer_types = [layer["type"] for layer in metadata.get("layer_types", [])]
        if "Convolution" in layer_types:
            metadata["is_cnn"] = True

        # Estimate if model is for classification based on final layers
        if "Softmax" in layer_types or "layer_types" in metadata and any(
                layer["type"] == "FullyConnected" for layer in metadata["layer_types"]):
            metadata["likely_classification"] = True

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check if architecture was directly identified
        if "architecture_hint" in metadata:
            return metadata["architecture_hint"], ModelConfidence.HIGH

        # Check if we found architecture in model name
        if "likely_architecture" in metadata:
            model_type = metadata["likely_architecture"]

            # Add task type if available
            if "likely_task" in metadata:
                model_type = f"{model_type}-{metadata['likely_task']}"

            return model_type, ModelConfidence.MEDIUM

        # Check for task type
        if "likely_task" in metadata:
            return f"Caffe-{metadata['likely_task']}", ModelConfidence.MEDIUM

        # Check model structure for clues
        if metadata.get("is_cnn", False):
            return "Caffe-CNN", ModelConfidence.LOW

        if metadata.get("likely_classification", False):
            return "Caffe-Classifier", ModelConfidence.LOW

        # Fallback based on input dimensions if available
        if "input_dimensions" in metadata:
            dims = metadata["input_dimensions"]
            # Check if it might be an image model
            if dims["channels"] in [1, 3, 4] and dims["height"] > 10 and dims["width"] > 10:
                return "Caffe-ImageModel", ModelConfidence.LOW

        return "Caffe-Unknown", ModelConfidence.UNKNOWN


    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely without security risks.

        Args:
            file_path: Path to the model file

        Returns:
            True if the file can be analyzed safely, False otherwise
        """
        # Caffe models use Protocol Buffers format which is generally safe to parse
        return True

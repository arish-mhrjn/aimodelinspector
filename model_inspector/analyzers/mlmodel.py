from typing import Dict, Any, Tuple, Set, Optional
import os
import struct
import logging
import re
from pathlib import Path
import json
import plistlib

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class MLModelAnalyzer(BaseAnalyzer):
    """
    Analyzer for Apple Core ML model files (.mlmodel).

    Core ML is Apple's framework for machine learning models that can be deployed
    on Apple devices like iOS, macOS, watchOS, and tvOS.
    """

    # Magic number for compiled Core ML model files
    MLMODEL_MAGIC = b'mlmodel'

    # Model type constants from Core ML specification
    ML_MODEL_TYPES = {
        "neuralNetworkClassifier": "Neural Network Classifier",
        "neuralNetwork": "Neural Network",
        "neuralNetworkRegressor": "Neural Network Regressor",
        "treeEnsembleClassifier": "Tree Ensemble Classifier",
        "treeEnsembleRegressor": "Tree Ensemble Regressor",
        "glmClassifier": "Generalized Linear Classifier",
        "glmRegressor": "Generalized Linear Regressor",
        "supportVectorClassifier": "Support Vector Classifier",
        "supportVectorRegressor": "Support Vector Regressor",
        "itemSimilarityRecommender": "Item Similarity Recommender",
        "mlProgram": "ML Program",
        "customModel": "Custom Model",
        "soundAnalysisPreprocessing": "Sound Analysis Preprocessing",
        "visionFeaturePrint": "Vision Feature Print",
    }

    # Common model architectures in Core ML
    COMMON_ARCHITECTURES = {
        "mobilenet": "MobileNet",
        "resnet": "ResNet",
        "efficientnet": "EfficientNet",
        "vision_transformer": "Vision Transformer",
        "yolo": "YOLO",
        "bert": "BERT",
        "gpt": "GPT",
        "ssd": "SSD",
        "unet": "UNet",
        "inception": "Inception",
        "vgg": "VGG",
        "densenet": "DenseNet",
    }

    def __init__(self):
        """Initialize the MLModel analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.mlmodel'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a Core ML model file to determine its model type and metadata.

        Args:
            file_path: Path to the Core ML model file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid Core ML model
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if this is a model package (.mlpackage) directory
        if path.is_dir() and path.suffix == ".mlpackage":
            # Look for model.mlmodel inside the package
            model_path = path / "Data" / "com.apple.CoreML" / "model.mlmodel"
            if model_path.exists():
                file_path = str(model_path)
            else:
                raise ValueError(f"Cannot find model.mlmodel inside package: {file_path}")

        try:
            # Extract metadata from the Core ML model
            metadata = self._extract_mlmodel_metadata(file_path)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing Core ML model file {file_path}: {e}")
            raise

    def _extract_mlmodel_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a Core ML model file.

        Args:
            file_path: Path to the Core ML model file

        Returns:
            Dictionary containing metadata

        Raises:
            ValueError: If the file is not a valid Core ML model
        """
        metadata = {
            "file_size": os.path.getsize(file_path),
            "format": "coreml_model"
        }

        # Try to identify if the file is a compiled or spec file
        with open(file_path, 'rb') as f:
            header = f.read(32)  # Read first 32 bytes to check format

            if self.MLMODEL_MAGIC in header:
                metadata["format_type"] = "compiled_mlmodel"
            else:
                metadata["format_type"] = "spec_mlmodel"

            # Use the same file handle to read the first chunk of data
            f.seek(0)
            model_data = f.read(8192)  # Read 8KB to look for model type indicators

            # Look for plist headers
            if b'<?xml' in model_data and b'plist' in model_data:
                metadata["contains_plist"] = True

            # Check for protobuf markers
            if b'\n\x0b' in model_data and b'type' in model_data:
                metadata["contains_protobuf"] = True

            # Check for common architecture identifiers
            for arch_key, arch_name in self.COMMON_ARCHITECTURES.items():
                if arch_key.encode() in model_data:
                    metadata["architecture_hint"] = arch_name
                    break

        # Try to extract model specification
        try:
            self._extract_model_spec(file_path, metadata)
        except Exception as e:
            self.logger.debug(f"Could not extract model spec: {e}")

        # Check for companion files
        model_dir = Path(file_path).parent
        model_name = Path(file_path).stem

        # Look for model metadata files
        metadata_candidates = [
            model_dir / f"{model_name}.json",
            model_dir / "metadata.json",
            model_dir / "model_metadata.json",
        ]

        for meta_file in metadata_candidates:
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        model_metadata = json.load(f)
                        metadata["external_metadata"] = model_metadata
                except (json.JSONDecodeError, IOError):
                    pass

        # If we're in a package, try to find weights.bin file size
        if "mlpackage" in file_path:
            weights_path = Path(file_path).parent.parent / "weights" / "weight.bin"
            if weights_path.exists():
                metadata["weights_size"] = os.path.getsize(weights_path)

        return metadata

    def _extract_model_spec(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Attempt to extract model specification from mlmodel file.

        Args:
            file_path: Path to the Core ML model file
            metadata: Dictionary to update with spec information
        """
        # This is a basic implementation that extracts just the model type
        # A full implementation would require CoreML protobuf definitions
        model_type = None

        # Try to find model type by scanning the file content
        with open(file_path, 'rb') as f:
            content = f.read()

            # Check for modelType indicators in the file
            for type_key in self.ML_MODEL_TYPES.keys():
                pattern = f"modelType{type_key}".encode()
                if pattern in content:
                    model_type = type_key
                    break

                # Alternative form: type_key directly
                if type_key.encode() in content:
                    model_type = type_key
                    break

        if model_type:
            metadata["model_type"] = model_type
            metadata["model_type_readable"] = self.ML_MODEL_TYPES.get(model_type, model_type)

        # Try to extract plist metadata if present
        try:
            self._extract_plist_metadata(file_path, metadata)
        except Exception as e:
            self.logger.debug(f"Could not extract plist metadata: {e}")

    def _extract_plist_metadata(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Attempt to extract plist metadata from mlmodel file.

        Args:
            file_path: Path to the Core ML model file
            metadata: Dictionary to update with plist information
        """
        # Look for plist data in the file
        with open(file_path, 'rb') as f:
            content = f.read()

            # Find start of plist (<?xml)
            plist_start = content.find(b'<?xml')

            if plist_start >= 0:
                # Find end of plist (</plist>)
                plist_data = content[plist_start:]
                plist_end = plist_data.find(b'</plist>') + 8  # Length of </plist>

                if plist_end > 8:  # Make sure we found end tag
                    plist_content = plist_data[:plist_end]

                    # Parse plist
                    try:
                        plist_dict = plistlib.loads(plist_content)

                        # Extract relevant metadata
                        if isinstance(plist_dict, dict):
                            # Look for description
                            if "description" in plist_dict:
                                metadata["description"] = plist_dict["description"]

                            # Look for author
                            if "author" in plist_dict:
                                metadata["author"] = plist_dict["author"]

                            # Look for license
                            if "license" in plist_dict:
                                metadata["license"] = plist_dict["license"]

                            # Other metadata fields
                            for key in ["version", "shortDescription", "inputDescription", "outputDescription"]:
                                if key in plist_dict:
                                    metadata[key.lower()] = plist_dict[key]
                    except Exception:
                        # Failed to parse plist, just extract raw metadata
                        for field in ["description", "author", "license", "version"]:
                            pattern = f"{field}</key><string>(.+?)</string>".encode()
                            match = re.search(pattern, plist_content)
                            if match:
                                metadata[field] = match.group(1).decode('utf-8')

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # First check for explicitly defined model type
        if "model_type" in metadata:
            model_type = metadata["model_type"]
            readable_type = metadata.get("model_type_readable", self.ML_MODEL_TYPES.get(model_type, model_type))

            # See if we can determine the architecture
            if "architecture_hint" in metadata:
                return f"{metadata['architecture_hint']}-{readable_type}", ModelConfidence.HIGH

            return readable_type, ModelConfidence.HIGH

        # Check for architecture hints
        if "architecture_hint" in metadata:
            return metadata["architecture_hint"], ModelConfidence.MEDIUM

        # Check external metadata
        if "external_metadata" in metadata:
            ext_meta = metadata["external_metadata"]

            # Look for model type or name
            if "model_type" in ext_meta:
                return ext_meta["model_type"], ModelConfidence.MEDIUM

            if "name" in ext_meta and ext_meta["name"] != "":
                for arch_key, arch_name in self.COMMON_ARCHITECTURES.items():
                    if arch_key.lower() in ext_meta["name"].lower():
                        return arch_name, ModelConfidence.MEDIUM

                return ext_meta["name"], ModelConfidence.LOW

        # Check for description fields for clues
        if "description" in metadata:
            desc = metadata["description"].lower()

            # Check for architecture mentions
            for arch_key, arch_name in self.COMMON_ARCHITECTURES.items():
                if arch_key.lower() in desc:
                    return arch_name, ModelConfidence.LOW

            # Look for task type mentions
            for task in ["classification", "detection", "segmentation", "generation"]:
                if task in desc:
                    return f"CoreML-{task.title()}", ModelConfidence.LOW

        # Fallback based on format
        return "CoreML-Model", ModelConfidence.LOW

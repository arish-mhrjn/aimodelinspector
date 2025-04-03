from typing import Dict, Any, Tuple, List, Optional, Set
import logging
import os
import re
import plistlib
import zipfile
import json
import struct
from pathlib import Path

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer
from ..sandbox import Sandbox
from ..models.permissions import Permission


class CoreMLAnalyzer(BaseAnalyzer):
    """
    Analyzer for CoreML (.mlmodel) files.

    This analyzer can identify various CoreML models, including neural networks,
    tree ensembles, and other ML models used in Apple's ecosystem.
    """

    # CoreML model type strings
    MODEL_TYPES = {
        'neuralNetwork': 'NeuralNetwork',
        'neuralNetworkRegressor': 'NeuralNetworkRegressor',
        'neuralNetworkClassifier': 'NeuralNetworkClassifier',
        'treeEnsembleRegressor': 'TreeEnsembleRegressor',
        'treeEnsembleClassifier': 'TreeEnsembleClassifier',
        'glmRegressor': 'GLMRegressor',
        'glmClassifier': 'GLMClassifier',
        'supportVectorRegressor': 'SVR',
        'supportVectorClassifier': 'SVC',
        'kNearestNeighborsClassifier': 'KNN',
        'itemSimilarityRecommender': 'Recommender',
        'arrayFeatureExtractor': 'FeatureExtractor',
        'nonMaximumSuppression': 'NonMaximumSuppression',
        'visionFeaturePrint': 'VisionFeaturePrint',
        'soundAnalysisPreprocessing': 'SoundAnalysis',
        'linkedModel': 'LinkedModel',
        'customModel': 'CustomModel',
        'mlProgram': 'MLProgram',
    }

    # Task type identifiers
    TASK_TYPES = {
        'classifier': 'Classification',
        'regressor': 'Regression',
        'transformer': 'Transformation',
        'featureExtractor': 'FeatureExtraction',
        'imageProcessor': 'ImageProcessing',
        'audioProcessor': 'AudioProcessing',
        'recommender': 'Recommendation',
        'custom': 'Custom',
    }

    def __init__(self):
        """Initialize the CoreML analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.mlmodel'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        CoreML files are generally safe to analyze without executing code.

        Args:
            file_path: Path to the file

        Returns:
            True as CoreML files are generally safe
        """
        return True

    def analyze(
            self,
            file_path: str,
            sandbox: Optional[Sandbox] = None
    ) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a CoreML model file.

        Args:
            file_path: Path to the CoreML file
            sandbox: Optional sandbox for safety (not required for CoreML)

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Ensure we have read permission if sandbox provided
            if sandbox:
                sandbox.check_format_permission('.mlmodel', Permission.READ_FILE)

            # Extract metadata
            metadata = self._extract_coreml_metadata(file_path)

            # Determine model type
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing CoreML file {file_path}: {e}")
            raise

    def _extract_coreml_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a CoreML file.

        Args:
            file_path: Path to the CoreML file

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
            'format': 'coreml'
        }

        try:
            # Try with coremltools if available
            import coremltools as ct
            model = ct.models.MLModel(file_path)

            # Extract basic info
            metadata['spec'] = {}
            spec = model.get_spec()

            # Extract model description
            if spec.description:
                metadata['description'] = {
                    'name': spec.description.name,
                    'short_description': spec.description.shortDescription,
                    'author': spec.description.author,
                    'license': spec.description.license,
                    'version': spec.description.versionString,
                }

            # Determine model type
            for model_type in self.MODEL_TYPES.keys():
                if spec.HasField(model_type):
                    metadata['model_type'] = model_type

                    # Get specific model info
                    model_spec = getattr(spec, model_type)

                    # Extract model-specific details
                    if model_type.startswith('neuralNetwork'):
                        # Neural network specific metadata
                        layers = []
                        for layer in model_spec.layers:
                            layers.append({
                                'name': layer.name,
                                'type': layer.WhichOneof('layer')
                            })
                        metadata['layers'] = layers

            # Extract input/output info
            metadata['inputs'] = []
            for input_feature in spec.description.input:
                input_type = input_feature.type.WhichOneof('Type')
                input_info = {
                    'name': input_feature.name,
                    'type': input_type,
                }

                # Add shape information for tensor inputs
                if input_type == 'multiArrayType':
                    input_info['shape'] = list(input_feature.type.multiArrayType.shape)
                    input_info['dataType'] = str(input_feature.type.multiArrayType.dataType)

                metadata['inputs'].append(input_info)

            metadata['outputs'] = []
            for output_feature in spec.description.output:
                output_type = output_feature.type.WhichOneof('Type')
                output_info = {
                    'name': output_feature.name,
                    'type': output_type,
                }

                # Add shape information for tensor outputs
                if output_type == 'multiArrayType':
                    output_info['shape'] = list(output_feature.type.multiArrayType.shape)
                    output_info['dataType'] = str(output_feature.type.multiArrayType.dataType)

                metadata['outputs'].append(output_info)

        except ImportError:
            # Fallback to file analysis if coremltools not available
            metadata.update(self._analyze_coreml_file(file_path))

        return metadata

    def _analyze_coreml_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a CoreML file without using coremltools.

        Args:
            file_path: Path to the CoreML file

        Returns:
            Metadata dictionary
        """
        result = {}

        # CoreML models are zip files containing model.mlmodel and other files
        try:
            if zipfile.is_zipfile(file_path):
                with zipfile.ZipFile(file_path) as zipf:
                    # List files in the archive
                    result['contained_files'] = zipf.namelist()

                    # Check for model.mlmodel
                    if 'model.mlmodel' in zipf.namelist():
                        # It's a compiled model
                        result['is_compiled'] = True

                    # Check for Info.plist
                    if 'Info.plist' in zipf.namelist():
                        with zipf.open('Info.plist') as f:
                            plist = plistlib.load(f)
                            result['plist_info'] = plist

                    # Look for metadata.json
                    if 'metadata.json' in zipf.namelist():
                        with zipf.open('metadata.json') as f:
                            try:
                                metadata_json = json.load(f)
                                result['metadata_json'] = metadata_json
                            except json.JSONDecodeError:
                                pass
            else:
                # It might be an uncompiled .mlmodel file
                result['is_compiled'] = False

                # Try to find model type by checking for signatures
                with open(file_path, 'rb') as f:
                    content = f.read(4096)  # Read first 4KB

                    # Look for model type signatures
                    for model_type in self.MODEL_TYPES.keys():
                        if model_type.encode() in content:
                            result['detected_model_type'] = model_type
                            break

        except Exception as e:
            self.logger.warning(f"Error in basic CoreML analysis: {e}")

        return result

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Direct model type from coremltools
        if 'model_type' in metadata:
            model_type = metadata['model_type']

            # Map to readable model type
            if model_type in self.MODEL_TYPES:
                return self.MODEL_TYPES[model_type], ModelConfidence.HIGH
            else:
                return f"CoreML-{model_type}", ModelConfidence.HIGH

        # Detected model type from basic file analysis
        if 'detected_model_type' in metadata:
            model_type = metadata['detected_model_type']
            if model_type in self.MODEL_TYPES:
                return self.MODEL_TYPES[model_type], ModelConfidence.MEDIUM

        # Try to infer from metadata.json
        if 'metadata_json' in metadata:
            json_data = metadata['metadata_json']

            # Check for task type
            if 'task_type' in json_data:
                task = json_data['task_type']
                if task in self.TASK_TYPES:
                    return f"CoreML-{self.TASK_TYPES[task]}", ModelConfidence.MEDIUM

            # Check for model type
            if 'model_type' in json_data:
                model_type = json_data['model_type']
                return f"CoreML-{model_type}", ModelConfidence.MEDIUM

        # If we have input/output info, make a guess
        if 'inputs' in metadata and metadata['inputs']:
            inputs = metadata['inputs']

            # Check for image inputs (common in vision models)
            for input_info in inputs:
                if input_info.get('type') == 'imageType':
                    return 'CoreML-VisionModel', ModelConfidence.MEDIUM

            # Check for neural network layers
            if 'layers' in metadata and metadata['layers']:
                return 'CoreML-NeuralNetwork', ModelConfidence.MEDIUM

        # Default to generic CoreML model
        return 'CoreML', ModelConfidence.LOW

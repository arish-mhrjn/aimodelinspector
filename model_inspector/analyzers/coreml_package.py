from typing import Dict, Any, Tuple, Optional, Set
import logging
import os
import json
import plistlib
import zipfile
from pathlib import Path

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer
from ..sandbox import Sandbox
from ..models.permissions import Permission

"""
CoreML Package (.mlpackage) Analyzer Module.

This module provides analysis capabilities for Apple's CoreML Package format,
which is a directory structure containing model files, weights, and metadata.
CoreML Package format (.mlpackage) is Apple's newer packaging format that 
supersedes the older .mlmodel format, offering more flexibility for complex
models like neural networks with multiple components.

Potential improvements:
1. Add operation counting to estimate model complexity
2. Extract more detailed layer information when possible
3. Improve detection of model families (vision, NLP, etc.) based on operation types
4. Better handling of non-standard CoreML packages
5. Support for extracting feature descriptions and metadata
6. Integration with coremltools when available for deeper analysis
7. Handle MIL (Model Intermediate Language) programs within .mlpackage files
"""


class CoreMLPackageAnalyzer(BaseAnalyzer):
    """
    Analyzer for CoreML Package (.mlpackage) files.

    The CoreML Package format is a directory structure containing:
    - model.mil: Model definition in Model Intermediate Language
    - Data/: Directory containing model weights
    - Manifest.json: File describing model structure and components
    - Info.plist: Metadata about the model
    - NeuroML/Network.json: Neural network architecture information

    This analyzer can identify various CoreML models packaged in the .mlpackage
    format, including neural networks, transformers, and specialized models for
    vision, NLP, and other ML tasks used in Apple's ecosystem.
    """

    # Common model types found in CoreML packages
    MODEL_TYPES = {
        'neuralNetwork': 'NeuralNetwork',
        'neuralNetworkRegressor': 'NeuralNetworkRegressor',
        'neuralNetworkClassifier': 'NeuralNetworkClassifier',
        'mlProgram': 'MLProgram',
        'transformer': 'Transformer',
        'tabTransformer': 'TabTransformer',
        'visionFeaturePrint': 'VisionFeaturePrint',
        'soundAnalysisPreprocessing': 'SoundAnalysis',
        'wordTagger': 'WordTagger',
        'textClassifier': 'TextClassifier',
        'berTextClassifier': 'BERTTextClassifier',
        'berWordTagger': 'BERTWordTagger',
    }

    # Task type identifiers
    TASK_TYPES = {
        'classification': 'Classification',
        'regression': 'Regression',
        'objectDetection': 'ObjectDetection',
        'segmentation': 'Segmentation',
        'featureExtraction': 'FeatureExtraction',
        'sequenceClassification': 'SequenceClassification',
        'tokenClassification': 'TokenClassification',
        'textGeneration': 'TextGeneration',
        'questionAnswering': 'QuestionAnswering',
        'audioClassification': 'AudioClassification',
    }

    def __init__(self):
        """Initialize the CoreML Package analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.mlpackage'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        CoreML Package files are generally safe to analyze without executing code.

        Args:
            file_path: Path to the file

        Returns:
            True as CoreML Package files are generally safe
        """
        return True

    def analyze(
            self,
            file_path: str,
            sandbox: Optional[Sandbox] = None
    ) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a CoreML Package file.

        Args:
            file_path: Path to the CoreML Package file
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
                sandbox.check_format_permission('.mlpackage', Permission.READ_FILE)

            # Extract metadata
            metadata = self._extract_coreml_package_metadata(file_path)

            # Determine model type
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing CoreML Package file {file_path}: {e}")
            raise

    def _extract_coreml_package_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a CoreML Package file/directory.

        Args:
            file_path: Path to the CoreML Package

        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_size': self._get_directory_size(file_path),
            'format': 'coreml_package'
        }

        path = Path(file_path)

        # Check for required files in the package
        metadata['package_structure'] = {}

        # Look for Manifest.json
        manifest_path = path / 'Manifest.json'
        if manifest_path.exists():
            metadata['package_structure']['has_manifest'] = True
            with open(manifest_path, 'r') as f:
                try:
                    manifest_data = json.load(f)
                    metadata['manifest'] = manifest_data

                    # Extract info from manifest
                    if 'rootModelContainer' in manifest_data:
                        container = manifest_data['rootModelContainer']
                        if 'modelDescriptionSummary' in container:
                            metadata['description'] = container['modelDescriptionSummary']

                        if 'rootModelSpec' in container:
                            metadata['model_spec'] = container['rootModelSpec']
                except json.JSONDecodeError:
                    metadata['package_structure']['manifest_valid'] = False
        else:
            metadata['package_structure']['has_manifest'] = False

        # Look for model.mil
        model_mil_path = path / 'model.mil'
        if model_mil_path.exists():
            metadata['package_structure']['has_model_mil'] = True
            metadata['model_size'] = model_mil_path.stat().st_size
        else:
            metadata['package_structure']['has_model_mil'] = False

        # Look for Info.plist
        info_plist_path = path / 'Info.plist'
        if info_plist_path.exists():
            metadata['package_structure']['has_info_plist'] = True
            with open(info_plist_path, 'rb') as f:
                try:
                    plist_data = plistlib.load(f)
                    metadata['info_plist'] = plist_data
                except:
                    metadata['package_structure']['info_plist_valid'] = False
        else:
            metadata['package_structure']['has_info_plist'] = False

        # Check for Data directory
        data_dir = path / 'Data'
        if data_dir.exists() and data_dir.is_dir():
            metadata['package_structure']['has_data_dir'] = True

            # Count weight files
            weight_files = list(data_dir.glob("*.bin"))
            metadata['weight_files_count'] = len(weight_files)

            # Get total size of weights
            metadata['weights_size'] = sum(f.stat().st_size for f in weight_files)
        else:
            metadata['package_structure']['has_data_dir'] = False

        # Check for NeuroML directory (contains neural network architecture)
        neuro_ml_dir = path / 'NeuroML'
        if neuro_ml_dir.exists() and neuro_ml_dir.is_dir():
            metadata['package_structure']['has_neuro_ml_dir'] = True

            # Check for Network.json
            network_json_path = neuro_ml_dir / 'Network.json'
            if network_json_path.exists():
                metadata['package_structure']['has_network_json'] = True
                with open(network_json_path, 'r') as f:
                    try:
                        network_data = json.load(f)
                        metadata['network_json'] = network_data

                        # Extract network structure
                        if 'layers' in network_data:
                            layers = network_data['layers']
                            layer_types = {}
                            for layer in layers:
                                if 'type' in layer:
                                    layer_type = layer['type']
                                    layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

                            metadata['layer_types'] = layer_types
                            metadata['layer_count'] = len(layers)
                    except json.JSONDecodeError:
                        metadata['package_structure']['network_json_valid'] = False
            else:
                metadata['package_structure']['has_network_json'] = False
        else:
            metadata['package_structure']['has_neuro_ml_dir'] = False

        # Try to extract model details using coremltools if available
        try:
            import coremltools as ct
            model = ct.models.MLModel(file_path)

            # Extract basic info
            metadata['coremltools_info'] = {}
            spec = model.get_spec()

            # Extract model description
            if spec.description:
                metadata['coremltools_info']['description'] = {
                    'name': spec.description.name,
                    'short_description': spec.description.shortDescription,
                    'author': spec.description.author,
                    'license': spec.description.license,
                    'version': spec.description.versionString,
                }

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

            # Determine model type
            for model_type in self.MODEL_TYPES.keys():
                if spec.HasField(model_type):
                    metadata['coremltools_info']['model_type'] = model_type
        except ImportError:
            # coremltools not available, continue with file-based analysis only
            pass
        except Exception as e:
            metadata['coremltools_error'] = str(e)

        return metadata

    def _get_directory_size(self, directory_path: str) -> int:
        """
        Calculate the total size of a directory and its contents.

        Args:
            directory_path: Path to the directory

        Returns:
            Size in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if not os.path.islink(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Direct model type from coremltools if available
        if 'coremltools_info' in metadata and 'model_type' in metadata['coremltools_info']:
            model_type = metadata['coremltools_info']['model_type']
            if model_type in self.MODEL_TYPES:
                return self.MODEL_TYPES[model_type], ModelConfidence.HIGH
            else:
                return f"CoreML-{model_type}", ModelConfidence.HIGH

        # Look for model type in the manifest
        if 'manifest' in metadata:
            manifest = metadata['manifest']
            if 'rootModelContainer' in manifest:
                container = manifest['rootModelContainer']

                # Check for model type in rootModelSpec
                if 'rootModelSpec' in container:
                    spec = container['rootModelSpec']
                    if 'type' in spec:
                        model_type = spec['type']
                        if model_type in self.MODEL_TYPES:
                            return self.MODEL_TYPES[model_type], ModelConfidence.HIGH
                        return f"CoreML-{model_type}", ModelConfidence.HIGH

                # Check for model description summary
                if 'modelDescriptionSummary' in container:
                    desc = container['modelDescriptionSummary']
                    if 'modelType' in desc:
                        model_type = desc['modelType']
                        if model_type in self.MODEL_TYPES:
                            return self.MODEL_TYPES[model_type], ModelConfidence.HIGH
                        return f"CoreML-{model_type}", ModelConfidence.HIGH

        # Check for network architecture based on layer types
        if 'layer_types' in metadata and metadata['layer_types']:
            layer_types = metadata['layer_types']

            # Detect common model architectures from layer composition
            if 'CONV' in layer_types or 'convolution' in layer_types:
                if 'LSTM' in layer_types or 'lstm' in layer_types:
                    return 'CoreML-CNN-LSTM', ModelConfidence.MEDIUM
                return 'CoreML-CNN', ModelConfidence.MEDIUM

            if 'LSTM' in layer_types or 'lstm' in layer_types or 'GRU' in layer_types or 'gru' in layer_types:
                return 'CoreML-RNN', ModelConfidence.MEDIUM

            if 'attention' in layer_types or 'multiheadattention' in layer_types:
                return 'CoreML-Transformer', ModelConfidence.MEDIUM

            if 'embedding' in layer_types and ('dense' in layer_types or 'innerproduct' in layer_types):
                return 'CoreML-NLP', ModelConfidence.MEDIUM

        # Check inputs for clues about model type
        if 'inputs' in metadata and metadata['inputs']:
            inputs = metadata['inputs']

            # Check for image inputs (common in vision models)
            for input_info in inputs:
                if input_info.get('type') == 'imageType':
                    return 'CoreML-VisionModel', ModelConfidence.MEDIUM

            # Check for multiArrayType with 3+ dimensions (likely CNN)
            for input_info in inputs:
                if (input_info.get('type') == 'multiArrayType' and
                        'shape' in input_info and len(input_info['shape']) >= 3):
                    return 'CoreML-NeuralNetwork', ModelConfidence.MEDIUM

        # Check if it's an ML Program (newer CoreML format)
        if metadata['package_structure'].get('has_model_mil', False):
            return 'CoreML-MLProgram', ModelConfidence.MEDIUM

        # Default to generic CoreML model
        return 'CoreML-Package', ModelConfidence.LOW

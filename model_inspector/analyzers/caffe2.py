"""
Module for analyzing Caffe2 models (.pb) to determine model type and extract metadata.

Caffe2 models are stored as Protocol Buffer files (.pb) with a binary encoding of
the model structure and parameters. This module parses these files to extract
useful metadata about the model architecture, inputs, outputs, and operations.

Potential improvements:
1. Enhanced architecture identification by creating a more comprehensive catalog
   of known Caffe2 architectures and their identifying operations
2. Support for extracting parameters sizes and memory footprint
3. Recognition of common CV model variants (ResNet50, etc.) based on layer patterns
4. Ability to identify quantized models and their quantization parameters
5. Support for reading newer Caffe2 model formats and extensions
6. Analysis of model complexity (FLOPs, parameter count) as seen in other analyzers
"""

import os
import io
import struct
from typing import Dict, Any, Tuple, Optional, List, Set
import logging
from pathlib import Path
import re
from collections import defaultdict, Counter
import google.protobuf.text_format as text_format

# Try to import Caffe2 protobuf definitions
try:
    from caffe2.proto import caffe2_pb2

    HAS_CAFFE2_PROTO = True
except ImportError:
    HAS_CAFFE2_PROTO = False

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class Caffe2Analyzer(BaseAnalyzer):
    """
    Analyzer for Caffe2 model files (.pb format).

    Caffe2 models are stored as Protocol Buffer files that contain the network
    definition, weights, and other metadata. This analyzer extracts information
    about model operations, inputs, outputs, and attempts to identify the model
    architecture.
    """

    # Known architecture patterns to identify models
    KNOWN_ARCHITECTURES = {
        'resnet': 'ResNet',
        'inception': 'Inception',
        'mobilenet': 'MobileNet',
        'squeezenet': 'SqueezeNet',
        'vgg': 'VGG',
        'alexnet': 'AlexNet',
        'densenet': 'DenseNet',
        'shufflenet': 'ShuffleNet',
        'efficientnet': 'EfficientNet',
        'yolo': 'YOLO',
        'faster_rcnn': 'Faster R-CNN',
        'mask_rcnn': 'Mask R-CNN',
        'ssd': 'SSD',
    }

    # Operator types that suggest CV model
    CV_OPERATORS = {
        'Conv', 'MaxPool', 'AveragePool', 'BatchNormalization',
        'Flatten', 'Resize', 'RoiAlign', 'NMS'
    }

    # Operator types that suggest NLP model
    NLP_OPERATORS = {
        'LSTM', 'GRU', 'Attention', 'Embedding', 'MatMul',
        'Softmax', 'LayerNormalization', 'Gather'
    }

    def __init__(self):
        """Initialize the Caffe2 analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.pb'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        Args:
            file_path: Path to the model file

        Returns:
            True if the file can be analyzed safely
        """
        # Caffe2 models are generally safe to analyze
        return True

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a Caffe2 model file to determine its type and extract metadata.

        Args:
            file_path: Path to the Caffe2 model file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid Caffe2 model
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not HAS_CAFFE2_PROTO:
            self.logger.warning("Caffe2 protobuf module not found. Limited analysis available.")
            return self._fallback_analysis(file_path)

        try:
            # Read the protobuf file
            metadata = self._parse_caffe2_model(file_path)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing Caffe2 model {file_path}: {e}")
            raise

    def _parse_caffe2_model(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Caffe2 model file to extract metadata.

        Args:
            file_path: Path to the Caffe2 model file

        Returns:
            Dictionary of metadata
        """
        metadata = {
            'format': 'caffe2',
            'file_size_bytes': os.path.getsize(file_path)
        }

        try:
            # Try to load as NetDef protobuf
            net_def = caffe2_pb2.NetDef()
            with open(file_path, 'rb') as f:
                net_def.ParseFromString(f.read())

            # Extract basic properties
            metadata['name'] = net_def.name if net_def.name else 'unnamed_model'
            metadata['device_option'] = str(net_def.device_option)
            metadata['num_operators'] = len(net_def.op)

            # Extract inputs and outputs
            metadata['external_inputs'] = list(net_def.external_input)
            metadata['external_outputs'] = list(net_def.external_output)

            # Extract operators and their types
            operators = []
            op_types = []

            for op in net_def.op:
                op_info = {
                    'type': op.type,
                    'name': op.name,
                    'inputs': list(op.input),
                    'outputs': list(op.output)
                }
                operators.append(op_info)
                op_types.append(op.type)

            metadata['operators'] = operators
            metadata['operator_types'] = Counter(op_types)

            # Try to identify model architecture from operator patterns
            self._identify_architecture(metadata)

        except Exception as e:
            self.logger.warning(f"Error parsing Caffe2 model details: {e}")
            metadata['parse_error'] = str(e)

        return metadata

    def _fallback_analysis(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Perform basic analysis when detailed parsing is not possible.

        Args:
            file_path: Path to the Caffe2 model file

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        metadata = {
            'format': 'caffe2',
            'file_size_bytes': os.path.getsize(file_path),
            'limited_analysis': True
        }

        with open(file_path, 'rb') as f:
            # Try to identify if this is actually a Caffe2 file by looking for protobuf markers
            # and common strings
            content = f.read(2048)  # Read first 2KB for detection

            # Check for signs of a protobuf file
            is_likely_protobuf = False
            if content[:4] == b'\x0A':  # Common protobuf start
                is_likely_protobuf = True

            # Look for Caffe2 operator strings
            caffe2_indicators = [b'Conv', b'Relu', b'Pool', b'Softmax', b'Caffe', b'NCHW']
            indicator_count = sum(1 for indicator in caffe2_indicators if indicator in content)

            if indicator_count > 2:
                metadata['detected_signatures'] = 'Found Caffe2 operator names'
                return "Caffe2-Model", ModelConfidence.MEDIUM, metadata

            if is_likely_protobuf:
                return "Possible-Caffe2-Model", ModelConfidence.LOW, metadata

        return "Unknown", ModelConfidence.UNKNOWN, metadata

    def _identify_architecture(self, metadata: Dict[str, Any]) -> None:
        """
        Identify the model architecture based on operators and patterns.

        Args:
            metadata: Current metadata dictionary to update
        """
        # Convert operator types to a string for pattern matching
        if 'operators' not in metadata:
            return

        op_types_str = ' '.join([op['type'] for op in metadata['operators']])

        # Check for known architecture patterns
        for pattern, arch_name in self.KNOWN_ARCHITECTURES.items():
            if re.search(pattern, op_types_str, re.IGNORECASE) or re.search(pattern, metadata.get('name', ''),
                                                                            re.IGNORECASE):
                metadata['architecture'] = arch_name
                break

        # Analyze operator distribution to determine model domain
        if 'operator_types' in metadata:
            op_counter = metadata['operator_types']

            # Count occurrences of computer vision and NLP operators
            cv_count = sum(op_counter.get(op, 0) for op in self.CV_OPERATORS)
            nlp_count = sum(op_counter.get(op, 0) for op in self.NLP_OPERATORS)

            # Determine domain based on operator counts
            if cv_count > nlp_count and cv_count > 0:
                metadata['likely_domain'] = 'Computer Vision'
            elif nlp_count > cv_count and nlp_count > 0:
                metadata['likely_domain'] = 'Natural Language Processing'

        # Look for clues about model size
        total_operators = len(metadata.get('operators', []))
        if total_operators > 200:
            metadata['model_size_hint'] = 'Large (>200 operators)'
        elif total_operators > 100:
            metadata['model_size_hint'] = 'Medium (100-200 operators)'
        elif total_operators > 0:
            metadata['model_size_hint'] = 'Small (<100 operators)'

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # If architecture was identified, use that
        if 'architecture' in metadata:
            arch = metadata['architecture']

            # Try to add domain if available
            if 'likely_domain' in metadata:
                return f"{arch} ({metadata['likely_domain']})", ModelConfidence.MEDIUM

            return arch, ModelConfidence.MEDIUM

        # If we have a name with version, use that
        if 'name' in metadata and metadata['name'] != 'unnamed_model':
            model_name = metadata['name']

            # Look for known patterns in the name
            for pattern, arch_name in self.KNOWN_ARCHITECTURES.items():
                if pattern.lower() in model_name.lower():
                    return arch_name, ModelConfidence.MEDIUM

            # Use the model name itself
            return f"Caffe2-{model_name}", ModelConfidence.LOW

        # Fall back to domain classification if available
        if 'likely_domain' in metadata:
            return f"Caffe2-{metadata['likely_domain']}", ModelConfidence.LOW

        # Very basic fallback
        return "Caffe2-Model", ModelConfidence.LOW

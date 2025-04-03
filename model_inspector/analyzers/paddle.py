"""
Analyzer for PaddlePaddle framework model files (.pdmodel, .pdiparams).

PaddlePaddle is a deep learning framework developed by Baidu, which supports
both static and dynamic computational graphs. This analyzer handles the PaddlePaddle
model files which typically come in two parts:
- .pdmodel: Contains the model structure/topology
- .pdiparams: Contains the model parameters/weights

Potential improvements:
1. Add support for .pdiparams.info files which contain additional metadata
2. Implement deeper structure analysis of computational graphs
3. Support the legacy .pdparams format
4. Add extraction of more model-specific details like input and output shapes
5. Integrate with Paddle's Python API for more thorough model inspection when available
6. Add support for quantized model formats (.nb format)
"""

from typing import Dict, Any, Tuple, Optional, List, Set
import os
import struct
import logging
import json
from pathlib import Path
import re
import numpy as np
from collections import defaultdict, Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class PaddleAnalyzer(BaseAnalyzer):
    """
    Analyzer for PaddlePaddle model files (.pdmodel, .pdiparams).

    This analyzer handles model files from the PaddlePaddle framework, which are
    typically distributed as pairs of files (.pdmodel for structure and .pdiparams
    for weights). It extracts information about model architecture, parameters,
    and attempts to identify the model type.
    """

    # PaddlePaddle model file signature bytes
    PADDLE_MAGIC = b'PADDLE'

    # Common model architecture types in PaddlePaddle ecosystem
    MODEL_ARCHITECTURES = {
        'resnet': 'ResNet',
        'mobilenet': 'MobileNet',
        'yolo': 'YOLO',
        'ppyolo': 'PP-YOLO',
        'efficientnet': 'EfficientNet',
        'hrnet': 'HRNet',
        'transformer': 'Transformer',
        'bert': 'BERT',
        'ernie': 'ERNIE',
        'gpt': 'GPT',
        'llama': 'LLaMA',
        'ocr': 'OCR',
        'pp-ocr': 'PP-OCR',
        'pp-tts': 'PP-TTS',
        'deeplabv3': 'DeepLabV3',
        'unet': 'U-Net',
        'gan': 'GAN',
        'cyclegan': 'CycleGAN',
        'ppgan': 'PP-GAN',
        'paddleseg': 'PaddleSeg',
        'paddledet': 'PaddleDetection',
        'paddlenlp': 'PaddleNLP',
    }

    def __init__(self):
        """Initialize the PaddlePaddle analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.pdmodel', '.pdiparams'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a PaddlePaddle model file to determine its model type and metadata.

        Args:
            file_path: Path to the model file (.pdmodel or .pdiparams)

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid PaddlePaddle file
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Determine file type based on extension
            file_extension = path.suffix.lower()

            # Initialize metadata dict with basic file info
            metadata = {
                "format": "paddle",
                "file_type": file_extension[1:],  # Remove the leading dot
                "file_size_bytes": os.path.getsize(file_path)
            }

            # Process based on file type
            if file_extension == '.pdmodel':
                self._analyze_pdmodel(file_path, metadata)
            elif file_extension == '.pdiparams':
                self._analyze_pdiparams(file_path, metadata)

            # Try to find companion file
            companion_file = self._find_companion_file(file_path)
            if companion_file:
                metadata["has_companion_file"] = True
                metadata["companion_file"] = os.path.basename(companion_file)

                # If we have both files, we can do a more complete analysis
                if file_extension == '.pdmodel' and companion_file.endswith('.pdiparams'):
                    self._add_combined_info(file_path, companion_file, metadata)
                elif file_extension == '.pdiparams' and companion_file.endswith('.pdmodel'):
                    self._add_combined_info(companion_file, file_path, metadata)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing PaddlePaddle file {file_path}: {e}")
            raise

    def _analyze_pdmodel(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Analyze a .pdmodel file to extract model structure information.

        Args:
            file_path: Path to the .pdmodel file
            metadata: Metadata dict to update
        """
        try:
            with open(file_path, 'rb') as f:
                # Check if it's a protobuf file (PaddlePaddle models use protobuf)
                header = f.read(16)
                f.seek(0)

                # Check for protobuf signature or paddle signature
                if self.PADDLE_MAGIC in header:
                    metadata["detected_format"] = "paddle_binary"
                else:
                    metadata["detected_format"] = "protobuf"

                # Read and analyze the file content
                content = f.read()

                # Extract operator types from the binary content
                ops = self._extract_ops_from_binary(content)
                if ops:
                    metadata["operators"] = ops
                    metadata["top_operators"] = self._get_top_n_items(ops, 10)

                # Extract network structure indicators
                metadata["network_indicators"] = self._extract_network_indicators(content)

                # Attempt to determine if this is a vision, nlp, or other type of model
                metadata["model_domain"] = self._determine_model_domain(content, ops)

        except Exception as e:
            self.logger.warning(f"Error analyzing pdmodel file: {e}")
            metadata["analysis_error"] = str(e)

    def _analyze_pdiparams(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Analyze a .pdiparams file to extract parameter information.

        Args:
            file_path: Path to the .pdiparams file
            metadata: Metadata dict to update
        """
        try:
            # Params files are typically binary files with model weights
            file_size = os.path.getsize(file_path)
            metadata["param_file_size_mb"] = round(file_size / (1024 * 1024), 2)

            with open(file_path, 'rb') as f:
                # Try to detect if it's a standard params file
                header = f.read(16)
                f.seek(0)

                # Check for protobuf or paddle signature
                if self.PADDLE_MAGIC in header:
                    metadata["detected_format"] = "paddle_binary"

                # We can't extract much from binary params directly without Paddle API
                # But we can estimate model size from file size
                if file_size > 1e9:  # > 1GB
                    metadata["estimated_size"] = "Large (1GB+)"
                elif file_size > 100e6:  # > 100MB
                    metadata["estimated_size"] = "Medium (100MB-1GB)"
                else:
                    metadata["estimated_size"] = "Small (<100MB)"

                # Look for common parameter tensors in the binary data
                sample_data = f.read(min(10 * 1024 * 1024, file_size))  # Read up to 10MB

                # Try to identify common tensor names in binary data
                tensor_indicators = self._extract_tensor_indicators(sample_data)
                if tensor_indicators:
                    metadata["tensor_indicators"] = tensor_indicators

        except Exception as e:
            self.logger.warning(f"Error analyzing pdiparams file: {e}")
            metadata["analysis_error"] = str(e)

    def _find_companion_file(self, file_path: str) -> Optional[str]:
        """
        Find the companion file for a given PaddlePaddle model file.

        Args:
            file_path: Path to either .pdmodel or .pdiparams file

        Returns:
            Path to companion file if found, None otherwise
        """
        path = Path(file_path)
        base_path = path.parent / path.stem

        if path.suffix == '.pdmodel':
            companion = str(base_path) + '.pdiparams'
            return companion if os.path.exists(companion) else None
        elif path.suffix == '.pdiparams':
            companion = str(base_path) + '.pdmodel'
            return companion if os.path.exists(companion) else None

        return None

    def _add_combined_info(self, model_path: str, params_path: str, metadata: Dict[str, Any]) -> None:
        """
        Add information from combined analysis of both model and params files.

        Args:
            model_path: Path to .pdmodel file
            params_path: Path to .pdiparams file
            metadata: Metadata dict to update
        """
        # When we have both files, we can do a more comprehensive analysis
        metadata["combined_size_mb"] = round(
            (os.path.getsize(model_path) + os.path.getsize(params_path)) / (1024 * 1024), 2
        )

        # Look for parameter info file which might contain additional metadata
        info_path = params_path + '.info'
        if os.path.exists(info_path):
            metadata["has_params_info"] = True
            try:
                with open(info_path, 'r') as f:
                    info_data = json.load(f)
                    if isinstance(info_data, dict):
                        metadata["params_info"] = info_data
            except Exception as e:
                self.logger.warning(f"Error reading params info file: {e}")

    def _extract_ops_from_binary(self, content: bytes) -> Dict[str, int]:
        """
        Extract operator types from binary content by looking for common patterns.

        Args:
            content: Binary content of the model file

        Returns:
            Dictionary mapping operator names to their frequency
        """
        # Common operator names in PaddlePaddle models
        common_ops = [
            'conv2d', 'batch_norm', 'pool2d', 'relu', 'softmax', 'matmul',
            'elementwise_add', 'elementwise_mul', 'dropout', 'concat',
            'transpose', 'reshape', 'slice', 'fill_constant', 'scale',
            'attention', 'layer_norm', 'multihead_attention', 'feed', 'fetch',
            'sigmoid', 'tanh', 'leaky_relu', 'fc', 'linear', 'lstm', 'gru'
        ]

        # Count occurrences of operator names in the binary content
        op_counts = {}
        content_str = str(content)

        for op in common_ops:
            # Check for the operator name with word boundaries
            pattern = r'\b' + re.escape(op) + r'\b'
            count = len(re.findall(pattern, content_str))
            if count > 0:
                op_counts[op] = count

        return op_counts

    def _extract_network_indicators(self, content: bytes) -> List[str]:
        """
        Extract indicators of network architecture from binary content.

        Args:
            content: Binary content of the model file

        Returns:
            List of identified network architecture indicators
        """
        indicators = []
        content_str = str(content)

        # Check for common architecture patterns
        architecture_patterns = {
            r'\bresnet\d+\b': 'ResNet',
            r'\bmobilenet\b': 'MobileNet',
            r'\befficientnet\b': 'EfficientNet',
            r'\bvgg\d*\b': 'VGG',
            r'\binception\b': 'Inception',
            r'\bdensenet\b': 'DenseNet',
            r'\byolo\b': 'YOLO',
            r'\bfaster_rcnn\b': 'Faster R-CNN',
            r'\bmask_rcnn\b': 'Mask R-CNN',
            r'\bssd\b': 'SSD',
            r'\bunet\b': 'U-Net',
            r'\bdeeplabv3\b': 'DeepLabV3',
            r'\bert\b': 'BERT',
            r'\bernie\b': 'ERNIE',
            r'\bgpt\b': 'GPT',
            r'\btransformer\b': 'Transformer',
            r'\bllama\b': 'LLaMA',
            r'\bcnn\b': 'CNN',
            r'\brnn\b': 'RNN',
            r'\blstm\b': 'LSTM',
            r'\bgru\b': 'GRU',
            r'\bgan\b': 'GAN',
        }

        for pattern, name in architecture_patterns.items():
            if re.search(pattern, content_str, re.IGNORECASE):
                indicators.append(name)

        return list(set(indicators))  # Remove duplicates

    def _extract_tensor_indicators(self, data: bytes) -> List[str]:
        """
        Extract tensor name indicators from binary parameter data.

        Args:
            data: Binary content sample from params file

        Returns:
            List of identified tensor type indicators
        """
        indicators = []
        data_str = str(data)

        # Look for common tensor naming patterns
        tensor_patterns = [
            (r'embedding', 'Embedding layers'),
            (r'conv\d+', 'Convolutional layers'),
            (r'bn\d+', 'Batch normalization layers'),
            (r'fc\d+', 'Fully connected layers'),
            (r'linear', 'Linear layers'),
            (r'attention', 'Attention mechanism'),
            (r'lstm', 'LSTM cells'),
            (r'gru', 'GRU cells'),
            (r'layer_norm', 'Layer normalization'),
            (r'bias', 'Bias terms'),
            (r'weight', 'Weight matrices')
        ]

        for pattern, desc in tensor_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                indicators.append(desc)

        return list(set(indicators))  # Remove duplicates

    def _determine_model_domain(self, content: bytes, ops: Dict[str, int]) -> str:
        """
        Determine the likely domain of the model (vision, NLP, etc.).

        Args:
            content: Binary content of the model file
            ops: Dictionary of operator frequencies

        Returns:
            String indicating the likely model domain
        """
        content_str = str(content)

        # Vision model indicators
        vision_indicators = ['conv2d', 'pool2d', 'batch_norm', 'resnet', 'mobilenet', 'yolo', 'detection',
                             'segment', 'vgg', 'inception', 'efficientnet', 'unet', 'deeplabv3', 'rcnn']

        # NLP model indicators
        nlp_indicators = ['transformer', 'attention', 'bert', 'ernie', 'gpt', 'llama', 'encoder', 'decoder',
                          'token', 'embedding', 'lstm', 'gru', 'language', 'text']

        # Audio model indicators
        audio_indicators = ['mel', 'spectral', 'spectrogram', 'audio', 'speech', 'voice', 'tts', 'asr', 'wav']

        # Count matches for each domain
        vision_score = sum(1 for ind in vision_indicators if ind.lower() in content_str.lower()) + \
                       sum(ops.get(op, 0) for op in ['conv2d', 'pool2d', 'batch_norm'])

        nlp_score = sum(1 for ind in nlp_indicators if ind.lower() in content_str.lower()) + \
                    sum(ops.get(op, 0) for op in ['attention', 'layer_norm', 'multihead_attention'])

        audio_score = sum(1 for ind in audio_indicators if ind.lower() in content_str.lower())

        # Determine domain based on highest score
        if vision_score > nlp_score and vision_score > audio_score:
            return "Computer Vision"
        elif nlp_score > vision_score and nlp_score > audio_score:
            return "Natural Language Processing"
        elif audio_score > vision_score and audio_score > nlp_score:
            return "Audio Processing"
        elif vision_score > 0 and nlp_score > 0:
            return "Multi-modal (Vision+NLP)"
        else:
            return "Unknown"

    def _get_top_n_items(self, counter: Dict[str, int], n: int) -> Dict[str, int]:
        """
        Get the top N items from a counter dictionary.

        Args:
            counter: Dictionary mapping items to their counts
            n: Number of top items to return

        Returns:
            Dictionary with the top N items
        """
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:n])

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check for network indicators first
        if 'network_indicators' in metadata and metadata['network_indicators']:
            primary_indicator = metadata['network_indicators'][0]
            domain = metadata.get('model_domain', '')

            # If we have both architecture and domain, combine them
            if domain and domain != "Unknown":
                model_type = f"{primary_indicator} ({domain})"
                return model_type, ModelConfidence.HIGH
            else:
                return primary_indicator, ModelConfidence.MEDIUM

        # Check for operator patterns to determine model type
        if 'operators' in metadata:
            ops = metadata['operators']

            # Check for transformers and attention-based models
            if 'attention' in ops or 'multihead_attention' in ops:
                if 'model_domain' in metadata:
                    if metadata['model_domain'] == "Natural Language Processing":
                        return "Transformer-NLP", ModelConfidence.MEDIUM
                    else:
                        return "Transformer", ModelConfidence.MEDIUM

            # Check for CNNs and vision models
            if 'conv2d' in ops and 'pool2d' in ops:
                domain = metadata.get('model_domain', '')
                if domain == "Computer Vision":
                    return "CNN-Vision", ModelConfidence.MEDIUM
                else:
                    return "CNN", ModelConfidence.MEDIUM

            # Check for RNNs
            if 'lstm' in ops or 'gru' in ops:
                return "RNN", ModelConfidence.MEDIUM

        # Check binary indicators in file content
        for arch_key, arch_name in self.MODEL_ARCHITECTURES.items():
            # Check if architecture name appears in tensor indicators or network indicators
            if 'tensor_indicators' in metadata and any(
                    arch_key.lower() in ind.lower() for ind in metadata['tensor_indicators']):
                return arch_name, ModelConfidence.LOW

            if 'network_indicators' in metadata and any(
                    arch_key.lower() in ind.lower() for ind in metadata['network_indicators']):
                return arch_name, ModelConfidence.MEDIUM

        # Check model domain if available
        if 'model_domain' in metadata and metadata['model_domain'] != "Unknown":
            return f"Paddle-{metadata['model_domain']}", ModelConfidence.LOW

        # Generic fallback based on file type
        if metadata.get('file_type') == 'pdmodel':
            return "Paddle-Model", ModelConfidence.LOW
        elif metadata.get('file_type') == 'pdiparams':
            return "Paddle-Weights", ModelConfidence.LOW

        return "Paddle-Unknown", ModelConfidence.UNKNOWN

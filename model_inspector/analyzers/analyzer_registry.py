"""
Module for managing the registry of model analyzers and dispatching analysis requests.

This module provides functions for registering analyzers and looking up the
appropriate analyzer for a given file extension.
"""

import os
from typing import Dict, Any, Type, List

from .base import BaseAnalyzer
from .safetensors import SafetensorsAnalyzer
from .gguf import GGUFAnalyzer
from .pytorch import PyTorchAnalyzer
from .onnx import ONNXAnalyzer
from .ort import ORTAnalyzer
from .tensorflow import TensorFlowAnalyzer
from .checkpoint import CheckpointAnalyzer
from .hdf5 import HDF5Analyzer
from .coreml_package import CoreMLPackageAnalyzer
from .mlmodel import MLModelAnalyzer
from .tflite import TFLiteAnalyzer
from .jax import JAXAnalyzer
from .sklearn import SklearnAnalyzer
from .bin_analyzer import BinAnalyzer
from .caffe import CaffeModelAnalyzer
from .mps import MPSAnalyzer
from .caffe2 import Caffe2Analyzer
from .openvino_ir import OpenVINOIRAnalyzer
from .mxnet import MXNetAnalyzer
from ..exceptions import UnsupportedFormatError
from .cuml import CuMLAnalyzer
from .tvm import TVMAnalyzer
from .paddle import PaddleAnalyzer
from .xgboost import XGBoostAnalyzer
from .enn import ENNAnalyzer
from ..exceptions import UnsupportedFormatError

# Registry of analyzers by file extension
ANALYZER_REGISTRY = {
    '.safetensors': SafetensorsAnalyzer,
    '.gguf': GGUFAnalyzer,
    '.ggml': GGUFAnalyzer,  # GGML uses the same analyzer as GGUF
    '.pt': PyTorchAnalyzer,  # Will be updated by PyTorchDispatchAnalyzer
    '.pth': PyTorchAnalyzer,  # Will be updated by PyTorchDispatchAnalyzer
    '.onnx': ONNXAnalyzer,
    '.ort': ORTAnalyzer,
    '.pb': TensorFlowAnalyzer,  # Will be updated by PBDispatchAnalyzer
    '.ckpt': CheckpointAnalyzer,
    '.h5': HDF5Analyzer,
    '.hdf5': HDF5Analyzer,
    '.mlpackage': CoreMLPackageAnalyzer,
    '.mlmodel': MLModelAnalyzer,
    '.msgpack': JAXAnalyzer,
    '.tflite': TFLiteAnalyzer,
    '.joblib': SklearnAnalyzer,
    '.pkl': SklearnAnalyzer,
    '.pickle': SklearnAnalyzer,
    '.bin': BinAnalyzer,  # Will be handled by BinAnalyzer smart detection
    '.cuml': CuMLAnalyzer,  # RAPIDS cuML model files
    '.mps': MPSAnalyzer,
    '.caffemodel': CaffeModelAnalyzer,
    '.xml': OpenVINOIRAnalyzer,  # OpenVINO IR XML file
    '.params': MXNetAnalyzer,  # MXNet params file
    '.so': TVMAnalyzer,  # Note: This might conflict with other .so analyzers - should use a dispatcher
    '.tar': TVMAnalyzer,  # May need to use a dispatcher for .tar files
    '.pdmodel': PaddleAnalyzer,  # PaddlePaddle model structure
    '.pdiparams': PaddleAnalyzer,  # PaddlePaddle model parameters
    '.json': XGBoostAnalyzer,  # XGBoost JSON format
    '.ubj': XGBoostAnalyzer,  # XGBoost Universal Binary JSON format
    '.enn': ENNAnalyzer,  # EPUB Neural Codec format
}


def get_analyzer_for_extension(extension: str) -> BaseAnalyzer:
    """
    Get the appropriate analyzer for a file extension.

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        Instantiated analyzer for the extension

    Raises:
        UnsupportedFormatError: If no analyzer is available for the extension
    """
    # Normalize extension
    if not extension.startswith('.'):
        extension = f'.{extension}'
    extension = extension.lower()

    analyzer_class = ANALYZER_REGISTRY.get(extension)
    if not analyzer_class:
        raise UnsupportedFormatError(f"No analyzer available for extension: {extension}")

    return analyzer_class()


def register_analyzer(extensions: List[str], analyzer_class: Type[BaseAnalyzer]) -> None:
    """
    Register a new analyzer for the specified extensions.

    Args:
        extensions: List of file extensions this analyzer handles
        analyzer_class: Analyzer class to register
    """
    for ext in extensions:
        # Normalize extension
        if not ext.startswith('.'):
            ext = f'.{ext}'
        ext = ext.lower()

        ANALYZER_REGISTRY[ext] = analyzer_class


# Update registry with dispatchers (will be defined in analyzer_dispatchers.py)
# This will be called from analyzer_dispatchers.py after dispatchers are defined
def update_registry_with_dispatchers(dispatcher_dict):
    """
    Update registry with dispatchers from analyzer_dispatchers.py.

    Args:
        dispatcher_dict: Dictionary mapping extensions to dispatcher classes
    """
    ANALYZER_REGISTRY.update(dispatcher_dict)

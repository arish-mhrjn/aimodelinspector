"""
Model analyzer package providing functionality for analyzing ML model files.
"""

from .base import BaseAnalyzer
from .analyzer_registry import get_analyzer_for_extension, register_analyzer
from .safetensors import SafetensorsAnalyzer
from .gguf import GGUFAnalyzer
from .pytorch import PyTorchAnalyzer
from .pytorch_jit import PyTorchJITAnalyzer
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
from .cuml import CuMLAnalyzer
from .tvm import TVMAnalyzer
from .paddle import PaddleAnalyzer
from .analyzer_dispatchers import PyTorchDispatchAnalyzer, PBDispatchAnalyzer
from .diffusers import DiffusersAnalyzer
from .xgboost import XGBoostAnalyzer
from .enn import ENNAnalyzer

# Import dispatchers to ensure they register themselves
import model_inspector.analyzers.analyzer_dispatchers

__all__ = [
    'BaseAnalyzer',
    'get_analyzer_for_extension',
    'register_analyzer',
    'SafetensorsAnalyzer',
    'GGUFAnalyzer',
    'PyTorchAnalyzer',
    'PyTorchJITAnalyzer',
    'ONNXAnalyzer',
    'ORTAnalyzer',
    'TensorFlowAnalyzer',
    'CheckpointAnalyzer',
    'HDF5Analyzer',
    'CoreMLPackageAnalyzer',
    'MLModelAnalyzer',
    'TFLiteAnalyzer',
    'JAXAnalyzer',
    'SklearnAnalyzer',
    'BinAnalyzer',
    'CaffeModelAnalyzer',
    'CuMLAnalyzer',
    'MPSAnalyzer',
    'Caffe2Analyzer',
    'OpenVINOIRAnalyzer',
    'MXNetAnalyzer',
    'PaddleAnalyzer',
    'PyTorchDispatchAnalyzer',
    'PBDispatchAnalyzer',
    'DiffusersAnalyzer',
    'TVMAnalyzer',
    'XGBoostAnalyzer',
    'ENNAnalyzer',
]

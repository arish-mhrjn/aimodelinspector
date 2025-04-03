"""
Module containing dispatcher classes that route model analysis to specialized analyzers.

These dispatchers handle file formats that could be processed by different analyzers
based on their specific content or structure.
"""

import os
from typing import Dict, Any, Tuple
import tarfile

from .base import BaseAnalyzer
from .pytorch import PyTorchAnalyzer
from .pytorch_jit import PyTorchJITAnalyzer
from .tensorflow import TensorFlowAnalyzer
from .caffe2 import Caffe2Analyzer
from .diffusers import DiffusersAnalyzer
from .tvm import TVMAnalyzer  # Add this import
from ..models.confidence import ModelConfidence
from .analyzer_registry import update_registry_with_dispatchers


class PyTorchDispatchAnalyzer(BaseAnalyzer):
    """Dispatcher that chooses between PyTorch and PyTorch JIT analyzers."""

    def __init__(self):
        """Initialize both analyzers."""
        super().__init__()
        self.pytorch_analyzer = PyTorchAnalyzer()
        self.jit_analyzer = PyTorchJITAnalyzer()

    def get_supported_extensions(self) -> set:
        """Get supported file extensions."""
        return {'.pt', '.pth'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """Check if file can be analyzed safely."""
        # Both PyTorch formats are unsafe due to pickle
        return False

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a PyTorch file by detecting whether it's JIT or regular format.

        Args:
            file_path: Path to the PyTorch file

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        # Check if it's a JIT model first
        try:
            is_jit, _ = self.jit_analyzer._check_jit_model(file_path)
            if is_jit:
                return self.jit_analyzer.analyze(file_path)
            else:
                return self.pytorch_analyzer.analyze(file_path)
        except Exception as e:
            self.logger.error(f"Error in PyTorch dispatcher: {e}")
            # Default to regular PyTorch if detection fails
            return self.pytorch_analyzer.analyze(file_path)


class PBDispatchAnalyzer(BaseAnalyzer):
    """Dispatcher that chooses between TensorFlow and Caffe2 analyzers."""

    def __init__(self):
        """Initialize both analyzers."""
        super().__init__()
        self.tensorflow_analyzer = TensorFlowAnalyzer()
        self.caffe2_analyzer = Caffe2Analyzer()

    def get_supported_extensions(self) -> set:
        """Get supported file extensions."""
        return {'.pb'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """Check if file can be analyzed safely."""
        # Both formats should be safe
        return True

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a .pb file by detecting whether it's TensorFlow or Caffe2 format.

        Args:
            file_path: Path to the .pb file

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        # First try TensorFlow (more common)
        try:
            return self.tensorflow_analyzer.analyze(file_path)
        except Exception as e:
            self.logger.info(f"Not a TensorFlow model, trying Caffe2: {e}")

        # If TensorFlow fails, try Caffe2
        try:
            return self.caffe2_analyzer.analyze(file_path)
        except Exception as e:
            self.logger.error(f"Error in PB dispatcher: {e}")
            # Fall back to basic file analysis
            metadata = {
                'format': 'unknown_pb',
                'file_size_bytes': os.path.getsize(file_path)
            }
            return "Unknown-PB-Model", ModelConfidence.UNKNOWN, metadata


class SODispatchAnalyzer(BaseAnalyzer):
    """
    Dispatcher analyzer for .so (shared object) files.

    This analyzer examines shared library files to determine the appropriate
    specific analyzer to use based on file contents.
    """

    def __init__(self):
        """Initialize the .so file dispatcher analyzer."""
        super().__init__()
        # Pre-initialize the TVM analyzer to avoid repeated initialization
        self.tvm_analyzer = TVMAnalyzer()

    def get_supported_extensions(self) -> set:
        """Get supported file extensions."""
        return {'.so'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a .so file to determine which specific analyzer to use.

        Args:
            file_path: Path to the .so file

        Returns:
            Results from the appropriate specific analyzer
        """
        # Try to determine if this is a TVM model first
        try:
            import subprocess
            nm_output = subprocess.check_output(['nm', '-D', file_path], stderr=subprocess.STDOUT).decode('utf-8',
                                                                                                          errors='ignore')

            # Check for TVM symbols
            tvm_symbols = ['TVMBackendGetFuncFromEnv', 'TVMBackendAllocWorkspace', 'TVMFuncCall']
            is_tvm_model = any(symbol in nm_output for symbol in tvm_symbols)

            if is_tvm_model:
                return self.tvm_analyzer.analyze(file_path)
        except Exception as e:
            self.logger.warning(f"Error during SO file symbol check: {e}")

        # If not a TVM model or can't determine, use other .so analyzers here
        # For example, you might have analyzers for ONNX Runtime, TensorFlow, etc.

        # Default fallback
        return "Unknown Shared Library", ModelConfidence.LOW, {"format": "unknown_so", "file_size_bytes": os.path.getsize(file_path)}


class TarDispatchAnalyzer(BaseAnalyzer):
    """
    Dispatcher analyzer for .tar archive files.

    This analyzer examines tar archives to determine the appropriate
    specific analyzer to use based on file contents.
    """

    def __init__(self):
        """Initialize the .tar file dispatcher analyzer."""
        super().__init__()
        # Pre-initialize analyzers to avoid repeated initialization
        self.tvm_analyzer = TVMAnalyzer()

    def get_supported_extensions(self) -> set:
        """Get supported file extensions."""
        return {'.tar'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a .tar file to determine which specific analyzer to use.

        Args:
            file_path: Path to the .tar file

        Returns:
            Results from the appropriate specific analyzer
        """
        # Check if it's a valid tar file
        if not tarfile.is_tarfile(file_path):
            return "Invalid TAR Archive", ModelConfidence.LOW, {"format": "invalid_tar"}

        # Try to determine if this is a TVM model archive
        try:
            with tarfile.open(file_path, 'r') as tar:
                file_list = tar.getnames()

                # Check for TVM model indicators
                has_lib = any(f.endswith('lib.so') or f == 'lib.so' for f in file_list)
                has_params = any(f.endswith('params') or f == 'params' for f in file_list)
                has_graph = any(f.endswith('json') or f == 'graph.json' for f in file_list)

                # If it looks like a TVM model, use the TVM analyzer
                if has_lib and (has_params or has_graph):
                    return self.tvm_analyzer.analyze(file_path)

                # Add checks for other formats that use .tar archives here

        except Exception as e:
            self.logger.warning(f"Error during TAR file inspection: {e}")

        # Default fallback if we can't determine the specific type
        return "Unknown Archive", ModelConfidence.LOW, {
            "format": "unknown_tar",
            "file_size_bytes": os.path.getsize(file_path)
        }


# Dictionary of dispatchers to update the registry
DISPATCHER_DICT = {
    '.pt': PyTorchDispatchAnalyzer,
    '.pth': PyTorchDispatchAnalyzer,
    '.safetensors': DiffusersAnalyzer,
    '.pb': PBDispatchAnalyzer,
    '.so': SODispatchAnalyzer,
    '.tar': TarDispatchAnalyzer
}

# Update the registry with our dispatchers
update_registry_with_dispatchers(DISPATCHER_DICT)

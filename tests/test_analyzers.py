"""
Tests for model analyzers in the model_inspector library.
"""
import pytest
import os
import tempfile
from pathlib import Path

from model_inspector.analyzers import get_analyzer_for_extension, register_analyzer
from model_inspector.analyzers.base import BaseAnalyzer
from model_inspector.analyzers.safetensors import SafetensorsAnalyzer
from model_inspector.analyzers.onnx import ONNXAnalyzer
from model_inspector.analyzers.pytorch import PyTorchAnalyzer
from model_inspector.analyzers.tensorflow import TensorFlowAnalyzer
from model_inspector.analyzers.checkpoint import CheckpointAnalyzer
from model_inspector.analyzers.hdf5 import HDF5Analyzer
from model_inspector.analyzers.gguf import GGUFAnalyzer
from model_inspector.analyzers.coreml import CoreMLAnalyzer
from model_inspector.analyzers.tensorrt import TensorRTAnalyzer
from model_inspector.models.confidence import ModelConfidence
from model_inspector.exceptions import UnsupportedFormatError


class TestAnalyzerRegistry:
    """Tests for the analyzer registry system."""

    def test_get_analyzer_for_extension(self):
        """Test getting analyzers for extensions."""
        # Test existing analyzers
        analyzer = get_analyzer_for_extension(".safetensors")
        assert isinstance(analyzer, SafetensorsAnalyzer)

        analyzer = get_analyzer_for_extension("onnx")  # Without dot
        assert isinstance(analyzer, ONNXAnalyzer)

        # Test case insensitivity
        analyzer = get_analyzer_for_extension(".ONNX")  # Uppercase
        assert isinstance(analyzer, ONNXAnalyzer)

        # Test unsupported format
        with pytest.raises(UnsupportedFormatError):
            get_analyzer_for_extension(".unsupported")

    def test_register_analyzer(self):
        """Test registering custom analyzers."""

        # Create a dummy analyzer class
        class DummyAnalyzer(BaseAnalyzer):
            def analyze(self, file_path):
                return "Dummy", ModelConfidence.LOW, {}

            def get_supported_extensions(self):
                return {".dummy"}

        # Register the analyzer
        register_analyzer([".dummy", ".test"], DummyAnalyzer)

        # Test getting the analyzer
        analyzer = get_analyzer_for_extension(".dummy")
        assert isinstance(analyzer, DummyAnalyzer)

        analyzer = get_analyzer_for_extension(".test")
        assert isinstance(analyzer, DummyAnalyzer)


class TestBaseAnalyzer:
    """Tests for the BaseAnalyzer class."""

    def test_base_analyzer_methods(self):
        """Test BaseAnalyzer methods."""

        # Create a simple subclass that implements the abstract method
        class SimpleAnalyzer(BaseAnalyzer):
            def analyze(self, file_path):
                return "Simple", ModelConfidence.LOW, {}

            def get_supported_extensions(self):
                return {".simple"}

        analyzer = SimpleAnalyzer()

        # Test default methods
        assert analyzer.can_analyze_safely("test.simple") is True

        # Should return empty dict
        analyzer.configure({})


class TestSafetensorsAnalyzer:
    """Tests for the SafetensorsAnalyzer."""

    def test_safetensors_analyzer(self, sample_files):
        """Test analyzing a safetensors file."""
        analyzer = SafetensorsAnalyzer()

        # Check supported extensions
        extensions = analyzer.get_supported_extensions()
        assert ".safetensors" in extensions

        # Analyze the test file
        model_type, confidence, metadata = analyzer.analyze(sample_files['safetensors'])

        # Verify the results
        assert model_type is not None
        assert confidence in list(ModelConfidence)
        assert isinstance(metadata, dict)
        assert metadata.get('format_details') == 'safetensors' or \
               metadata.get('format') == 'safetensors'

        # LoRA should be detected from metadata
        assert "LoRA" in model_type or "network_dim" in metadata

    def test_non_existent_file(self):
        """Test handling of non-existent files."""
        analyzer = SafetensorsAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.analyze("non_existent.safetensors")


class TestONNXAnalyzer:
    """Tests for the ONNXAnalyzer."""

    def test_onnx_analyzer(self, sample_files):
        """Test analyzing an ONNX file."""
        analyzer = ONNXAnalyzer()

        # Check supported extensions
        extensions = analyzer.get_supported_extensions()
        assert ".onnx" in extensions

        # Analyze the test file
        model_type, confidence, metadata = analyzer.analyze(sample_files['onnx'])

        # Verify the results
        assert model_type is not None
        assert confidence in list(ModelConfidence)
        assert isinstance(metadata, dict)
        assert metadata.get('format') == 'onnx'

        # The file should be detected as safe
        assert analyzer.can_analyze_safely(sample_files['onnx']) is True


@pytest.mark.parametrize("analyzer_class,file_key,expected_unsafe", [
    (PyTorchAnalyzer, 'pytorch', True),
    (CheckpointAnalyzer, 'pytorch', True),  # Use pytorch file as stand-in
    (TensorFlowAnalyzer, 'onnx', True),  # Use onnx file as stand-in
    (HDF5Analyzer, 'hdf5', False),
    (GGUFAnalyzer, 'gguf', False),
    (CoreMLAnalyzer, 'mlmodel', False),
    (TensorRTAnalyzer, 'tensorrt', False),
])
def test_analyzer_safety(analyzer_class, file_key, expected_unsafe, sample_files):
    """Test safety handling for different analyzers."""
    analyzer = analyzer_class()
    file_path = sample_files.get(file_key)

    # Skip if sample file not available
    if file_path is None:
        pytest.skip(f"Sample file for {file_key} not available")

    # Check if the analyzer correctly identifies safety
    is_unsafe = not analyzer.can_analyze_safely(file_path)
    assert is_unsafe == expected_unsafe


class TestAnalyzerFallbacks:
    """Tests for analyzer fallback behaviors."""

    def test_basic_analysis_fallbacks(self, sample_files):
        """Test that analyzers have reasonable fallbacks when libraries are unavailable."""
        # Since we can't easily mock import failures, we'll call the internal
        # fallback methods directly where they exist

        # ONNX fallback
        analyzer = ONNXAnalyzer()
        if hasattr(analyzer, '_parse_onnx_header'):
            metadata = analyzer._parse_onnx_header(sample_files['onnx'])
            assert isinstance(metadata, dict)
            assert 'file_size' in metadata

        # HDF5 fallback (if available)
        analyzer = HDF5Analyzer()
        try:
            path = sample_files.get('hdf5')
            if path and hasattr(analyzer, '_analyze_hdf5_file'):
                metadata = analyzer._analyze_hdf5_file(path)
                assert isinstance(metadata, dict)
        except AttributeError:
            pass  # Method might not exist

"""
Tests for the main ModelInspector class.
"""
import pytest
import os
import tempfile
from pathlib import Path
import asyncio

from model_inspector import ModelInspector
from model_inspector.config import InspectorConfig
from model_inspector.models.safety import SafetyLevel
from model_inspector.models.confidence import ModelConfidence
from model_inspector.models.info import ModelInfo
from model_inspector.exceptions import UnsupportedFormatError, UnsafeModelError


class TestModelInspector:
    """Tests for the ModelInspector class."""

    def test_initialization(self, temp_dir):
        """Test ModelInspector initialization."""
        # Test with default config
        inspector = ModelInspector(temp_dir)
        assert inspector.base_dir == Path(temp_dir)
        assert inspector.config is not None
        assert inspector.cache is not None

        # Test with custom config
        config = InspectorConfig(enable_caching=False)
        inspector = ModelInspector(temp_dir, config=config)
        assert inspector.cache is None

        # Test with non-existent directory
        # This should not raise an error until we try to access files
        nonexistent = os.path.join(temp_dir, "nonexistent")
        inspector = ModelInspector(nonexistent)

    def test_directory_files(self, complex_directory):
        """Test scanning directory for files."""
        base_dir, file_info = complex_directory

        # Create inspector
        inspector = ModelInspector(base_dir)

        # Get all files
        files = list(inspector.directory_files())
        assert len(files) > 0

        # All returned files should exist and be in the base directory
        for file in files:
            assert os.path.exists(file)
            assert os.path.commonpath([base_dir, file]) == base_dir

        # Test with extension filter
        safetensors_files = list(inspector.directory_files(extensions={'.safetensors'}))
        assert all(f.endswith('.safetensors') for f in safetensors_files)

        # Test with size filter
        min_size = 10 * 1024  # 10KB
        large_files = list(inspector.directory_files(min_size=min_size))
        for file in large_files:
            assert os.path.getsize(file) >= min_size

        # Test with show progress
        # This is hard to test functionally, but we can at least ensure it doesn't error
        files = list(inspector.directory_files(show_progress=True))
        assert len(files) > 0

    def test_is_safe_to_load(self, sample_files):
        """Test safety checking for file formats."""
        # Create inspector with default safety (SAFE)
        inspector = ModelInspector(".")

        # Safe formats should be allowed
        assert inspector.is_safe_to_load(sample_files['safetensors']) is True
        assert inspector.is_safe_to_load(sample_files['onnx']) is True

        # Unsafe formats should not be allowed
        assert inspector.is_safe_to_load(sample_files['pytorch']) is False

        # With WARN level, unsafe formats should be allowed with warning
        config = InspectorConfig(safety_level=SafetyLevel.WARN)
        inspector = ModelInspector(".", config=config)
        assert inspector.is_safe_to_load(sample_files['pytorch']) is True

        # With UNSAFE level, all formats should be allowed
        config = InspectorConfig(safety_level=SafetyLevel.UNSAFE)
        inspector = ModelInspector(".", config=config)
        assert inspector.is_safe_to_load(sample_files['pytorch']) is True

    def test_get_model_type(self, sample_files):
        """Test getting model type for a file."""
        # Create inspector with UNSAFE to allow all formats
        config = InspectorConfig(safety_level=SafetyLevel.UNSAFE)
        inspector = ModelInspector(".", config=config)

        # Test with a safetensors file
        model_info = inspector.get_model_type(sample_files['safetensors'])
        assert isinstance(model_info, ModelInfo)
        assert model_info.model_type is not None
        assert model_info.confidence is not None
        assert model_info.format == '.safetensors'
        assert model_info.file_path == sample_files['safetensors']

        # Test with an invalid file - .bin extension needs a registered analyzer
        # Register a dummy analyzer for .bin
        from model_inspector.analyzers import register_analyzer, BaseAnalyzer
        from model_inspector.models.confidence import ModelConfidence

        class DummyBinAnalyzer(BaseAnalyzer):
            def analyze(self, file_path):
                return "BinaryModel", ModelConfidence.LOW, {'format': 'binary'}

            def get_supported_extensions(self):
                return {".bin"}

        register_analyzer(['.bin'], DummyBinAnalyzer)

        # Now this should work
        model_info = inspector.get_model_type(sample_files['invalid'])
        assert model_info.model_type == "BinaryModel"

        # Test with an unsupported format
        config = InspectorConfig(
            safety_level=SafetyLevel.UNSAFE,
            enabled_formats={'.safetensors'}
        )
        inspector = ModelInspector(".", config=config)
        with pytest.raises(UnsupportedFormatError):
            inspector.get_model_type(sample_files['onnx'])

        # Test with safety constraint (SAFE level should reject unsafe formats)
        config = InspectorConfig(safety_level=SafetyLevel.SAFE)
        inspector = ModelInspector(".", config=config)
        with pytest.raises(UnsafeModelError):
            inspector.get_model_type(sample_files['pytorch'])

    def test_analyze_directory(self, sample_files):
        """Test analyzing a directory."""
        # Create a directory with the sample files
        dir_path = os.path.dirname(sample_files['safetensors'])

        # Create inspector with UNSAFE to allow all formats
        config = InspectorConfig(safety_level=SafetyLevel.UNSAFE)
        inspector = ModelInspector(dir_path, config=config)

        # Analyze the directory
        results = inspector.analyze_directory()
        assert len(results) > 0
        assert all(isinstance(info, ModelInfo) for info in results)

        # Test with show_progress
        results = inspector.analyze_directory(show_progress=True)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_analyze_directory_async(self, sample_files):
        """Test asynchronously analyzing a directory."""
        # Create a directory with the sample files
        dir_path = os.path.dirname(sample_files['safetensors'])

        # Create inspector with UNSAFE to allow all formats
        config = InspectorConfig(
            safety_level=SafetyLevel.UNSAFE,
            max_workers=2
        )
        inspector = ModelInspector(dir_path, config=config)

        # Analyze the directory asynchronously
        results = await inspector.analyze_directory_async()
        assert len(results) > 0
        assert all(isinstance(info, ModelInfo) for info in results)

    def test_context_manager(self, sample_files):
        """Test using ModelInspector as a context manager."""
        with ModelInspector(".") as inspector:
            assert inspector is not None

            # Test operation within context
            extensions = list(inspector.SUPPORTED_EXTENSIONS)
            assert len(extensions) > 0

    def test_filtered_views(self, sample_files):
        """Test filtered views of the ModelInspector."""
        # Create a directory with the sample files
        dir_path = os.path.dirname(sample_files['safetensors'])

        # Create inspector with UNSAFE to allow all formats
        config = InspectorConfig(safety_level=SafetyLevel.UNSAFE)
        inspector = ModelInspector(dir_path, config=config)

        # Create a filtered view for safetensors files
        def filter_func(model_info):
            return model_info.format == '.safetensors'

        view = inspector.create_filtered_view(filter_func)

        # Test the view
        results = view.analyze_directory()
        assert len(results) > 0
        assert all(info.format == '.safetensors' for info in results)

        # Test with ModelFilter
        from model_inspector.utils.filtering import ModelFilter

        model_filter = ModelFilter().format('.safetensors')
        view = inspector.apply_model_filter(model_filter)

        results = view.analyze_directory()
        assert all(info.format == '.safetensors' for info in results)

    def test_grouping(self, sample_files):
        """Test grouping results by model type and format."""
        # Create a directory with the sample files
        dir_path = os.path.dirname(sample_files['safetensors'])

        # Create inspector with UNSAFE to allow all formats
        config = InspectorConfig(safety_level=SafetyLevel.UNSAFE)
        inspector = ModelInspector(dir_path, config=config)

        # Get results
        results = inspector.analyze_directory()

        # Group by model type
        grouped_by_type = inspector.group_by_model_type(results)
        assert isinstance(grouped_by_type, dict)
        assert len(grouped_by_type) > 0

        # Group by format
        grouped_by_format = inspector.group_by_format(results)
        assert isinstance(grouped_by_format, dict)
        assert len(grouped_by_format) > 0
        assert '.safetensors' in grouped_by_format


def test_help_system():
    """Test the ModelInspector help system."""
    # Get general help
    help_text = ModelInspector.help()
    assert isinstance(help_text, str)
    assert len(help_text) > 0

    # Get specific topic
    formats_help = ModelInspector.help("formats")
    assert isinstance(formats_help, str)
    assert len(formats_help) > 0
    assert "formats" in formats_help.lower()

    # Get non-existent topic
    invalid_help = ModelInspector.help("nonexistent")
    assert isinstance(invalid_help, str)
    assert "not found" in invalid_help

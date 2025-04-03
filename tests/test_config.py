"""
Tests for the configuration system in the model_inspector library.
"""
import pytest
from model_inspector.config import InspectorConfig
from model_inspector.models.safety import SafetyLevel
from model_inspector.models.permissions import Permission, PermissionSet
from model_inspector.utils.progress import ProgressFormat


class TestInspectorConfig:
    """Tests for the InspectorConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InspectorConfig()

        # Check default values
        assert config.recursive is True
        assert config.max_workers == 4
        assert config.safety_level == SafetyLevel.SAFE
        assert config.enabled_formats is None
        assert config.disabled_formats == set()
        assert config.min_size == 0
        assert config.max_size is None
        assert config.enable_caching is True
        assert config.cache_size == 1000

    def test_custom_config(self):
        """Test setting custom configuration values."""
        config = InspectorConfig(
            recursive=False,
            max_workers=8,
            safety_level=SafetyLevel.WARN,
            enabled_formats={'.safetensors', '.onnx'},
            disabled_formats={'.bin'},
            min_size=1024,
            max_size=1024 * 1024,
            enable_caching=False
        )

        # Check custom values
        assert config.recursive is False
        assert config.max_workers == 8
        assert config.safety_level == SafetyLevel.WARN
        assert config.enabled_formats == {'.safetensors', '.onnx'}
        assert config.disabled_formats == {'.bin'}
        assert config.min_size == 1024
        assert config.max_size == 1024 * 1024
        assert config.enable_caching is False

    def test_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = InspectorConfig(max_workers=2, min_size=100, max_size=200)

        # Test invalid configurations
        with pytest.raises(ValueError):
            InspectorConfig(max_workers=0)

        with pytest.raises(ValueError):
            InspectorConfig(min_size=-1)

        with pytest.raises(ValueError):
            InspectorConfig(min_size=200, max_size=100)

        with pytest.raises(ValueError):
            InspectorConfig(cache_size=-10)

        with pytest.raises(ValueError):
            InspectorConfig(enable_sandbox=True, sandbox_type="invalid")

    def test_format_enabled(self):
        """Test checking if formats are enabled."""
        # With no restrictions, all formats should be enabled
        config = InspectorConfig()
        assert config.is_format_enabled('.safetensors') is True
        assert config.is_format_enabled('.onnx') is True
        assert config.is_format_enabled('pt') is True  # Without dot

        # With enabled_formats set, only those formats should be enabled
        config = InspectorConfig(enabled_formats={'.safetensors', '.onnx'})
        assert config.is_format_enabled('.safetensors') is True
        assert config.is_format_enabled('.onnx') is True
        assert config.is_format_enabled('.pt') is False

        # With disabled_formats set, those formats should be disabled
        config = InspectorConfig(disabled_formats={'.pt'})
        assert config.is_format_enabled('.safetensors') is True
        assert config.is_format_enabled('.pt') is False
        assert config.is_format_enabled('pt') is False  # Without dot

        # Disabled_formats takes precedence over enabled_formats
        config = InspectorConfig(
            enabled_formats={'.safetensors', '.onnx', '.pt'},
            disabled_formats={'.pt'}
        )
        assert config.is_format_enabled('.pt') is False

    def test_get_progress_config(self):
        """Test getting progress configuration."""
        # Test default
        config = InspectorConfig()
        progress_config = config.get_progress_config()
        assert progress_config.format == ProgressFormat.BAR

        # Test custom format
        config = InspectorConfig(progress_format='spinner')
        progress_config = config.get_progress_config()
        assert progress_config.format == ProgressFormat.SPINNER

        # Test invalid format falls back to default
        config = InspectorConfig(progress_format='invalid')
        progress_config = config.get_progress_config()
        assert progress_config.format == ProgressFormat.BAR

    def test_default_permissions(self):
        """Test default permissions based on safety level."""
        # Safe level should have restricted permissions
        config = InspectorConfig(safety_level=SafetyLevel.SAFE)
        perms = config._get_default_permissions()
        assert Permission.READ_FILE in perms.base_permissions
        assert Permission.READ_METADATA in perms.base_permissions
        assert Permission.EXECUTE_CODE not in perms.base_permissions

        # Warn level should have more permissions but not execute code
        config = InspectorConfig(safety_level=SafetyLevel.WARN)
        perms = config._get_default_permissions()
        assert Permission.READ_FILE in perms.base_permissions
        assert Permission.READ_WEIGHTS in perms.base_permissions
        assert Permission.EXECUTE_CODE not in perms.base_permissions

        # Unsafe level should have all permissions
        config = InspectorConfig(safety_level=SafetyLevel.UNSAFE)
        perms = config._get_default_permissions()
        assert Permission.EXECUTE_CODE in perms.base_permissions
        assert Permission.NETWORK_ACCESS in perms.base_permissions

"""
Tests for model data classes in the model_inspector library.
"""
import pytest
from model_inspector.models.safety import SafetyLevel
from model_inspector.models.confidence import ModelConfidence
from model_inspector.models.info import ModelInfo
from model_inspector.models.permissions import Permission, PermissionSet


class TestSafetyLevel:
    """Tests for the SafetyLevel enum."""

    def test_enum_values(self):
        """Test that the enum has the expected values."""
        assert SafetyLevel.SAFE.value == 0
        assert SafetyLevel.WARN.value == 1
        assert SafetyLevel.UNSAFE.value == 2

    def test_comparison(self):
        """Test that safety levels can be compared."""
        assert SafetyLevel.SAFE < SafetyLevel.WARN
        assert SafetyLevel.WARN < SafetyLevel.UNSAFE
        assert SafetyLevel.SAFE < SafetyLevel.UNSAFE


class TestModelConfidence:
    """Tests for the ModelConfidence enum."""

    def test_enum_values(self):
        """Test that the enum has the expected values."""
        assert ModelConfidence.UNKNOWN.value == 0
        assert ModelConfidence.LOW.value == 1
        assert ModelConfidence.MEDIUM.value == 2
        assert ModelConfidence.HIGH.value == 3

    def test_comparison(self):
        """Test that confidence levels can be compared."""
        assert ModelConfidence.UNKNOWN < ModelConfidence.LOW
        assert ModelConfidence.LOW < ModelConfidence.MEDIUM
        assert ModelConfidence.MEDIUM < ModelConfidence.HIGH
        assert ModelConfidence.UNKNOWN < ModelConfidence.HIGH


class TestModelInfo:
    """Tests for the ModelInfo class."""

    def test_creation(self):
        """Test creating a ModelInfo object."""
        info = ModelInfo(
            model_type="TestModel",
            confidence=ModelConfidence.HIGH,
            format=".test",
            metadata={"key": "value"},
            file_path="/path/to/model.test",
            file_size=1024,
            is_safe=True
        )

        assert info.model_type == "TestModel"
        assert info.confidence == ModelConfidence.HIGH
        assert info.format == ".test"
        assert info.metadata == {"key": "value"}
        assert info.file_path == "/path/to/model.test"
        assert info.file_size == 1024
        assert info.is_safe is True

    def test_properties(self):
        """Test ModelInfo properties."""
        info = ModelInfo(
            model_type="TestModel",
            confidence=ModelConfidence.HIGH,
            format=".test",
            metadata={"key": "value"},
            file_path="/path/to/dir/model.test",
            file_size=1024,
            is_safe=True
        )

        assert info.filename == "model.test"
        assert info.extension == ".test"
        assert info.is_high_confidence is True

    def test_string_representation(self):
        """Test string representation of ModelInfo."""
        info = ModelInfo(
            model_type="TestModel",
            confidence=ModelConfidence.HIGH,
            format=".test",
            metadata={"key": "value"},
            file_path="/path/to/model.test",
            file_size=1024,
            is_safe=True
        )

        assert str(info) == "model.test: TestModel (HIGH)"


class TestPermission:
    """Tests for the Permission enum."""

    def test_basic_permissions(self):
        """Test basic permission attributes and methods."""
        assert Permission.READ_FILE in Permission.safe_set()
        assert Permission.EXECUTE_CODE not in Permission.safe_set()

        assert Permission.READ_FILE in Permission.default_set()
        assert Permission.READ_METADATA in Permission.default_set()

        # All permissions should include everything
        all_perms = Permission.all_permissions()
        for perm in Permission:
            assert perm in all_perms


class TestPermissionSet:
    """Tests for the PermissionSet class."""

    def test_creation(self):
        """Test creating a PermissionSet."""
        # Default permissions
        perm_set = PermissionSet()
        assert perm_set.base_permissions == Permission.default_set()
        assert perm_set.format_permissions == {}

        # Custom permissions
        custom_perms = {Permission.READ_FILE, Permission.READ_METADATA}
        perm_set = PermissionSet(base_permissions=custom_perms)
        assert perm_set.base_permissions == custom_perms

    def test_permission_checking(self):
        """Test checking permissions."""
        perm_set = PermissionSet()

        # Default permissions should include READ_FILE
        assert perm_set.has_permission(Permission.READ_FILE)

        # Remove the permission
        perm_set.set_permission(Permission.READ_FILE, False)
        assert not perm_set.has_permission(Permission.READ_FILE)

        # Add it back
        perm_set.set_permission(Permission.READ_FILE, True)
        assert perm_set.has_permission(Permission.READ_FILE)

    def test_format_permissions(self):
        """Test format-specific permissions."""
        perm_set = PermissionSet()

        # By default, format should inherit from base
        assert perm_set.format_has_permission('.pt', Permission.READ_FILE)

        # Set format-specific permission
        perm_set.set_format_permission('.pt', Permission.EXECUTE_CODE, True)
        assert perm_set.format_has_permission('.pt', Permission.EXECUTE_CODE)

        # Other formats should not have this permission
        assert not perm_set.format_has_permission('.onnx', Permission.EXECUTE_CODE)

        # Remove the permission
        perm_set.set_format_permission('.pt', Permission.EXECUTE_CODE, False)
        assert not perm_set.format_has_permission('.pt', Permission.EXECUTE_CODE)

    def test_factory_methods(self):
        """Test PermissionSet factory methods."""
        safe_perms = PermissionSet.create_safe_permissions()
        assert safe_perms.base_permissions == Permission.safe_set()

        unrestricted = PermissionSet.create_unrestricted_permissions()
        assert unrestricted.base_permissions == Permission.all_permissions()

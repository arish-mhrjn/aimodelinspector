# tests/test_exceptions.py
"""
Tests for the exceptions in the model_inspector library.
"""
import pytest

from model_inspector.exceptions import (
    ModelInspectorError, UnsupportedFormatError,
    UnsafeModelError, ModelAnalysisError,
    InvalidConfigurationError, FileAccessError,
    SecurityViolationError, SandboxError,
    PermissionError, CacheError
)


class TestExceptions:
    """Tests for the exception classes."""

    def test_exception_hierarchy(self):
        """Test the exception class hierarchy."""
        # All exceptions should inherit from ModelInspectorError
        exceptions = [
            UnsupportedFormatError,
            UnsafeModelError,
            ModelAnalysisError,
            InvalidConfigurationError,
            FileAccessError,
            SecurityViolationError,
            SandboxError,
            PermissionError,
            CacheError
        ]

        for exception_class in exceptions:
            # Create an instance
            instance = exception_class("Test message")

            # Check inheritance
            assert isinstance(instance, ModelInspectorError)
            assert isinstance(instance, Exception)

            # Check string representation
            assert str(instance) == "Test message"

    def test_exception_usage(self):
        """Test using the exceptions in try/except blocks."""
        # Test catching specific exceptions
        try:
            raise UnsupportedFormatError("Unsupported format")
        except UnsupportedFormatError as e:
            assert str(e) == "Unsupported format"

        # Test catching parent exception
        try:
            raise UnsafeModelError("Unsafe model")
        except ModelInspectorError as e:
            assert str(e) == "Unsafe model"

        # Test re-raising
        with pytest.raises(SecurityViolationError):
            try:
                raise SecurityViolationError("Security violation")
            except ModelInspectorError:
                raise

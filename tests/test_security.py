# tests/test_security.py
"""
Tests for security features including sandbox and permissions.
"""
import pytest
import os
import tempfile
from pathlib import Path
import time

from model_inspector.models.permissions import Permission, PermissionSet
from model_inspector.sandbox import (
    Sandbox, InProcessSandbox, SandboxFactory, HAS_RESOURCE
)
from model_inspector.exceptions import SecurityViolationError, SandboxError

# Import ProcessSandbox only if resource module is available
if HAS_RESOURCE:
    from model_inspector.sandbox import ProcessSandbox


class TestSandbox:
    """Tests for the sandbox system."""

    def test_base_sandbox(self):
        """Test the base Sandbox class."""
        # Create a permission set
        perms = PermissionSet()
        perms.set_permission(Permission.READ_FILE, True)
        perms.set_permission(Permission.EXECUTE_CODE, False)

        # Create a sandbox
        sandbox = Sandbox(perms)

        # Test permission checking
        assert sandbox.check_permission(Permission.READ_FILE) is True
        assert sandbox.check_permission(Permission.EXECUTE_CODE) is False

        # Test format-specific permissions
        perms.set_format_permission('.pt', Permission.EXECUTE_CODE, True)
        assert sandbox.check_format_permission('.pt', Permission.EXECUTE_CODE) is True
        assert sandbox.check_format_permission('.onnx', Permission.EXECUTE_CODE) is False

        # Test security violations when active
        sandbox.enter()
        assert sandbox.active is True

        # Should be allowed
        sandbox.check_permission(Permission.READ_FILE)

        # Should raise error
        with pytest.raises(SecurityViolationError):
            sandbox.check_permission(Permission.EXECUTE_CODE)

        with pytest.raises(SecurityViolationError):
            sandbox.check_format_permission('.onnx', Permission.EXECUTE_CODE)

        # Test context manager
        sandbox.exit()
        assert sandbox.active is False

        with sandbox.active_session():
            assert sandbox.active is True

        assert sandbox.active is False


class TestInProcessSandbox:
    """Tests for the in-process sandbox."""

    def test_in_process_sandbox(self, temp_dir):
        """Test the in-process sandbox."""
        # Create a permission set that disallows writing
        perms = PermissionSet()
        perms.set_permission(Permission.WRITE_FILE, False)

        # Create the sandbox
        sandbox = InProcessSandbox(perms)

        # Remember the original directory
        original_dir = os.getcwd()

        # Use the sandbox
        with sandbox.active_session():
            assert sandbox.active is True

            # Directory should have changed
            current_dir = os.getcwd()
            assert current_dir != original_dir
            assert current_dir == sandbox.temp_dir

            # We should still be able to read but not write
            with pytest.raises(SecurityViolationError):
                sandbox.check_permission(Permission.WRITE_FILE)

        # Directory should be restored
        assert os.getcwd() == original_dir
        assert sandbox.temp_dir is None


@pytest.mark.skipif(not HAS_RESOURCE, reason="Resource module not available on this platform")
class TestProcessSandbox:
    """Tests for the process sandbox."""

    def test_process_sandbox_function(self):
        """Test running a function in a process sandbox."""
        # Create a permission set
        perms = PermissionSet.create_safe_permissions()

        # Create the sandbox
        sandbox = ProcessSandbox(perms)

        # Define a test function to run in the sandbox
        def test_func(a, b):
            return a + b

        # Run the function
        result = sandbox.run_function(test_func, 2, 3)
        assert result == 5

        # Test with timeout
        def slow_func():
            time.sleep(0.5)
            return 42

        # Should complete normally
        result = sandbox.run_function(slow_func, timeout=1)
        assert result == 42

        # Should timeout
        with pytest.raises(SandboxError):
            sandbox.run_function(slow_func, timeout=0.1)

        # Test with exception in function
        def error_func():
            raise ValueError("Test error")

        with pytest.raises(SandboxError):
            sandbox.run_function(error_func)


class TestSandboxFactory:
    """Tests for the SandboxFactory."""

    def test_sandbox_factory(self):
        """Test creating sandboxes with the factory."""
        # Create different types of sandboxes
        in_process = SandboxFactory.create_sandbox("inprocess")
        assert isinstance(in_process, InProcessSandbox)

        if HAS_RESOURCE:
            process = SandboxFactory.create_sandbox("process")
            assert isinstance(process, ProcessSandbox)

        # Test with custom permissions
        perms = PermissionSet.create_safe_permissions()
        sandbox = SandboxFactory.create_sandbox("inprocess", permissions=perms)
        assert sandbox.permissions == perms

        # Test with invalid type
        with pytest.raises(ValueError):
            SandboxFactory.create_sandbox("invalid")

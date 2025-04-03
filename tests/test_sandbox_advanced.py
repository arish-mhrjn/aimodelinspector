# tests/test_sandbox_advanced.py
"""
Advanced tests for the sandbox system to improve coverage.
"""
import pytest
import os
import tempfile
import time
import multiprocessing
from pathlib import Path
import sys

from model_inspector.models.permissions import Permission, PermissionSet
from model_inspector.sandbox import (
    Sandbox, InProcessSandbox, ProcessSandbox,
    SandboxFactory, ContainerSandbox,
    SandboxError, SecurityViolationError, HAS_RESOURCE
)

# Import resource only if available
if HAS_RESOURCE:
    import resource


class TestSandboxAdvanced:
    """Advanced tests for the sandbox system."""

    def test_sandbox_permission_checking(self):
        """Test permission checking in the Sandbox class."""
        # Create a permission set with mixed permissions
        perms = PermissionSet()
        perms.set_permission(Permission.READ_FILE, True)
        perms.set_permission(Permission.EXECUTE_CODE, False)

        # Create a sandbox
        sandbox = Sandbox(perms)

        # Test inactive sandbox
        assert sandbox.check_permission(Permission.READ_FILE) is True
        assert sandbox.check_permission(Permission.EXECUTE_CODE) is False

        # Test active sandbox - should raise for denied permissions
        sandbox.enter()
        assert sandbox.check_permission(Permission.READ_FILE) is True

        with pytest.raises(SecurityViolationError):
            sandbox.check_permission(Permission.EXECUTE_CODE)

        # Test format-specific permissions
        perms.set_format_permission('.pt', Permission.EXECUTE_CODE, True)
        perms.set_format_permission('.onnx', Permission.EXECUTE_CODE, False)

        assert sandbox.check_format_permission('.pt', Permission.EXECUTE_CODE) is True

        with pytest.raises(SecurityViolationError):
            sandbox.check_format_permission('.onnx', Permission.EXECUTE_CODE)

        # Test context manager error handling
        try:
            with sandbox.active_session():
                raise ValueError("Test error")
        except ValueError:
            # The exception should propagate
            pass

        # Sandbox should be inactive after context exit
        assert sandbox.active is False

    def test_inprocess_sandbox_directory_isolation(self, temp_dir):
        """Test directory isolation in the InProcessSandbox."""
        # Create a permission set that disallows writing
        perms = PermissionSet()
        perms.set_permission(Permission.WRITE_FILE, False)

        # Create the sandbox
        sandbox = InProcessSandbox(perms)

        # Remember original directory
        original_dir = os.getcwd()

        # Enter the sandbox
        sandbox.enter()

        # Directory should have changed
        temp_dir = sandbox.temp_dir
        assert os.getcwd() == temp_dir

        # Create a test file in the temp directory
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test content")

        # Exit the sandbox
        sandbox.exit()

        # Directory should be restored
        assert os.getcwd() == original_dir

        # Temp directory should be cleaned up (may fail on Windows)
        if os.name != 'nt':  # Skip on Windows
            assert not os.path.exists(temp_dir)

        # Test without directory isolation
        perms.set_permission(Permission.WRITE_FILE, True)
        sandbox = InProcessSandbox(perms)

        # Enter the sandbox
        sandbox.enter()

        # Directory should not change when writing is allowed
        assert sandbox.temp_dir is None
        assert os.getcwd() == original_dir

        # Exit the sandbox
        sandbox.exit()

    @pytest.mark.skipif(not HAS_RESOURCE, reason="Resource module not available on this platform")
    def test_process_sandbox_function_execution(self):
        """Test executing functions in the process sandbox."""
        # Create a permission set
        perms = PermissionSet()

        # Create resource limits
        resource_limits = {
            resource.RLIMIT_CPU: 1,  # 1 second CPU time
        }

        # Create the sandbox
        sandbox = ProcessSandbox(perms, resource_limits=resource_limits)

        # Define test functions
        def add(a, b):
            return a + b

        def use_kwargs(**kwargs):
            return kwargs

        def cpu_intensive():
            """Function that uses a lot of CPU time."""
            result = 0
            for i in range(10000000):
                result += i
            return result

        def raises_exception():
            raise ValueError("Test exception")

        def access_unavailable_module():
            # Try to import a module that likely doesn't exist
            import non_existent_module
            return True

        # Test basic function execution
        result = sandbox.run_function(add, 2, 3)
        assert result == 5

        # Test with keyword arguments
        kwargs = {'a': 1, 'b': 2, 'c': 3}
        result = sandbox.run_function(use_kwargs, **kwargs)
        assert result == kwargs

        # Test CPU limit (should raise due to timeout or resource limit)
        with pytest.raises(SandboxError):
            sandbox.run_function(cpu_intensive, timeout=1)

        # Test exception propagation
        with pytest.raises(SandboxError) as excinfo:
            sandbox.run_function(raises_exception)
        assert "Test exception" in str(excinfo.value)

        # Test import error handling
        with pytest.raises(SandboxError) as excinfo:
            sandbox.run_function(access_unavailable_module)
        # The error could be an ImportError or a RuntimeError depending on the environment
        assert "Error in sandboxed function" in str(excinfo.value)

    def test_sandbox_factory_basic(self):
        """Test creating sandboxes with basic options."""
        # Test creating process sandbox
        process_sandbox = SandboxFactory.create_sandbox("process")
        assert isinstance(process_sandbox, ProcessSandbox)

        # Test creating in-process sandbox
        in_process = SandboxFactory.create_sandbox("inprocess")
        assert isinstance(in_process, InProcessSandbox)

        # Test with custom permissions
        safe_perms = PermissionSet.create_safe_permissions()
        sandbox = SandboxFactory.create_sandbox("inprocess", permissions=safe_perms)
        assert sandbox.permissions == safe_perms

        # Test with invalid type
        with pytest.raises(ValueError):
            SandboxFactory.create_sandbox("invalid_type")

    @pytest.mark.skipif(not HAS_RESOURCE, reason="Resource module not available on this platform")
    def test_sandbox_factory_options(self):
        """Test creating sandboxes with various options."""
        # Test creating process sandbox with custom options
        process_sandbox = SandboxFactory.create_sandbox(
            "process",
            resource_limits={
                resource.RLIMIT_CPU: 10,
                resource.RLIMIT_DATA: 1024 * 1024 * 100  # 100MB
            }
        )

        assert isinstance(process_sandbox, ProcessSandbox)
        assert process_sandbox.resource_limits[resource.RLIMIT_CPU] == 10
        assert process_sandbox.resource_limits[resource.RLIMIT_DATA] == 1024 * 1024 * 100

# model_inspector/sandbox.py
from __future__ import annotations
import os
import sys
import tempfile
import threading
import multiprocessing
import time
import signal
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple, Set, Union
from contextlib import contextmanager
import subprocess

from .models.permissions import Permission, PermissionSet
from .exceptions import SecurityViolationError, SandboxError

logger = logging.getLogger(__name__)

# Import resource module if available (Unix-like systems)
try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


    # Create dummy resource constants for Windows
    class DummyResource:
        RLIMIT_CPU = 0
        RLIMIT_DATA = 1
        RLIMIT_FSIZE = 2
        RLIMIT_NOFILE = 3


    resource = DummyResource()


class Sandbox:
    """
    Base class for sandboxes that isolate potentially unsafe operations.

    Sandboxes provide a controlled environment for running operations that
    could be risky, such as loading models that might contain executable code.
    """

    def __init__(self, permissions: PermissionSet):
        """
        Initialize the sandbox with the specified permissions.

        Args:
            permissions: Permissions granted to the sandbox
        """
        self.permissions = permissions
        self.active = False

    def check_permission(self, permission: Permission) -> bool:
        """
        Check if the sandbox has a specific permission.

        Args:
            permission: Permission to check

        Returns:
            True if the permission is granted

        Raises:
            SecurityViolationError: If the sandbox is active and the permission is denied
        """
        has_permission = self.permissions.has_permission(permission)

        if self.active and not has_permission:
            logger.warning(f"Security violation: {permission.name} not permitted in this sandbox")
            raise SecurityViolationError(f"Operation requires permission: {permission.name}")

        return has_permission

    def check_format_permission(self, format_name: str, permission: Permission) -> bool:
        """
        Check if the sandbox has a specific permission for a format.

        Args:
            format_name: File format to check
            permission: Permission to check

        Returns:
            True if the permission is granted

        Raises:
            SecurityViolationError: If the sandbox is active and the permission is denied
        """
        has_permission = self.permissions.format_has_permission(format_name, permission)

        if self.active and not has_permission:
            logger.warning(
                f"Security violation: {permission.name} not permitted for format {format_name}"
            )
            raise SecurityViolationError(
                f"Operation requires permission: {permission.name} for format: {format_name}"
            )

        return has_permission

    def enter(self) -> None:
        """
        Enter the sandbox - activate its restrictions.
        """
        self.active = True

    def exit(self) -> None:
        """
        Exit the sandbox - deactivate its restrictions.
        """
        self.active = False

    @contextmanager
    def active_session(self) -> None:
        """
        Context manager to activate the sandbox for a block of code.

        Yields:
            None

        Raises:
            SandboxError: If an error occurs during setup or teardown
        """
        try:
            self.enter()
            yield
        except Exception as e:
            logger.error(f"Error in sandbox session: {e}")
            raise
        finally:
            self.exit()


class InProcessSandbox(Sandbox):
    """
    A sandbox that applies restrictions within the current process.

    This sandbox provides basic security by checking permissions but does not
    provide strong isolation from the main process.
    """

    def __init__(self, permissions: PermissionSet):
        """
        Initialize the in-process sandbox.

        Args:
            permissions: Permissions granted to the sandbox
        """
        super().__init__(permissions)
        self.original_dir = None
        self.temp_dir = None

    def enter(self) -> None:
        """
        Enter the sandbox, applying restrictions.
        """
        super().enter()

        # Create temporary directory if we don't have full file system access
        if not self.permissions.has_permission(Permission.WRITE_FILE):
            self.original_dir = os.getcwd()
            self.temp_dir = tempfile.mkdtemp(prefix="model_inspector_sandbox_")
            os.chdir(self.temp_dir)

    def exit(self) -> None:
        """
        Exit the sandbox, removing restrictions.
        """
        # Restore original directory if changed
        if self.original_dir is not None:
            os.chdir(self.original_dir)

            # Clean up temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    logger.warning(f"Error cleaning up sandbox directory: {e}")

            self.original_dir = None
            self.temp_dir = None

        super().exit()


class ProcessSandbox(Sandbox):
    """
    A sandbox that runs operations in a separate process.

    This provides stronger isolation by running untrusted operations in a
    separate process with restricted permissions.
    """

    # Default resource limits (UNIX-only)
    DEFAULT_LIMITS = {
        resource.RLIMIT_CPU: 60,  # 60 seconds of CPU time
        resource.RLIMIT_DATA: 1024 * 1024 * 1024,  # 1GB of RAM
        resource.RLIMIT_FSIZE: 100 * 1024 * 1024,  # 100MB file size
        resource.RLIMIT_NOFILE: 100,  # 100 open files
    }

    def __init__(
            self,
            permissions: PermissionSet,
            resource_limits: Optional[Dict[int, int]] = None
    ):
        """
        Initialize the process sandbox.

        Args:
            permissions: Permissions granted to the sandbox
            resource_limits: Optional resource limits to apply (UNIX-only)
        """
        super().__init__(permissions)
        self.resource_limits = resource_limits or self.DEFAULT_LIMITS.copy()

    def run_function(
            self,
            func: Callable,
            *args,
            timeout: int = 60,
            **kwargs
    ) -> Any:
        """
        Run a function in a sandboxed subprocess.

        Args:
            func: Function to run
            *args: Arguments to pass to the function
            timeout: Maximum execution time in seconds
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The return value of the function

        Raises:
            SandboxError: If the function fails or times out
        """
        # Function must be picklable
        if not hasattr(func, '__module__'):
            raise SandboxError("Function must be defined in a module for sandbox execution")

        # Create a pipe for communication
        parent_conn, child_conn = multiprocessing.Pipe()

        # Create a process to run the function
        process = multiprocessing.Process(
            target=self._run_sandboxed,
            args=(func, args, kwargs, child_conn)
        )

        try:
            process.start()

            # Wait for result with timeout
            if parent_conn.poll(timeout):
                success, result = parent_conn.recv()

                if success:
                    return result
                else:
                    raise SandboxError(f"Error in sandboxed function: {result}")
            else:
                # Timeout occurred
                process.terminate()
                raise SandboxError(f"Sandboxed function timed out after {timeout} seconds")

        finally:
            # Clean up
            if process.is_alive():
                process.terminate()
                process.join(1)  # Wait 1 second for termination

                if process.is_alive():
                    # Force kill if necessary
                    if hasattr(signal, 'SIGKILL'):  # Unix only
                        try:
                            os.kill(process.pid, signal.SIGKILL)
                        except (OSError, AttributeError):
                            pass
                    else:
                        # Windows - use taskkill if available
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', str(process.pid)],
                                           check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        except (FileNotFoundError, subprocess.SubprocessError):
                            pass
                    process.join()

            parent_conn.close()
            child_conn.close()

    def _run_sandboxed(
            self,
            func: Callable,
            args: tuple,
            kwargs: dict,
            conn: multiprocessing.connection.Connection
    ) -> None:
        """
        Run the function in a sandboxed environment and send the result back.

        Args:
            func: Function to run
            args: Arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            conn: Connection to send result back to parent process
        """
        try:
            # Set resource limits on Unix-like systems
            if HAS_RESOURCE:
                for limit_type, limit_value in self.resource_limits.items():
                    resource.setrlimit(limit_type, (limit_value, limit_value))

            # Enter sandbox
            self.enter()

            try:
                # Run the function
                result = func(*args, **kwargs)
                conn.send((True, result))
            except Exception as e:
                conn.send((False, str(e)))
            finally:
                # Exit sandbox
                self.exit()

        except Exception as e:
            logger.error(f"Error in sandbox setup: {e}")
            try:
                conn.send((False, f"Sandbox setup error: {str(e)}"))
            except Exception:
                # If we can't send the error, there's not much we can do
                pass
        finally:
            conn.close()


class ContainerSandbox(Sandbox):
    """
    A sandbox that runs operations in a container.

    This provides the strongest isolation by running untrusted operations in a
    separate container with strict resource limits and permissions.

    Note: Requires Docker to be installed and accessible.
    """

    DEFAULT_IMAGE = "python:3.9-slim"

    def __init__(
            self,
            permissions: PermissionSet,
            image: Optional[str] = None,
            memory_limit: str = "1g",
            cpu_limit: str = "1.0",
            network: str = "none"
    ):
        """
        Initialize the container sandbox.

        Args:
            permissions: Permissions granted to the sandbox
            image: Docker image to use
            memory_limit: Memory limit for the container
            cpu_limit: CPU limit for the container
            network: Network mode for the container
        """
        super().__init__(permissions)
        self.image = image or self.DEFAULT_IMAGE
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network = network

        # Only allow network access if permission granted
        if self.permissions.has_permission(Permission.NETWORK_ACCESS):
            self.network = "bridge"
        else:
            self.network = "none"

    def run_script(
            self,
            script_path: str,
            input_files: List[str],
            output_dir: str,
            timeout: int = 60
    ) -> Tuple[int, str, str]:
        """
        Run a Python script in a container with the specified input files.

        Args:
            script_path: Path to the Python script to run
            input_files: List of input files to mount in the container
            output_dir: Directory to mount for outputs
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)

        Raises:
            SandboxError: If the container fails to run or another error occurs
        """
        try:
            # Check if Docker is installed
            try:
                subprocess.run(
                    ["docker", "--version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                raise SandboxError("Docker is required for container sandbox but not found")

            # Enter sandbox
            self.enter()

            # Prepare directories
            script_dir = Path(script_path).parent
            script_name = Path(script_path).name

            # Create command
            cmd = ["docker", "run", "--rm"]

            # Add resource limits
            cmd.extend(["--memory", self.memory_limit])
            cmd.extend(["--cpus", self.cpu_limit])
            cmd.extend(["--network", self.network])

            # Mount input/output volumes
            cmd.extend(["-v", f"{script_dir}:/scripts:ro"])
            cmd.extend(["-v", f"{output_dir}:/output:rw"])

            # Mount each input file to a read-only location
            for i, input_file in enumerate(input_files):
                input_path = Path(input_file)
                cmd.extend(["-v", f"{input_path.parent}:/input{i}:ro"])

                # Pass file location as an environment variable
                cmd.extend(["-e", f"INPUT{i}=/input{i}/{input_path.name}"])

            # Set output directory environment variable
            cmd.extend(["-e", "OUTPUT=/output"])

            # Set timeout
            cmd.extend(["--stop-timeout", str(timeout)])

            # Select image
            cmd.append(self.image)

            # Run script
            cmd.extend(["python", f"/scripts/{script_name}"])

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10  # Add buffer for container setup/teardown
            )

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            raise SandboxError(f"Container execution timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error in container sandbox: {e}")
            raise SandboxError(f"Container sandbox error: {str(e)}")
        finally:
            # Exit sandbox
            self.exit()


class SandboxFactory:
    """
    Factory class for creating appropriate sandbox instances.
    """

    @staticmethod
    def create_sandbox(
            sandbox_type: str,
            permissions: Optional[PermissionSet] = None,
            **kwargs
    ) -> Sandbox:
        """
        Create a sandbox of the specified type.

        Args:
            sandbox_type: Type of sandbox to create ("process", "container", or "inprocess")
            permissions: Permissions for the sandbox
            **kwargs: Additional arguments for the specific sandbox type

        Returns:
            A Sandbox instance

        Raises:
            ValueError: If the sandbox type is invalid
        """
        # Create default permissions if none provided
        if permissions is None:
            permissions = PermissionSet()

        # Create the requested sandbox type
        if sandbox_type == "process":
            return ProcessSandbox(permissions, resource_limits=kwargs.get("resource_limits"))
        elif sandbox_type == "container":
            return ContainerSandbox(
                permissions,
                image=kwargs.get("image"),
                memory_limit=kwargs.get("memory_limit", "1g"),
                cpu_limit=kwargs.get("cpu_limit", "1.0")
            )
        elif sandbox_type == "inprocess":
            return InProcessSandbox(permissions)
        else:
            raise ValueError(f"Invalid sandbox type: {sandbox_type}")

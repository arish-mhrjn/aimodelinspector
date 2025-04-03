"""
Sandbox example for the model_inspector library.

This script demonstrates how to use the sandbox system to safely analyze
potentially unsafe model files.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_inspector import ModelInspector, SafetyLevel, InspectorConfig
from model_inspector.models.permissions import Permission, PermissionSet
from model_inspector.sandbox import SandboxFactory


def main(directory=None):
    """Run the sandbox example."""
    # Use current directory if none provided
    if directory is None:
        directory = os.getcwd()

    print(f"Safely analyzing models in: {directory}")

    # Create a custom permission set
    permissions = PermissionSet()

    # Configure permissions
    permissions.set_permission(Permission.READ_FILE, True)
    permissions.set_permission(Permission.READ_METADATA, True)
    permissions.set_permission(Permission.READ_WEIGHTS, True)
    permissions.set_permission(Permission.EXECUTE_CODE, False)  # No code execution
    permissions.set_permission(Permission.NETWORK_ACCESS, False)  # No network access

    # Allow PyTorch files to execute code (for demonstration)
    permissions.set_format_permission('.pt', Permission.EXECUTE_CODE, True)

    # Create configuratino with sandbox enabled
    config = InspectorConfig(
        safety_level=SafetyLevel.WARN,
        enable_sandbox=True,
        sandbox_type="process",  # Use process sandbox for better isolation
        permissions=permissions
    )

    # Create the inspector
    inspector = ModelInspector(directory, config=config)

    # First, scan for all model files
    print("\nScanning for model files...")
    model_files = list(inspector.directory_files(show_progress=True))
    print(f"Found {len(model_files)} model files")

    # Analyze each file individually to demonstrate sandbox behavior
    for file_path in model_files:
        print(f"\nAnalyzing: {os.path.basename(file_path)}")

        try:
            model_info = inspector.get_model_type(file_path)
            print(f"  Success! Identified as: {model_info.model_type}")
            print(f"  Confidence: {model_info.confidence.name}")
            print(f"  Format: {model_info.format}")

            # Print whether sandbox was used
            format_has_permission = permissions.format_has_permission(
                model_info.format,
                Permission.EXECUTE_CODE
            )
            if not model_info.is_safe and not format_has_permission:
                print("  (Analyzed safely using sandbox)")

        except Exception as e:
            print(f"  Error: {e}")

    # Create a custom sandbox for demonstration
    print("\nCreating a custom sandbox for demonstration:")

    # Create a very restrictive permission set
    restricted_perms = PermissionSet.create_safe_permissions()
    sandbox = SandboxFactory.create_sandbox(
        "inprocess",
        permissions=restricted_perms
    )

    print("Using sandbox to run a simple function:")
    with sandbox.active_session():
        print("  Sandbox is active")

        # Try an allowed operation
        is_allowed = sandbox.check_permission(Permission.READ_FILE)
        print(f"  READ_FILE permission: {is_allowed}")

        # Try a disallowed operation (doesn't raise because we're checking directly)
        try:
            is_allowed = sandbox.check_permission(Permission.EXECUTE_CODE)
            print(f"  EXECUTE_CODE permission: {is_allowed}")
        except Exception as e:
            print(f"  EXECUTE_CODE permission: Error - {e}")


if __name__ == "__main__":
    # Allow specifying a directory as a command-line argument
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    main(directory)

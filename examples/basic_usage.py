"""
Basic usage example for the model_inspector library.

This script demonstrates how to use the ModelInspector to analyze model files
in a directory and get information about them.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_inspector import ModelInspector, SafetyLevel, InspectorConfig


def format_value(value, max_length=200):
    """Format a value for display, truncating if too long."""
    if isinstance(value, (list, tuple)) and len(value) > 10:
        return f"[{len(value)} items - list truncated]"
    elif isinstance(value, dict) and len(value) > 5:
        return f"{{{len(value)} key-value pairs - dict truncated}}"

    str_value = str(value)
    if len(str_value) > max_length:
        return f"{str_value[:max_length]}... [truncated, total length: {len(str_value)}]"
    return str_value


def main(directory=None):
    """Run the basic usage example."""
    show_abreviated_metadata = False
    # Use current directory if none provided
    if directory is None:
        directory = os.getcwd()

    print(f"Analyzing models in: {directory}")

    # Create a simple configuration
    config = InspectorConfig(
        recursive=True,
        safety_level=SafetyLevel.WARN,  # Allow loading most formats with warnings
        show_progress=True
    )

    # Create the inspector
    inspector = ModelInspector(directory, config=config)

    # List all model files
    print("\nModel files found:")
    for file_path in inspector.directory_files(show_progress=True):
        print(f"  - {file_path}")

    # Analyze all models
    print("\nAnalyzing models:")
    results = inspector.analyze_directory(show_progress=True)

    # Display results
    print(f"\nFound {len(results)} models:")
    for model_info in results:
        print(f"\n--- {model_info.filename} ---")
        print(f"Type: {model_info.model_type}")
        print(f"Confidence: {model_info.confidence.name}")
        print(f"Format: {model_info.format}")
        print(f"Size: {model_info.file_size:,} bytes")

        # Print some metadata highlights
        if model_info.metadata:
            print("Metadata Highlights:")
            # Display a few key metadata items
            for key in list(model_info.metadata.keys())[:5]:  # First 5 keys
                value = model_info.metadata[key]
                formatted_value = format_value(value)
                print(f"  {key}: {formatted_value}")

            if show_abreviated_metadata and len(model_info.metadata) > 5:
                print(f"  ... ({len(model_info.metadata) - 5} more items)")
            else:
                for key in list(model_info.metadata.keys())[5:]:
                    value = model_info.metadata[key]
                    formatted_value = format_value(value)
                    print(f"  {key}: {formatted_value}")


if __name__ == "__main__":
    # Allow specifying a directory as a command-line argument
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    main(directory)

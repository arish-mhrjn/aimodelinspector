"""
Filtering example for the model_inspector library.

This script demonstrates how to use FilterCondition, Filter, and ModelFilter
to filter model files based on various criteria.
"""
import os
import sys
from pathlib import Path
import json

# Add the parent directory to the path so we can import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_inspector import ModelInspector, SafetyLevel, InspectorConfig
from model_inspector.utils.filtering import ModelFilter, FilterOperator


def main(directory=None):
    """Run the filtering example."""
    # Use current directory if none provided
    if directory is None:
        directory = os.getcwd()

    print(f"Analyzing and filtering models in: {directory}")

    # Create a configuration
    config = InspectorConfig(
        recursive=True,
        safety_level=SafetyLevel.WARN
    )

    # Create the inspector
    inspector = ModelInspector(directory, config=config)

    # First get all models and count them
    all_results = inspector.analyze_directory(show_progress=True)
    print(f"\nFound {len(all_results)} total models")

    # Filter by size (models larger than 1 MB)
    min_size_filter = ModelFilter().min_size(1024 * 1024)
    inspector_view = inspector.apply_model_filter(min_size_filter)
    large_models = inspector_view.analyze_directory()

    print(f"\nLarge models (>1MB): {len(large_models)}")
    for model in large_models:
        print(f"  - {model.filename}: {model.file_size / (1024 * 1024):.2f} MB")

    # Filter by format
    formats_filter = ModelFilter().formats(['.safetensors', '.onnx'])
    inspector_view = inspector.apply_model_filter(formats_filter)
    filtered_formats = inspector_view.analyze_directory()

    print(f"\nSafetensors and ONNX models: {len(filtered_formats)}")
    for model in filtered_formats:
        print(f"  - {model.filename}: {model.format}")

    # Filter by model type with partial match
    model_type_filter = ModelFilter().model_type("Neural", partial_match=True)
    inspector_view = inspector.apply_model_filter(model_type_filter)
    neural_models = inspector_view.analyze_directory()

    print(f"\nNeural network models: {len(neural_models)}")
    for model in neural_models:
        print(f"  - {model.filename}: {model.model_type}")

    # Combine filters with chaining (small ONNX models)
    combined_filter = (
        ModelFilter()
        .format('onnx')
        .max_size(1024 * 1024)  # < 1MB
    )
    inspector_view = inspector.apply_model_filter(combined_filter)
    small_onnx_models = inspector_view.analyze_directory()

    print(f"\nSmall ONNX models (<1MB): {len(small_onnx_models)}")
    for model in small_onnx_models:
        print(f"  - {model.filename}: {model.file_size / 1024:.2f} KB")

    # Filter by path
    path_filter = ModelFilter().path_contains("models")
    inspector_view = inspector.apply_model_filter(path_filter)
    models_in_path = inspector_view.analyze_directory()

    print(f"\nModels in paths containing 'models': {len(models_in_path)}")

    # Group filtered results
    if large_models:
        grouped = inspector.group_by_model_type(large_models)
        print("\nLarge models by type:")
        for model_type, models in grouped.items():
            total_size = sum(m.file_size for m in models) / (1024 * 1024)
            print(f"  {model_type}: {len(models)} models, {total_size:.2f} MB total")


if __name__ == "__main__":
    # Allow specifying a directory as a command-line argument
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    main(directory)

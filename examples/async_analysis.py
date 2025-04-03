"""
Asynchronous analysis example for the model_inspector library.

This script demonstrates how to use the ModelInspector asynchronously to
analyze model files in a directory, which can be more efficient for
large numbers of files.
"""
import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_inspector import ModelInspector, SafetyLevel, InspectorConfig
from model_inspector.utils.progress import ProgressCallback


async def main(directory=None):
    """Run the async analysis example."""
    # Use current directory if none provided
    if directory is None:
        directory = os.getcwd()

    print(f"Asynchronously analyzing models in: {directory}")

    # Create a configuration with multiple workers
    config = InspectorConfig(
        recursive=True,
        safety_level=SafetyLevel.WARN,
        max_workers=4  # Use multiple threads for faster processing
    )

    # Create the inspector
    inspector = ModelInspector(directory, config=config)

    # Create a progress callback
    progress_info = {'total': 0, 'current': 0, 'start_time': None}

    def on_start():
        import time
        progress_info['start_time'] = time.time()
        print("Starting analysis...")

    def on_progress(current, total, info):
        progress_info['current'] = current
        progress_info['total'] = total

        percent = (current / total * 100) if total else 0
        print(f"Progress: {current}/{total} ({percent:.1f}%) - Rate: {info.get('rate_formatted', 'N/A')}")

        if 'eta_formatted' in info:
            print(f"Estimated time remaining: {info['eta_formatted']}")

    def on_complete(results):
        import time
        elapsed = time.time() - progress_info['start_time']
        print(f"\nAnalysis complete! Processed {len(results)} models in {elapsed:.2f} seconds")

    def on_error(exception):
        print(f"Error during analysis: {exception}")

    # Create the callback object
    callback = ProgressCallback(
        on_start=on_start,
        on_progress=on_progress,
        on_complete=on_complete,
        on_error=on_error,
        throttle_ms=500  # Only update every 500ms to avoid too much output
    )

    # Run the analysis with the callback
    results = await inspector.analyze_directory_async(progress_callback=callback)

    # Group results by model type
    grouped = inspector.group_by_model_type(results)

    # Display summary
    print("\nSummary by Model Type:")
    for model_type, models in grouped.items():
        print(f"  {model_type}: {len(models)} models")

    # Display detailed results for the first few models
    display_count = min(3, len(results))
    if display_count > 0:
        print(f"\nDetails for {display_count} example models:")

        for i, model_info in enumerate(results[:display_count]):
            print(f"\n--- Example {i + 1}: {model_info.filename} ---")
            print(f"Type: {model_info.model_type}")
            print(f"Confidence: {model_info.confidence.name}")
            print(f"Format: {model_info.format}")
            print(f"Size: {model_info.file_size:,} bytes")


if __name__ == "__main__":
    # Allow specifying a directory as a command-line argument
    directory = sys.argv[1] if len(sys.argv) > 1 else None

    # Run the async function
    asyncio.run(main(directory))

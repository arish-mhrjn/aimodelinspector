# Model Inspector

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

<!--
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
-->

> ### Notes: 
> #### - A comprehensive tool for identifying, analyzing, and extracting metadata from various AI model files by Stephen Genusa
> #### - This is a first release. Although highly capable, consider this an alpha version. This is a WIP I began for my own self-education
> #### - The requirements.txt file is not complete yet

## Features

- **Multi-Format Support**: Analyze models in many formats including SafeTensors, GGUF, PyTorch, ONNX, TensorFlow, HDF5, CoreML, XGBoost, Scikit-learn, and many more.
- **Model Type Identification**: Automatically identify model architecture and type with confidence levels.
- **Metadata Extraction**: Extract detailed metadata like model parameters, structure, training information, and format-specific details.
- **Advanced Filtering**: Filter models by size, type, format, and metadata contents.
- **Safe Analysis**: Security sandbox for safely analyzing potentially unsafe formats. Unsafe formats are not loaded but parsed for the metadata. The sandbox should not be necessary but this library is an Alpha version and you need to exercise care when dealing with unsafe formats. An [analysis of the Sandbox mode](sandbox_analysis.md) is available in the repo.
- **Asynchronous Processing**: Efficiently process large collections of models.
- **Caching**: Memory and disk caching for improved performance with large model collections.
- **Progress Reporting**: Customizable progress reporting for long-running operations.
- **Comprehensive API**: Clean, well-documented API with both synchronous and asynchronous interfaces.
- **Custom Analyzer Extensions**: Easily extend with custom analyzers for proprietary formats.

<!--

## Installation

```bash
pip install model-inspector
```

### Optional Dependencies

```bash
# For full format support
pip install model-inspector[all]

# For specific formats
pip install model-inspector[pytorch]  # PyTorch support
pip install model-inspector[tensorflow]  # TensorFlow support
pip install model-inspector[onnx]  # ONNX support
pip install model-inspector[xgboost]  # XGBoost support
```
--> 

## Quick Start

```python
from model_inspector import ModelInspector

# Create an inspector for a directory containing model files
inspector = ModelInspector("/path/to/models")

# Analyze all models in the directory
results = inspector.analyze_directory(show_progress=True)

# Print information about each model
for model_info in results:
    print(f"File: {model_info.filename}")
    print(f"Model type: {model_info.model_type}")
    print(f"Confidence: {model_info.confidence.name}")
    print(f"Format: {model_info.format}")
    print(f"Size: {model_info.file_size:,} bytes")
    print(f"Metadata: {model_info.metadata}")
    print()
```

## Usage Examples

### Basic Model Analysis

```python
from model_inspector import ModelInspector, SafetyLevel, InspectorConfig

# Configure the inspector
config = InspectorConfig(
    recursive=True,  # Search subdirectories
    safety_level=SafetyLevel.WARN,  # Allow potentially unsafe formats with warnings
    enable_caching=True  # Cache results for faster repeated analysis
)

# Create the inspector
inspector = ModelInspector("/path/to/models", config=config)

# List all model files
for file_path in inspector.directory_files():
    print(f"Found model file: {file_path}")

# Analyze a specific model file
model_info = inspector.get_model_type("/path/to/models/model.safetensors")
print(f"Model type: {model_info.model_type}")
print(f"Metadata: {model_info.metadata}")
```

### Asynchronous Processing

```python
import asyncio
from model_inspector import ModelInspector
from model_inspector.utils.progress import ProgressCallback

# Define progress callbacks
callback = ProgressCallback(
    on_start=lambda: print("Starting analysis..."),
    on_progress=lambda current, total, info: print(f"Progress: {current}/{total}"),
    on_complete=lambda results: print(f"Completed! Found {len(results)} models"),
    on_error=lambda e: print(f"Error: {e}")
)

async def analyze_models():
    inspector = ModelInspector("/path/to/models")
    results = await inspector.analyze_directory_async(progress_callback=callback)
    
    # Group by model type
    grouped = inspector.group_by_model_type(results)
    
    for model_type, models in grouped.items():
        print(f"{model_type}: {len(models)} models")

# Run the async function
asyncio.run(analyze_models())
```

### Filtering Models

```python
from model_inspector import ModelInspector
from model_inspector.utils.filtering import ModelFilter

# Create the inspector
inspector = ModelInspector("/path/to/models")

# Create a filter for large LoRA models
model_filter = (
    ModelFilter()
    .min_size(10 * 1024 * 1024)  # At least 10MB
    .model_type("LoRA", partial_match=True)  # Type contains "LoRA"
    .formats([".safetensors"])  # Only safetensors format
)

# Apply the filter
filtered_view = inspector.apply_model_filter(model_filter)

# Analyze only the filtered models
results = filtered_view.analyze_directory(show_progress=True)

print(f"Found {len(results)} large LoRA models")

# Create another filter by model metadata
sd_filter = ModelFilter().metadata_contains("architecture", "StableDiffusion")
sd_models = inspector.apply_model_filter(sd_filter).analyze_directory()
print(f"Found {len(sd_models)} Stable Diffusion models")
```

### Safe Analysis with Sandbox

```python
from model_inspector import ModelInspector, InspectorConfig, SafetyLevel
from model_inspector.models.permissions import Permission, PermissionSet

# Create permissions that allow reading files but not executing code
permissions = PermissionSet.create_safe_permissions()

# Configure with sandbox for safer execution
config = InspectorConfig(
    safety_level=SafetyLevel.WARN,
    enable_sandbox=True, 
    sandbox_type="process",  # Use process isolation
    permissions=permissions
)

inspector = ModelInspector("/path/to/models", config=config)

# This will safely analyze potentially unsafe formats
results = inspector.analyze_directory()
```

### Context Manager Usage

```python
from model_inspector import ModelInspector

# Use as a context manager to ensure proper resource cleanup
with ModelInspector("/path/to/models") as inspector:
    # Analyze all models in the directory
    results = inspector.analyze_directory()
    
    # Group by format
    by_format = inspector.group_by_format(results)
    
    for format_name, models in by_format.items():
        print(f"{format_name}: {len(models)} models")
```

### Working with Large Model Collections

```python
from model_inspector import ModelInspector, InspectorConfig

# Configure for efficient processing of large collections
config = InspectorConfig(
    recursive=True,
    max_workers=8,  # Adjust based on CPU cores
    enable_caching=True,
    persistent_cache=True,
    cache_directory="~/.model_cache"
)

inspector = ModelInspector("/path/to/large/collection", config=config)

# Use chunked processing for very large directories
chunk_size = 100
all_files = list(inspector.directory_files())

for i in range(0, len(all_files), chunk_size):
    chunk = all_files[i:i+chunk_size]
    results = inspector.analyze_files(chunk, show_progress=True)
    # Process this chunk of results
    print(f"Processed chunk {i//chunk_size + 1}: {len(results)} models")
```

### Custom Analyzers

```python
from model_inspector import ModelInspector
from model_inspector.analyzers.base import BaseAnalyzer
from model_inspector.models.confidence import ModelConfidence
from typing import Dict, Any, Tuple

# Create a custom analyzer for proprietary format
class MyCustomAnalyzer(BaseAnalyzer):
    def get_supported_extensions(self) -> set:
        return {'.myformat'}
        
    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        # Custom analysis logic here
        return "My-Custom-Model", ModelConfidence.HIGH, {"custom_field": "value"}

# Register the custom analyzer
from model_inspector.analyzers.analyzer_registry import register_analyzer
register_analyzer(['.myformat'], MyCustomAnalyzer)

# Now the inspector can analyze .myformat files
inspector = ModelInspector("/path/to/models")
results = inspector.analyze_directory()
```

## Supported Formats

| Format                    | Extensions                   | Safety             | Description                                  |
|---------------------------|------------------------------|--------------------|----------------------------------------------|
| SafeTensors               | `.safetensors`               | Safe               | Format designed for secure tensor storage    |
| GGUF/GGML                 | `.gguf`, `.ggml`             | Safe               | Large language model format (llama.cpp)      |
| PyTorch                   | `.pt`, `.pth`                | Potentially unsafe | PyTorch model and checkpoint formats         |
| PyTorch JIT               | `.pt`, `.pth`                | Potentially unsafe | PyTorch JIT compiled models                  |
| ONNX                      | `.onnx`                      | Safe               | Open Neural Network Exchange format          |
| ONNX Runtime              | `.ort`                       | Safe               | ONNX Runtime optimized models                |
| Diffusers                 | multiple                     | Potentially unsafe | Diffuser models                              |
| TensorFlow                | `.pb`                        | Safe               | TensorFlow frozen graph format               |
| TFLite                    | `.tflite`                    | Safe               | TensorFlow Lite format for mobile/edge       |
| HDF5                      | `.h5`, `.hdf5`               | Safe               | Hierarchical Data Format (TensorFlow, Keras) |
| Checkpoint                | `.ckpt`                      | Potentially unsafe | Generic checkpoint format                    |
| CoreML                    | `.mlmodel`                   | Safe               | Apple CoreML format                          |
| CoreML Package            | `.mlpackage`                 | Safe               | Apple CoreML package format                  |
| XGBoost                   | `.json`, `.ubj`              | Safe               | XGBoost model formats                        |
| Scikit-learn              | `.pkl`, `.joblib`, `.pickle` | Potentially unsafe | Scikit-learn serialized models               |
| JAX                       | `.msgpack`                   | Safe               | JAX model format                             |
| MXNet                     | `.params`, `.json`           | Safe               | MXNet model formats                          |
| Caffe                     | `.caffemodel`, `.prototxt`   | Safe               | Caffe model format                           |
| Caffe2                    | `.pb`                        | Safe               | Caffe2 model format                          |
| PaddlePaddle              | `.pdmodel`, `.pdiparams`     | Safe               | PaddlePaddle model formats                   |
| TVM                       | `.so`, `.tar`                | Potentially unsafe | Apache TVM compiled models                   |
| OpenVINO IR               | `.xml`, `.bin`               | Safe               | OpenVINO Intermediate Representation         |
| RAPIDS cuML               | `.cuml`                      | Safe               | RAPIDS cuML model format                     |
| Neural Codec              | `.enn`                       | Safe               | EPUB Neural Codec model format               |
| Generic Binary            | `.bin`                       | Needs inspection   | Generic binary format (requires detection)   |
| Metal Performance Shaders | `.mps`                       | Safe               | Apple MPS models                             |

## Configuration Options

The `InspectorConfig` class provides numerous configuration options:

```python
config = InspectorConfig(
    # Basic configuration
    recursive=True,          # Search subdirectories
    max_workers=4,           # Number of worker threads for async operations
    safety_level=SafetyLevel.SAFE,  # SAFE, WARN, or UNSAFE
    
    # Format handling
    enabled_formats={".safetensors", ".onnx"},  # Only analyze these formats (None for all)
    disabled_formats={".bin"},       # Don't analyze these formats
    
    # File filtering
    min_size=1024,                   # Minimum file size in bytes
    max_size=1024*1024*1024,         # Maximum file size in bytes
    exclude_patterns=["**/cache/*"],  # Glob patterns to exclude
    include_patterns=["**/*.safetensors"],  # Glob patterns to include
    
    # Caching
    enable_caching=True,              # Enable results caching
    cache_size=1000,                  # Max items in memory cache
    persistent_cache=True,            # Save cache to disk
    cache_directory="~/.model_inspector_cache",  # Cache location
    
    # Progress reporting
    progress_format="bar",            # 'bar', 'plain', 'pct', or 'spinner'
    show_progress=True,               # Show progress by default
    
    # Sandbox configuration
    enable_sandbox=True,              # Enable sandbox for unsafe formats
    sandbox_type="process",           # 'inprocess', 'process', or 'container'
    
    # Format-specific analyzer configurations
    analyzer_configs={
        "ONNXAnalyzer": {"custom_option": True},
        "GGUFAnalyzer": {"load_tensors": False},
        "TensorFlowAnalyzer": {"cache_graphs": True},
    }
)
```

## Output Format

The `ModelInfo` object returned by analysis operations contains:

```python
class ModelInfo:
    """Comprehensive information about an analyzed model."""
    
    # File information
    filepath: str             # Full path to the model file
    filename: str             # Just the filename portion
    file_size: int            # File size in bytes
    format: str               # Format identifier (e.g., 'onnx', 'safetensors')
    extension: str            # File extension (with dot)
    
    # Model identification
    model_type: str           # Identified model type/architecture
    confidence: ModelConfidence  # Confidence in the identification
    
    # Detailed metadata
    metadata: Dict[str, Any]  # Format-specific metadata dictionary
    
    # Analysis details
    analysis_time: float      # Time taken to analyze (seconds)
    analysis_date: datetime   # When the analysis was performed
    analyzer_name: str        # Name of the analyzer used
    
    # Error information (if analysis failed)
    error: Optional[str]      # Error message if analysis failed
    
    # Helper properties and methods...
```

## Security and Safety

Model Inspector provides three safety levels:

1. **SAFE**: Only formats known to be completely safe are loaded
2. **WARN**: Potentially unsafe formats are loaded with warnings
3. **UNSAFE**: All formats are loaded regardless of safety concerns

For additional security, the sandbox system provides isolation when analyzing potentially unsafe formats:

- **InProcessSandbox**: Basic restrictions within the current process
- **ProcessSandbox**: Runs analysis in a separate process with resource limits
- **ContainerSandbox**: Uses Docker containers for maximum isolation (requires Docker)

Customizable permissions allow fine-grained control over what operations are permitted during analysis:

```python
from model_inspector.models.permissions import Permission, PermissionSet

# Create a custom permission set
permissions = PermissionSet()
permissions.set_permission(Permission.READ_FILE, True)
permissions.set_permission(Permission.EXECUTE_CODE, False)
permissions.set_permission(Permission.NETWORK_ACCESS, False)
permissions.set_permission(Permission.WRITE_FILE, False)

# Set format-specific permissions
permissions.set_format_permission(".pt", Permission.EXECUTE_CODE, True)
```

## Common Metadata Fields

While metadata varies by format, common fields include:

# Common across many formats
- format: Format identifier
- format_version: Format version number
- file_size_bytes: Size of the file
- architecture: Model architecture identifier
- parameters_count: Number of parameters
- creation_date: When the model was created
- framework: Framework that created the model
- framework_version: Version of the framework
- description: Model description
- license: Model license information

# Format-specific examples

# GGUF models
- gguf_version: GGUF format version
- tensor_count: Number of tensors
- vocab_size: Size of the vocabulary
- embedding_length: Embedding dimensions
- block_count: Number of transformer blocks
- head_count: Number of attention heads

# SafeTensors models
- tensor_count: Number of tensors
- metadata: User-specified metadata dictionary
- total_tensor_bytes: Size of all tensors
- largest_tensor_shape: Shape of the largest tensor

# ONNX models
- opset_version: ONNX operator set version
- producer_name: Name of the producer
- input_names: List of input tensor names
- output_names: List of output tensor names
- graph_structure: Structure of the computation graph

# XGBoost models
- objective: Objective function
- booster_type: Type of the booster
- num_features: Number of features
- num_trees: Number of trees

# Neural Codec models
- architecture: Audio codec architecture
- sample_rate: Audio sample rate
- audio_channels: Number of audio channels
- bitrate: Coding bitrate

## Architecture

### Core Components

- **ModelInspector**: Main interface for interacting with the library
- **Analyzers**: Format-specific components that extract model information
   - BaseAnalyzer: Abstract base class for all analyzers
   - Format-specific analyzers (GGUFAnalyzer, ONNXAnalyzer, etc.)
   - Dispatcher analyzers for handling ambiguous formats
- **Filters**: Components for selecting subsets of models
- **Sandbox**: Security system for safely handling potentially unsafe formats
- **Caching**: Performance optimization for repeated analysis

### Data Models

- **ModelInfo**: Contains all details about an identified model
- **SafetyLevel**: Enum defining safety policies (SAFE, WARN, UNSAFE)
- **ModelConfidence**: Enum indicating confidence in type identification
   - HIGH: Certain about model identification
   - MEDIUM: Reasonably confident but some uncertainty
   - LOW: Tentative identification
   - UNKNOWN: Unable to identify model type
- **PermissionSet**: Configuration of security permissions

## Performance Tips

1. **Use Asynchronous API**: For large collections, the async API can be much faster
2. **Enable Caching**: Caching results greatly speeds up repeated analysis
3. **Adjust Worker Count**: Set `max_workers` based on your CPU cores
4. **Filter Early**: Use directory_files() filtering to reduce the number of files analyzed
5. **Use Persistent Cache**: Enable persistent caching for reuse across program runs
6. **Selective Format Analysis**: Use enabled_formats to only analyze necessary formats
7. **Chunk Processing**: Process very large collections in chunks to manage memory
8. **Disable Tensor Loading**: For formats like GGUF, disable tensor loading for faster analysis
9. **Use SSD Storage**: When analyzing large collections, SSD storage provides significant speedups
10. **Profile Memory Usage**: Monitor memory when analyzing large models and adjust accordingly

## Extending Model Inspector

### Creating Custom Analyzers

Create a custom analyzer by inheriting from `BaseAnalyzer`:

```python
from model_inspector.analyzers.base import BaseAnalyzer
from model_inspector.models.confidence import ModelConfidence

class CustomFormatAnalyzer(BaseAnalyzer):
    """Analyzer for custom model format."""

    def get_supported_extensions(self) -> set:
        """Return the set of file extensions this analyzer supports."""
        return {'.custom', '.cust'}

    def analyze(self, file_path: str):
        """Analyze the model file."""
        # Your analysis logic here
        
        # Return a tuple of (model_type, confidence, metadata)
        return "Custom-Model", ModelConfidence.HIGH, {"format": "custom"}
        
    def can_analyze_safely(self, file_path: str) -> bool:
        """Determine if this file can be analyzed safely."""
        return True
```

### Custom Processing Hooks

Register callbacks to implement custom processing:

```python
from model_inspector import ModelInspector

def pre_analysis_hook(file_path):
    print(f"About to analyze: {file_path}")
    
def post_analysis_hook(file_path, model_info):
    if model_info.model_type == "Unknown":
        print(f"Failed to identify: {file_path}")

inspector = ModelInspector("/path/to/models")

# Register hooks
inspector.register_pre_analysis_hook(pre_analysis_hook)
inspector.register_post_analysis_hook(post_analysis_hook)

# Run analysis with hooks
results = inspector.analyze_directory()
```

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

### Reporting Issues

- Report bugs and issues on the GitHub issue tracker
- Include detailed information about the problem and environment
- Provide minimal code examples to reproduce the issue

### Development Setup

```bash
# Clone the repository
git clone https://github.com/StephenGenusa/aimodelinspector.git
cd model-inspector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- Format specifications and documentation from various ML framework projects
- Security design principles from secure sandbox research
- Community feedback and contributions
- Open source libraries that make this project possible

-----------------------

Copyright (c) 2025 Stephen Genusa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This project includes components and inspiration from various open-source 
projects. Individual files may contain additional attribution notices.

Stephen Genusa is not affiliated with, endorsed by, or sponsored by any of the 
ML frameworks or model formats it supports. All trademarks are the property of 
their respective owners.

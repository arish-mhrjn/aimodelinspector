from typing import Dict, Any, Tuple, Optional, List, Set
import struct
import logging
import json
import os
import tarfile
import tempfile
from pathlib import Path
import re
from collections import defaultdict, Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer

"""
Analyzer module for Apache TVM compiled model files (.so, .tar).

This module provides functionality to analyze machine learning models compiled
using the Apache TVM framework. TVM is a deep learning compiler stack that optimizes
models for deployment across various hardware targets including CPUs, GPUs, and 
specialized accelerators.

The analyzer extracts metadata from TVM-compiled models, which typically come in
two formats:
1. Dynamic shared libraries (.so files) containing the compiled operators
2. Tar archives (.tar files) containing the compiled library, parameters, and model metadata

The analyzer attempts to extract information such as:
- Target hardware information
- Input and output tensor specifications
- Operator counts and types
- Memory requirements
- Optimization settings
- Original framework information if available

Potential improvements:
1. More robust extraction of original model architecture from compiled code
2. Support for additional TVM serialization formats (e.g., .json + .params)
3. Integration with TVM runtime for test loading and benchmarking
4. Analysis of quantization patterns and precision formats
5. Extraction of operator fusion patterns for performance insights
6. Detection of custom TVM schedules and optimizations
7. Cross-reference with TVMC compilation options
8. Support for Relay model information when available
9. Analysis of memory access patterns and cache optimization
10. Extraction of AutoTVM/AutoScheduler tuning records if embedded
"""


class TVMAnalyzer(BaseAnalyzer):
    """
    Analyzer for Apache TVM compiled model files.

    This analyzer extracts information from models compiled with the TVM framework
    which are typically stored as .so (shared libraries) or .tar (archives containing
    the library, parameters, and metadata) files. It can determine target hardware,
    model structure, and optimization settings applied during compilation.

    TVM compiles high-level model descriptions (from frameworks like PyTorch, TensorFlow, etc.)
    into optimized code for various hardware targets. This analyzer attempts to recover
    information about the original model and the compilation process from the compiled artifacts.
    """

    # Hardware targets commonly used in TVM
    HARDWARE_TARGETS = {
        "llvm": "CPU (LLVM)",
        "cuda": "NVIDIA GPU",
        "opencl": "OpenCL",
        "metal": "Apple Metal GPU",
        "vulkan": "Vulkan",
        "rocm": "AMD ROCm GPU",
        "vpi": "NVIDIA VPI",
        "mali": "ARM Mali GPU",
        "adreno": "Qualcomm Adreno GPU",
        "intel_graphics": "Intel Graphics",
        "nvptx": "NVIDIA PTX",
        "webgpu": "WebGPU",
        "cm": "Intel CM",
        "hexagon": "Qualcomm Hexagon DSP",
        "tflite": "TensorFlow Lite",
        "tensorrt": "NVIDIA TensorRT",
        "arm_cpu": "ARM CPU",
        "x86": "x86 CPU",
    }

    # Common original frameworks detected in TVM models
    SOURCE_FRAMEWORKS = {
        "pytorch": "PyTorch",
        "torch": "PyTorch",
        "tensorflow": "TensorFlow",
        "tf": "TensorFlow",
        "mxnet": "MXNet",
        "onnx": "ONNX",
        "keras": "Keras",
        "tflite": "TensorFlow Lite",
        "darknet": "Darknet",
        "caffe": "Caffe",
        "caffe2": "Caffe2",
        "paddle": "PaddlePaddle",
        "relay": "TVM Relay",
    }

    # Common models compiled with TVM
    COMMON_MODELS = {
        "resnet": "ResNet",
        "mobilenet": "MobileNet",
        "inception": "Inception",
        "vgg": "VGG",
        "densenet": "DenseNet",
        "squeezenet": "SqueezeNet",
        "efficientnet": "EfficientNet",
        "yolo": "YOLO",
        "ssd": "SSD",
        "bert": "BERT",
        "transformer": "Transformer",
        "lstm": "LSTM",
        "gru": "GRU",
        "alexnet": "AlexNet",
        "shufflenet": "ShuffleNet",
    }

    def __init__(self):
        """Initialize the TVM analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.so', '.tar'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a TVM model file to determine its model type and metadata.

        Args:
            file_path: Path to the TVM model file (.so or .tar)

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid TVM model file
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Determine file type and parse accordingly
            extension = path.suffix.lower()

            if extension == '.tar':
                metadata = self._parse_tvm_tar(file_path)
            elif extension == '.so':
                metadata = self._parse_tvm_so(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing TVM model file {file_path}: {e}")
            raise

    def _parse_tvm_tar(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a TVM model in TAR archive format.

        Args:
            file_path: Path to the TVM .tar file

        Returns:
            Dictionary containing extracted metadata

        Raises:
            ValueError: If the file is not a valid TVM tar archive
        """
        metadata = {
            "format": "tvm_tar",
            "file_path": file_path,
            "file_size_bytes": Path(file_path).stat().st_size,
            "size_mb": Path(file_path).stat().st_size / (1024 * 1024),
        }

        # Check if this is a valid tar file
        if not tarfile.is_tarfile(file_path):
            raise ValueError(f"Not a valid tar file: {file_path}")

        # Extract information from tar content
        with tarfile.open(file_path, 'r') as tar:
            # List all files in the archive
            file_list = tar.getnames()
            metadata["archive_contents"] = file_list

            # Look for common files in TVM exports
            has_lib = any(f.endswith('lib.so') or f.endswith('lib.so.so') or f == 'lib.so' for f in file_list)
            has_params = any(f.endswith('params') or f == 'params' for f in file_list)
            has_graph = any(f.endswith('json') or f == 'graph.json' for f in file_list)

            metadata["has_lib"] = has_lib
            metadata["has_params"] = has_params
            metadata["has_graph"] = has_graph

            # If this appears to be a TVM model, extract more metadata
            if has_lib and (has_params or has_graph):
                metadata["is_valid_tvm_model"] = True

                # Extract graph information if available
                if has_graph:
                    graph_file = next((f for f in file_list if f.endswith('json') or f == 'graph.json'), None)
                    if graph_file:
                        try:
                            with tempfile.TemporaryDirectory() as temp_dir:
                                tar.extract(graph_file, path=temp_dir)
                                graph_path = os.path.join(temp_dir, graph_file)

                                with open(graph_path, 'r') as f:
                                    graph_data = json.load(f)
                                    metadata["graph_info"] = self._extract_graph_info(graph_data)
                        except Exception as e:
                            self.logger.warning(f"Error extracting graph info: {e}")

                # Look for metadata file
                if "metadata.json" in file_list:
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            tar.extract("metadata.json", path=temp_dir)

                            with open(os.path.join(temp_dir, "metadata.json"), 'r') as f:
                                meta_data = json.load(f)
                                metadata["tvm_metadata"] = meta_data

                                # Extract target information if available
                                if "target" in meta_data:
                                    metadata["target_info"] = self._parse_target_string(meta_data["target"])
                    except Exception as e:
                        self.logger.warning(f"Error extracting metadata.json: {e}")

                # Extract shared library information
                lib_file = next((f for f in file_list if f.endswith('lib.so') or f == 'lib.so'), None)
                if lib_file:
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            tar.extract(lib_file, path=temp_dir)
                            lib_path = os.path.join(temp_dir, lib_file)

                            # Get SO file metadata
                            so_metadata = self._extract_so_metadata(lib_path)
                            metadata.update(so_metadata)
                    except Exception as e:
                        self.logger.warning(f"Error extracting library info: {e}")
            else:
                metadata["is_valid_tvm_model"] = False

        # Try using TVM library if available
        self._try_tvm_runtime_analysis(file_path, metadata)

        return metadata

    def _parse_tvm_so(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a TVM model in shared library (.so) format.

        Args:
            file_path: Path to the TVM .so file

        Returns:
            Dictionary containing extracted metadata

        Raises:
            ValueError: If the file doesn't appear to be a TVM shared library
        """
        metadata = {
            "format": "tvm_so",
            "file_path": file_path,
            "file_size_bytes": Path(file_path).stat().st_size,
            "size_mb": Path(file_path).stat().st_size / (1024 * 1024),
        }

        # Extract information from the SO file
        so_metadata = self._extract_so_metadata(file_path)
        metadata.update(so_metadata)

        # Try using TVM library if available
        self._try_tvm_runtime_analysis(file_path, metadata)

        return metadata

    def _extract_so_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a shared library file.

        Args:
            file_path: Path to the .so file

        Returns:
            Dictionary of extracted metadata
        """
        result = {}

        try:
            # Check TVM-specific symbols in the shared library
            import subprocess

            # Run nm command to get symbols
            try:
                nm_output = subprocess.check_output(['nm', '-D', file_path], stderr=subprocess.STDOUT).decode('utf-8',
                                                                                                              errors='ignore')

                # Look for TVM-specific symbols
                tvm_runtime_symbols = [
                    'TVMBackendGetFuncFromEnv',
                    'TVMBackendAllocWorkspace',
                    'TVMBackendFreeWorkspace',
                    'TVMAPISetLastError',
                    'TVMGraphExecutorCreate',
                    'TVMFuncCall',
                    'TVMModGetFunction',
                ]

                found_symbols = []
                for symbol in tvm_runtime_symbols:
                    if symbol in nm_output:
                        found_symbols.append(symbol)

                result["tvm_symbols_found"] = found_symbols
                result["is_tvm_library"] = len(found_symbols) > 0

                # Look for potential operator hints
                operators = []
                for line in nm_output.split('\n'):
                    if 'tvm::relay' in line.lower():
                        operators.append(line)
                    elif 'nn::' in line or 'conv2d' in line.lower() or 'dense' in line.lower():
                        operators.append(line)

                if operators:
                    result["potential_operators"] = operators[:30]  # Limit to avoid huge output
            except subprocess.SubprocessError:
                self.logger.warning("Failed to run 'nm' command to extract library symbols")

            # Check library dependencies with ldd
            try:
                ldd_output = subprocess.check_output(['ldd', file_path], stderr=subprocess.STDOUT).decode('utf-8',
                                                                                                          errors='ignore')
                dependencies = [line.strip() for line in ldd_output.split('\n') if line.strip()]

                # Extract interesting dependencies
                interesting_deps = []
                for dep in dependencies:
                    if any(keyword in dep.lower() for keyword in
                           ['cuda', 'gpu', 'opencl', 'vulkan', 'metal', 'tvm', 'llvm', 'mkl']):
                        interesting_deps.append(dep)

                if interesting_deps:
                    result["library_dependencies"] = interesting_deps

                # Try to determine hardware target from dependencies
                for target_key, target_name in self.HARDWARE_TARGETS.items():
                    if any(target_key.lower() in dep.lower() for dep in dependencies):
                        result.setdefault("detected_targets", []).append(target_name)
            except subprocess.SubprocessError:
                self.logger.warning("Failed to run 'ldd' command to extract library dependencies")

            # Extract strings from the binary that might indicate model type
            try:
                strings_output = subprocess.check_output(['strings', file_path], stderr=subprocess.STDOUT).decode(
                    'utf-8', errors='ignore')

                # Look for model type hints
                model_hints = []
                framework_hints = []
                target_hints = []

                # Check for common model architectures
                for key, name in self.COMMON_MODELS.items():
                    if re.search(r'\b' + re.escape(key) + r'\b', strings_output, re.IGNORECASE):
                        model_hints.append(name)

                # Check for source frameworks
                for key, name in self.SOURCE_FRAMEWORKS.items():
                    if re.search(r'\b' + re.escape(key) + r'\b', strings_output, re.IGNORECASE):
                        framework_hints.append(name)

                # Check for hardware targets
                for key, name in self.HARDWARE_TARGETS.items():
                    if re.search(r'\b' + re.escape(key) + r'\b', strings_output, re.IGNORECASE):
                        target_hints.append(name)

                if model_hints:
                    result["model_hints"] = list(set(model_hints))
                if framework_hints:
                    result["framework_hints"] = list(set(framework_hints))
                if target_hints:
                    result["target_hints"] = list(set(target_hints))

                # Look for TVM compiled module signatures
                if "tvm_module_ctx" in strings_output or "runtime.SystemLib" in strings_output:
                    result["is_tvm_library"] = True
            except subprocess.SubprocessError:
                self.logger.warning("Failed to run 'strings' command to extract string information")

            # Extract file info
            file_stats = os.stat(file_path)
            result["file_created"] = file_stats.st_ctime
            result["file_modified"] = file_stats.st_mtime

        except Exception as e:
            self.logger.warning(f"Error extracting shared library metadata: {e}")

        return result

    def _extract_graph_info(self, graph_data: Dict) -> Dict[str, Any]:
        """
        Extract useful information from TVM graph JSON.

        Args:
            graph_data: TVM graph JSON data

        Returns:
            Dictionary of extracted graph information
        """
        graph_info = {}

        try:
            # Extract nodes information
            if "nodes" in graph_data:
                nodes = graph_data["nodes"]
                graph_info["node_count"] = len(nodes)

                # Count operator types
                op_counts = Counter()
                for node in nodes:
                    if "op" in node:
                        op_counts[node["op"]] += 1

                graph_info["operator_counts"] = dict(op_counts)

                # Find input and output nodes
                input_nodes = []
                output_nodes = []

                for i, node in enumerate(nodes):
                    # Input nodes typically have op = "null" and no inputs
                    if node.get("op") == "null" and "inputs" in node and len(node["inputs"]) == 0:
                        if "name" in node:
                            input_nodes.append({"index": i, "name": node["name"]})
                        else:
                            input_nodes.append({"index": i})

                # Find arg_nodes (typically inputs) if specified
                if "arg_nodes" in graph_data:
                    arg_indices = graph_data["arg_nodes"]
                    for idx in arg_indices:
                        if idx < len(nodes):
                            node = nodes[idx]
                            if "name" in node:
                                if not any(input_node["index"] == idx for input_node in input_nodes):
                                    input_nodes.append({"index": idx, "name": node["name"]})

                # Find heads/outputs if specified
                if "heads" in graph_data:
                    heads = graph_data["heads"]
                    for head in heads:
                        if isinstance(head, list) and len(head) >= 1:
                            node_idx = head[0]
                            if node_idx < len(nodes):
                                node = nodes[node_idx]
                                if "name" in node:
                                    output_nodes.append({"index": node_idx, "name": node["name"]})
                                else:
                                    output_nodes.append({"index": node_idx})

                graph_info["input_nodes"] = input_nodes
                graph_info["output_nodes"] = output_nodes

                # Try to extract input shapes
                if "attrs" in graph_data and "shape" in graph_data["attrs"]:
                    shapes = graph_data["attrs"]["shape"]
                    if "data" in shapes:
                        shape_data = shapes["data"]
                        if isinstance(shape_data, list) and len(shape_data) > 0:
                            # Try to match shapes with input nodes
                            input_shapes = []
                            for i, node in enumerate(input_nodes):
                                if i < len(shape_data):
                                    input_shapes.append(shape_data[i])

                            if input_shapes:
                                graph_info["input_shapes"] = input_shapes

                # Extract dtype information if available
                if "attrs" in graph_data and "dltype" in graph_data["attrs"]:
                    dtypes = graph_data["attrs"]["dltype"]
                    if "data" in dtypes:
                        dtype_data = dtypes["data"]
                        if isinstance(dtype_data, list) and len(dtype_data) > 0:
                            graph_info["dtypes"] = dtype_data

            # Extract additional attributes
            if "attrs" in graph_data:
                attrs = graph_data["attrs"]
                filtered_attrs = {}

                # Filter out large or less useful attributes
                for key, value in attrs.items():
                    if key not in ["shape", "dltype"]:  # Already extracted
                        filtered_attrs[key] = value

                if filtered_attrs:
                    graph_info["additional_attrs"] = filtered_attrs

        except Exception as e:
            self.logger.warning(f"Error extracting graph info: {e}")

        return graph_info

    def _parse_target_string(self, target_str: str) -> Dict[str, Any]:
        """
        Parse a TVM target string to extract hardware and configuration information.

        Args:
            target_str: TVM target specification string

        Returns:
            Dictionary of target information
        """
        target_info = {}

        try:
            # Basic target parsing
            main_target = target_str.split()[0]
            target_info["main_target"] = main_target

            # Check for known targets
            for key, name in self.HARDWARE_TARGETS.items():
                if key in main_target:
                    target_info["target_type"] = name
                    break

            # Extract target options
            options = {}
            option_matches = re.findall(r'-(\w+)=([^ ,]+)', target_str)
            for key, value in option_matches:
                options[key] = value

            if options:
                target_info["options"] = options

            # Check for special target configurations
            if "mcpu" in options:
                target_info["cpu_arch"] = options["mcpu"]

            if "mattr" in options:
                target_info["cpu_features"] = options["mattr"]

            # Extract additional information based on target type
            if "cuda" in target_str.lower():
                # Extract CUDA architecture if available
                arch_match = re.search(r'arch=(\w+)', target_str)
                if arch_match:
                    target_info["cuda_arch"] = arch_match.group(1)
                target_info["accelerator_type"] = "NVIDIA GPU"

            elif "opencl" in target_str.lower():
                target_info["accelerator_type"] = "OpenCL Device"

            elif "metal" in target_str.lower():
                target_info["accelerator_type"] = "Apple Metal GPU"

            elif "vulkan" in target_str.lower():
                target_info["accelerator_type"] = "Vulkan Device"

            elif "llvm" in target_str.lower():
                target_info["accelerator_type"] = "CPU (LLVM)"

        except Exception as e:
            self.logger.warning(f"Error parsing target string: {e}")

        return target_info

    def _try_tvm_runtime_analysis(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Attempt to use TVM runtime to load and analyze the model.

        Args:
            file_path: Path to the TVM model file
            metadata: Dictionary to update with runtime information
        """
        try:
            import tvm
            from tvm import relay
            from tvm.contrib import graph_executor

            # Different loading methods based on file type
            if file_path.endswith('.tar'):
                try:
                    # Try to load as module with parameters
                    lib = tvm.runtime.load_module(file_path)
                    metadata["runtime_load_success"] = True
                    metadata["runtime_type"] = "module"

                    # Get available functions
                    func_names = [func_name for func_name in lib.get_function_names()]
                    metadata["available_functions"] = func_names

                except Exception as e:
                    try:
                        # Fallback to graph executor
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Extract archive contents
                            with tarfile.open(file_path, 'r') as tar:
                                tar.extractall(path=temp_dir)

                            # Look for graph and params
                            graph_path = None
                            params_path = None
                            lib_path = None

                            for root, _, files in os.walk(temp_dir):
                                for file in files:
                                    if file.endswith('.json') or file == 'graph.json':
                                        graph_path = os.path.join(root, file)
                                    elif file.endswith('.params') or file == 'params':
                                        params_path = os.path.join(root, file)
                                    elif file.endswith('.so') or file == 'lib.so':
                                        lib_path = os.path.join(root, file)

                            if graph_path and lib_path:
                                # Load graph
                                with open(graph_path, 'r') as f:
                                    graph_json = f.read()

                                # Load lib
                                loaded_lib = tvm.runtime.load_module(lib_path)

                                # Load params if available
                                loaded_params = None
                                if params_path:
                                    loaded_params = tvm.runtime.load_param_dict(params_path)

                                # Create graph executor
                                ctx = tvm.cpu(0)
                                module = graph_executor.create(graph_json, loaded_lib, ctx)
                                if loaded_params:
                                    module.load_params(loaded_params)

                                metadata["runtime_load_success"] = True
                                metadata["runtime_type"] = "graph_executor"

                                # Get input names and shapes
                                try:
                                    input_names = []
                                    input_shapes = []
                                    graph_json_obj = json.loads(graph_json)
                                    if "attrs" in graph_json_obj and "dltype" in graph_json_obj["attrs"]:
                                        input_names = [f"input_{i}" for i in
                                                       range(len(graph_json_obj["attrs"]["dltype"]["data"]))]

                                    metadata["input_names"] = input_names
                                except Exception as inner_e:
                                    self.logger.warning(f"Error extracting input information: {inner_e}")
                    except Exception as graph_e:
                        metadata["runtime_load_success"] = False
                        metadata["runtime_error"] = str(e) + " / " + str(graph_e)

            elif file_path.endswith('.so'):
                try:
                    # Try to load as module
                    lib = tvm.runtime.load_module(file_path)
                    metadata["runtime_load_success"] = True
                    metadata["runtime_type"] = "module"

                    # Get available functions
                    func_names = [func_name for func_name in lib.get_function_names()]
                    metadata["available_functions"] = func_names
                except Exception as e:
                    metadata["runtime_load_success"] = False
                    metadata["runtime_error"] = str(e)

        except ImportError:
            self.logger.info("TVM runtime not available, skipping runtime analysis")
        except Exception as e:
            self.logger.warning(f"Error during TVM runtime analysis: {e}")

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check if it's a valid TVM model
        if metadata.get("format") == "tvm_tar":
            is_valid = metadata.get("is_valid_tvm_model", False)
            if not is_valid:
                return "Unknown Archive", ModelConfidence.LOW

        # Try to identify the model architecture
        model_name = None
        confidence = ModelConfidence.UNKNOWN

        # First check if we have direct model hints
        if "model_hints" in metadata and metadata["model_hints"]:
            model_hints = metadata["model_hints"]
            model_name = model_hints[0]
            confidence = ModelConfidence.MEDIUM

            # If multiple hints agree, increase confidence
            if len(model_hints) > 1:
                model_name = "/".join(model_hints[:2])

            # If we have framework hints as well, add them
            if "framework_hints" in metadata and metadata["framework_hints"]:
                framework = metadata["framework_hints"][0]
                model_name = f"{model_name} ({framework})"

        # Check graph information
        elif "graph_info" in metadata and metadata["graph_info"]:
            graph_info = metadata["graph_info"]

            # Check operator counts for hints
            if "operator_counts" in graph_info:
                ops = graph_info["operator_counts"]

                # CNN detection
                if "nn.conv2d" in ops or "conv2d" in ops:
                    # Check for specific architectures
                    if "input_shapes" in graph_info:
                        # Check input shape for clues
                        shapes = graph_info["input_shapes"]
                        if shapes and len(shapes) > 0:
                            if shapes[0][1] == 3:  # RGB input
                                model_name = "CNN (RGB input)"
                                confidence = ModelConfidence.MEDIUM

                    if not model_name:
                        model_name = "CNN"
                        confidence = ModelConfidence.MEDIUM

                # RNN detection
                elif any(op in ops for op in ["lstm", "gru", "rnn", "nn.lstm", "nn.gru"]):
                    model_name = "RNN/LSTM"
                    confidence = ModelConfidence.MEDIUM

                # Transformer detection
                elif any(op in ops for op in ["attention", "multihead", "layernorm", "self_attention"]):
                    model_name = "Transformer"
                    confidence = ModelConfidence.MEDIUM

                # Look for common model operations
                elif "dense" in ops or "nn.dense" in ops:
                    model_name = "Neural Network"
                    confidence = ModelConfidence.LOW

            # If we have input shapes, add that information
            if "input_shapes" in graph_info and graph_info["input_shapes"] and not model_name:
                shapes = graph_info["input_shapes"]
                if shapes:
                    shape_str = "x".join(str(dim) for dim in shapes[0])
                    model_name = f"TVM Model (Input: {shape_str})"
                    confidence = ModelConfidence.LOW

        # Check TVM metadata
        if not model_name and "tvm_metadata" in metadata:
            tvm_meta = metadata["tvm_metadata"]
            if "model" in tvm_meta:
                model_name = tvm_meta["model"]
                confidence = ModelConfidence.HIGH

        # If we have target information, add it
        target_info = None
        if "target_info" in metadata:
            if "target_type" in metadata["target_info"]:
                target_info = metadata["target_info"]["target_type"]
        elif "target_hints" in metadata and metadata["target_hints"]:
            target_info = metadata["target_hints"][0]

        if model_name and target_info:
            model_name = f"{model_name} for {target_info}"

        # Runtime information can increase confidence
        if metadata.get("runtime_load_success", False):
            if confidence == ModelConfidence.MEDIUM:
                confidence = ModelConfidence.HIGH
            elif confidence == ModelConfidence.LOW:
                confidence = ModelConfidence.MEDIUM

        # Fallback to format information
        if not model_name:
            format_type = metadata.get("format", "unknown")
            if format_type == "tvm_tar":
                model_name = "TVM Packaged Model"
            elif format_type == "tvm_so":
                model_name = "TVM Compiled Model"
            else:
                model_name = "Unknown Model"

            confidence = ModelConfidence.LOW

            # If we at least have target information
            if target_info:
                model_name = f"{model_name} for {target_info}"

        return model_name, confidence

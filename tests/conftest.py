"""
Common fixtures and utilities for testing the model_inspector library.
"""
import os
import tempfile
import shutil
import pytest
from pathlib import Path
import numpy as np
import random


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.

    Yields:
        Path to the temporary directory
    """
    temp_path = tempfile.mkdtemp(prefix="model_inspector_test_")
    yield temp_path
    try:
        shutil.rmtree(temp_path)
    except (PermissionError, OSError):
        # Sometimes Windows has issues removing temp files immediately
        pass


@pytest.fixture
def sample_files(temp_dir):
    """
    Create a set of sample files with different formats for testing.

    Args:
        temp_dir: Temporary directory (from fixture)

    Returns:
        Dictionary mapping file format to file path
    """
    files = {}

    # Create a directory structure
    main_dir = Path(temp_dir) / "models"
    main_dir.mkdir()

    subfolder = main_dir / "subfolder"
    subfolder.mkdir()

    # Create a sample safetensors file
    safetensors_path = main_dir / "model.safetensors"
    with open(safetensors_path, 'wb') as f:
        # Write safetensors header (simplified for testing)
        header_length = 100
        header = {
            "__metadata__": {
                "ss_network_module": "networks.lora",
                "ss_network_dim": 8,
                "ss_network_alpha": 4
            },
            "tensor1": {"dtype": "F32", "shape": [32, 32], "data_offsets": [0, 4096]}
        }
        import json
        header_bytes = json.dumps(header).encode('utf-8')
        header_length_bytes = len(header_bytes).to_bytes(8, byteorder='little')
        f.write(header_length_bytes)
        f.write(header_bytes)

        # Write dummy tensor data
        f.write(b'\x00' * 4096)  # 4KB of zeros

    files['safetensors'] = str(safetensors_path)

    # Create a sample ONNX file
    onnx_path = main_dir / "model.onnx"
    with open(onnx_path, 'wb') as f:
        # Write ONNX magic string and some dummy content
        f.write(b'ONNX\x00\x00\x00\x00')
        fake_model_proto = b'producer_name\x00onnx-example\x00domain\x00test\x00Conv\x00'
        f.write(fake_model_proto)
        f.write(b'\x00' * 1024)  # Padding

    files['onnx'] = str(onnx_path)

    # Create a PyTorch .pt file (simplified mock)
    pt_path = main_dir / "model.pt"
    with open(pt_path, 'wb') as f:
        # Write a fake PyTorch file header
        f.write(b'\x80\x02PyTorch')
        f.write(b'model_state_dict')
        f.write(b'\x00' * 1024)  # Padding

    files['pytorch'] = str(pt_path)

    # Create an HDF5-like file
    h5_path = main_dir / "model.h5"
    with open(h5_path, 'wb') as f:
        # Write the HDF5 signature
        f.write(b'\x89HDF\r\n\x1a\n')
        f.write(b'NUMPY1.0')
        # Write some fake content
        f.write(b'layer_names\x00conv1,conv2,dense1\x00')
        f.write(b'\x00' * 1024)  # Padding

    files['hdf5'] = str(h5_path)

    # Create a CoreML-like file
    mlmodel_path = subfolder / "model.mlmodel"
    with open(mlmodel_path, 'wb') as f:
        # Fake CoreML metadata
        f.write(b'CoreML\x00mlmodel\x00')
        f.write(b'neuralNetwork\x00')
        f.write(b'\x00' * 1024)  # Padding

    files['mlmodel'] = str(mlmodel_path)

    # Create a TensorRT-like file
    trt_path = subfolder / "model.engine"
    with open(trt_path, 'wb') as f:
        # Write TensorRT magic
        magic = bytes.fromhex('6a8b4567')
        f.write(magic)
        f.write(b'TensorRT-')
        f.write(b'\x00' * 1024)  # Padding

    files['tensorrt'] = str(trt_path)

    # Create a GGUF-like file
    gguf_path = main_dir / "model.gguf"
    with open(gguf_path, 'wb') as f:
        # Write GGUF magic
        f.write(b'GGUF')
        # Version
        f.write((1).to_bytes(4, byteorder='little'))
        # Counts
        f.write((10).to_bytes(4, byteorder='little'))  # 10 tensors
        f.write((5).to_bytes(4, byteorder='little'))  # 5 metadata items

        # Write fake metadata
        f.write(b'general.architecture\x00')
        f.write(b'llama\x00')
        f.write(b'\x00' * 1024)  # Padding

    files['gguf'] = str(gguf_path)

    # Create invalid/empty files
    empty_path = main_dir / "empty.onnx"
    with open(empty_path, 'wb') as f:
        pass
    files['empty'] = str(empty_path)

    invalid_path = main_dir / "invalid.bin"
    with open(invalid_path, 'wb') as f:
        f.write(b'NOT A VALID MODEL FILE')
    files['invalid'] = str(invalid_path)

    return files


@pytest.fixture
def mock_model_info():
    """Create a sample ModelInfo object."""
    from model_inspector.models.info import ModelInfo
    from model_inspector.models.confidence import ModelConfidence

    return ModelInfo(
        model_type="TestModel",
        confidence=ModelConfidence.HIGH,
        format=".test",
        metadata={"test_key": "test_value"},
        file_path="/path/to/model.test",
        file_size=1024,
        is_safe=True
    )


def create_test_directory_structure(base_dir, num_files=20, depth=3):
    """
    Create a more complex directory structure with various model files for testing.

    Args:
        base_dir: Base directory to create the structure in
        num_files: Number of files to create
        depth: Maximum directory depth

    Returns:
        Dictionary with test file information
    """
    formats = [
        ('.safetensors', b'header_length\x00\x00\x00\x00\x00'),
        ('.onnx', b'ONNX\x00\x00\x00\x00'),
        ('.pt', b'\x80\x02PyTorch'),
        ('.h5', b'\x89HDF\r\n\x1a\n'),
        ('.gguf', b'GGUF'),
        ('.bin', b'\x00bin\x00'),
        ('.mlmodel', b'mlmodel\x00'),
        ('.engine', b'\x6a\x8b\x45\x67'),
    ]

    created_files = {}
    directories = [Path(base_dir)]

    # Create directory structure
    for i in range(depth):
        new_dirs = []
        for parent in directories:
            for j in range(random.randint(1, 3)):
                new_dir = parent / f"dir_{i}_{j}"
                new_dir.mkdir(exist_ok=True)
                new_dirs.append(new_dir)
        directories.extend(new_dirs)

    # Create files
    for i in range(num_files):
        # Select random directory
        dir_path = random.choice(directories)
        # Select random format
        ext, content_start = random.choice(formats)

        # Create file
        file_path = dir_path / f"model_{i}{ext}"
        with open(file_path, 'wb') as f:
            f.write(content_start)
            # Add some random content with size between 1KB and 50KB
            size = random.randint(1024, 50 * 1024)
            f.write(b'\x00' * size)

        created_files[str(file_path)] = {
            'format': ext,
            'size': size + len(content_start)
        }

    return created_files


@pytest.fixture
def complex_directory(temp_dir):
    """
    Create a complex directory structure for testing scanner functionality.

    Args:
        temp_dir: Temporary directory (from fixture)

    Returns:
        Tuple of (base directory, file information dictionary)
    """
    complex_dir = Path(temp_dir) / "complex"
    complex_dir.mkdir()

    file_info = create_test_directory_structure(complex_dir)
    return str(complex_dir), file_info



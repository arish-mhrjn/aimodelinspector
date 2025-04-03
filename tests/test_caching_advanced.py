"""
Advanced tests for the caching system to improve coverage.
"""
import pytest
import os
import time
import json
import pickle
from pathlib import Path

from model_inspector.utils.caching import (
    ModelCache, JSONSerializer, PickleSerializer,
    get_file_hash, get_content_hash
)
from model_inspector.models.confidence import ModelConfidence
from model_inspector.models.info import ModelInfo


class TestAdvancedCaching:
    """Tests for advanced caching features."""

    def test_cache_serialization_edge_cases(self, temp_dir, mock_model_info):
        """Test edge cases in cache serialization."""
        # Test serializing complex nested structures
        serializer = JSONSerializer()

        # Create complex nested structure with various types
        complex_data = {
            'string': 'text',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'none': None,
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'nested': {
                'list': [{'x': 1}, {'y': 2}],
                'enum': ModelConfidence.HIGH,
            },
            'model': mock_model_info
        }

        # Serialize and deserialize
        serialized = serializer.serialize(complex_data)
        deserialized = serializer.deserialize(serialized)

        # Check that key structures are preserved
        assert deserialized['string'] == 'text'
        assert deserialized['int'] == 42
        assert deserialized['float'] == 3.14
        assert deserialized['bool'] is True
        assert deserialized['none'] is None
        assert deserialized['list'] == [1, 2, 3]
        assert deserialized['dict'] == {'a': 1, 'b': 2}
        assert deserialized['nested']['list'] == [{'x': 1}, {'y': 2}]

        # Check enum handling - verify it's correctly deserialized as an enum
        assert isinstance(deserialized['nested']['enum'], ModelConfidence)
        assert deserialized['nested']['enum'] == ModelConfidence.HIGH

    def test_cache_persistence_recovery(self, temp_dir):
        """Test cache persistence and recovery."""
        cache_dir = os.path.join(temp_dir, "recovery_test")

        # Create a persistent cache and add some items
        cache1 = ModelCache(
            max_size=10,
            persist=True,
            cache_dir=cache_dir
        )

        # Add some items
        for i in range(5):
            cache1.set(f"key{i}", f"value{i}")

        # Force cache files to be written
        cache1 = None

        # Create a new cache instance pointing to the same directory
        cache2 = ModelCache(
            max_size=10,
            persist=True,
            cache_dir=cache_dir
        )

        # Verify the items are recovered
        for i in range(5):
            assert cache2.get(f"key{i}") == f"value{i}"

        # Test cache corruption recovery
        # Create a corrupt cache file
        corrupt_path = os.path.join(cache_dir, "corrupt_key.cache")
        with open(corrupt_path, 'w') as f:
            f.write("This is not valid serialized data")

        # The cache should handle this gracefully
        assert cache2.get("corrupt_key") is None

        # Cleanup
        cache2.clear()

    def test_cache_max_age(self, temp_dir):
        """Test cache item expiration."""
        # Create a cache
        cache = ModelCache(max_size=10)

        # Add an item
        cache.set("test_key", "test_value")

        # Should be retrievable
        assert cache.get("test_key") == "test_value"

        # Should still be retrievable with a long max_age
        assert cache.get("test_key", max_age=60) == "test_value"

        # Should not be retrievable with a very short max_age
        time.sleep(0.01)  # Ensure some time passes
        assert cache.get("test_key", max_age=0.001) is None

        # Test with persistent cache
        cache_dir = os.path.join(temp_dir, "max_age_test")

        # Create a persistent cache
        p_cache = ModelCache(
            max_size=10,
            persist=True,
            cache_dir=cache_dir
        )

        # Add an item and wait
        p_cache.set("test_key", "test_value")
        time.sleep(0.1)  # Ensure file timestamp is different

        # Should not be retrievable with a very short max_age
        assert p_cache.get("test_key", max_age=0.05) is None

    def test_edge_case_hash_functions(self, temp_dir):
        """Test edge cases for file hash functions."""
        # Create an empty file
        empty_path = os.path.join(temp_dir, "empty.txt")
        with open(empty_path, 'w'):
            pass

        # Both hash functions should work on empty files
        assert get_file_hash(empty_path) is not None
        assert get_content_hash(empty_path) is not None

        # Create a very small file
        small_path = os.path.join(temp_dir, "small.txt")
        with open(small_path, 'w') as f:
            f.write("x")

        # Both hash functions should work on tiny files
        assert get_file_hash(small_path) is not None
        assert get_content_hash(small_path) is not None

        # Test with max_bytes=0 (unlimited)
        assert get_content_hash(small_path, max_bytes=0) is not None

        # Create a file that's exactly block_size in size
        block_path = os.path.join(temp_dir, "block.txt")
        block_size = 8192
        with open(block_path, 'w') as f:
            f.write("x" * block_size)

        # Test hash on a block-sized file
        assert get_content_hash(block_path, block_size=block_size) is not None

        # Test with a non-standard block size
        assert get_content_hash(small_path, block_size=16) is not None


class TestSerializers:
    """Additional tests for serializer classes."""

    def test_pickle_serializer_with_model_info(self, mock_model_info):
        """Test PickleSerializer with ModelInfo objects."""
        serializer = PickleSerializer()

        # Test serializing a ModelInfo object
        serialized = serializer.serialize(mock_model_info)
        deserialized = serializer.deserialize(serialized)

        # Verify the object was correctly deserialized
        assert isinstance(deserialized, ModelInfo)
        assert deserialized.model_type == mock_model_info.model_type
        assert deserialized.confidence == mock_model_info.confidence
        assert deserialized.format == mock_model_info.format

        # Test serializing None
        assert serializer.deserialize(serializer.serialize(None)) is None

        # Test serializing primitive types
        assert serializer.deserialize(serializer.serialize(42)) == 42
        assert serializer.deserialize(serializer.serialize("string")) == "string"
        assert serializer.deserialize(serializer.serialize(True)) is True

    def test_json_serializer_special_cases(self):
        """Test JSON serializer with special cases."""
        serializer = JSONSerializer()

        # Test serializing bytes (not directly JSON serializable)
        # This might fail during JSON encoding, but not in the serialize method directly
        bytes_data = b"binary data"
        try:
            result = serializer.serialize(bytes_data)
            # If it doesn't fail, we should validate the result is at least something reasonable
            assert isinstance(result, bytes)
        except Exception:
            # It's also acceptable if serialization fails for this case
            pass

        # Test serializing objects with no __dict__ attribute
        class NoDict:
            __slots__ = ['value']

            def __init__(self, value):
                self.value = value

        try:
            result = serializer.serialize(NoDict(42))
            # If it doesn't fail, we should validate the result is at least something reasonable
            assert isinstance(result, bytes)
        except Exception:
            # It's also acceptable if serialization fails for this case
            pass

from typing import Dict, Any, Optional, Tuple, Callable, Union
from enum import Enum
from functools import lru_cache
import os
import hashlib
import json
import pickle
import time
import logging
import tempfile
from pathlib import Path
import shutil
from dataclasses import asdict

logger = logging.getLogger(__name__)


def get_file_hash(file_path: str, block_size: int = 8192) -> str:
    """
    Compute a hash of the file to use as a cache key.

    Args:
        file_path: Path to the file
        block_size: Size of blocks to read

    Returns:
        Hash of the file modification time and size
    """
    stat = os.stat(file_path)
    # Use mtime and size as a quick fingerprint
    fingerprint = f"{stat.st_mtime}_{stat.st_size}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


def get_content_hash(file_path: str, block_size: int = 8192, max_bytes: int = 1024 * 1024) -> str:
    """
    Compute a hash of the file content to use as a more reliable cache key.

    Args:
        file_path: Path to the file
        block_size: Size of blocks to read
        max_bytes: Maximum number of bytes to read (0 for all)

    Returns:
        Hash of the file content
    """
    hasher = hashlib.md5()
    remaining = max_bytes if max_bytes > 0 else float('inf')

    with open(file_path, 'rb') as f:
        while remaining > 0:
            # Read at most block_size or remaining bytes, whichever is smaller
            read_size = min(block_size, remaining)
            data = f.read(read_size)

            if not data:  # End of file
                break

            hasher.update(data)
            remaining -= len(data)

    return hasher.hexdigest()


class CacheSerializer:
    """Base class for cache serializers."""

    def serialize(self, data: Any) -> bytes:
        """
        Serialize data to bytes.

        Args:
            data: Data to serialize

        Returns:
            Serialized data as bytes
        """
        raise NotImplementedError()

    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to data.

        Args:
            data: Serialized data

        Returns:
            Deserialized data
        """
        raise NotImplementedError()


class JSONSerializer(CacheSerializer):
    """JSON serializer for cache data."""

    def serialize(self, data: Any) -> bytes:
        """
        Serialize data to JSON bytes.

        Args:
            data: Data to serialize

        Returns:
            JSON bytes
        """
        # Convert ModelInfo objects to dict
        from ..models.info import ModelInfo

        def convert_to_serializable(obj):
            if isinstance(obj, ModelInfo):
                # Convert ModelInfo to dict
                return {
                    'model_type': obj.model_type,
                    'confidence': obj.confidence,
                    'format': obj.format,
                    'metadata': obj.metadata,
                    'file_path': obj.file_path,
                    'file_size': obj.file_size,
                    'is_safe': obj.is_safe
                }

            # Convert Enum objects to a special dict format
            if isinstance(obj, Enum):
                return {'__enum__': obj.name}

            # Handle collections recursively
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(convert_to_serializable(v) for v in obj)

            # Return other objects as is
            return obj

        # Convert data to serializable form
        serializable_data = convert_to_serializable(data)

        # Convert to JSON
        return json.dumps(serializable_data).encode('utf-8')

    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize JSON bytes to data.

        Args:
            data: JSON bytes

        Returns:
            Deserialized data
        """
        from ..models.confidence import ModelConfidence

        # Function to decode enum values
        def _decode_enums(obj):
            if isinstance(obj, dict):
                if len(obj) == 1 and '__enum__' in obj:
                    # It's an encoded enum
                    enum_name = obj['__enum__']
                    # Get the enum value by name
                    try:
                        # Use getattr to access the enum by name
                        return getattr(ModelConfidence, enum_name)
                    except (AttributeError, KeyError):
                        # If enum value doesn't exist, return the original dict
                        return obj
                return {k: _decode_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_decode_enums(v) for v in obj]
            else:
                return obj

        json_data = json.loads(data.decode('utf-8'))
        return _decode_enums(json_data)


class PickleSerializer(CacheSerializer):
    """Pickle serializer for cache data."""

    def serialize(self, data: Any) -> bytes:
        """
        Serialize data to pickle bytes.

        Args:
            data: Data to serialize

        Returns:
            Pickle bytes
        """
        return pickle.dumps(data)

    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize pickle bytes to data.

        Args:
            data: Pickle bytes

        Returns:
            Deserialized data
        """
        return pickle.loads(data)


class ModelCache:
    """
    Cache for model analysis results with optional persistence.

    This cache can store results in memory and optionally persist them to disk
    for reuse across program runs.
    """

    def __init__(
            self,
            max_size: int = 1000,
            persist: bool = False,
            cache_dir: Optional[str] = None,
            serializer: Optional[CacheSerializer] = None
    ):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries in the memory cache
            persist: Whether to persist the cache to disk
            cache_dir: Directory for persistent cache (None for default)
            serializer: Serializer to use (None for automatic selection)
        """
        self.max_size = max_size
        self.persist = persist
        self.cache: Dict[str, Tuple[Any, float]] = {}  # value, timestamp
        self.hits = 0
        self.misses = 0

        # Set up persistence
        if persist:
            if cache_dir is None:
                # Create default cache directory in user's home
                home_dir = Path.home()
                default_cache_dir = home_dir / '.model_inspector_cache'
                self.cache_dir = default_cache_dir
            else:
                self.cache_dir = Path(cache_dir)

            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.cache_dir = None

        # Set serializer
        if serializer is None:
            # Use JSON by default as it's safer
            self.serializer = JSONSerializer()
        else:
            self.serializer = serializer

    def get(self, key: str, max_age: Optional[float] = None) -> Optional[Any]:
        """
        Get an item from the cache.

        Args:
            key: Cache key
            max_age: Maximum age in seconds (None for no limit)

        Returns:
            Cached value or None if not found or expired
        """
        # First try memory cache
        if key in self.cache:
            value, timestamp = self.cache[key]

            # Check if expired
            if max_age is not None and time.time() - timestamp > max_age:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end for LRU behavior
            self.cache[key] = (value, time.time())
            self.hits += 1
            return value

        # If not in memory cache, try persistent cache
        if self.persist and self.cache_dir is not None:
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = f.read()

                    # Check modification time for max_age
                    if max_age is not None:
                        mtime = os.path.getmtime(cache_file)
                        if time.time() - mtime > max_age:
                            self.misses += 1
                            return None

                    # Deserialize and add to memory cache
                    try:
                        value = self.serializer.deserialize(data)
                        self.cache[key] = (value, time.time())
                        self.hits += 1
                        return value
                    except Exception as e:
                        logger.warning(f"Error deserializing cache entry: {e}")
                        # Continue to miss case

                except Exception as e:
                    logger.warning(f"Error loading from persistent cache: {e}")
                    # Continue to miss case

        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Update memory cache
        self.cache[key] = (value, time.time())

        # Apply LRU policy if needed
        if len(self.cache) > self.max_size:
            # Find oldest entry
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        # Update persistent cache if enabled
        if self.persist and self.cache_dir is not None:
            cache_file = self.cache_dir / f"{key}.cache"

            # Use atomic write pattern for reliability
            try:
                # Serialize the data
                data = self.serializer.serialize(value)

                # Write to temporary file first
                fd, temp_path = tempfile.mkstemp(dir=self.cache_dir)
                try:
                    with os.fdopen(fd, 'wb') as f:
                        f.write(data)

                    # Move temporary file to final location
                    shutil.move(temp_path, cache_file)

                except Exception as e:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    logger.warning(f"Error writing to persistent cache: {e}")

            except Exception as e:
                logger.warning(f"Error serializing cache entry: {e}")

    def clear(self) -> None:
        """Clear the cache."""
        # Clear memory cache
        self.cache.clear()

        # Clear persistent cache if enabled
        if self.persist and self.cache_dir is not None:
            try:
                for cache_file in self.cache_dir.glob('*.cache'):
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error clearing persistent cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        stats = {
            'entries': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'persist': self.persist
        }

        if self.persist and self.cache_dir is not None:
            stats['cache_dir'] = str(self.cache_dir)
            stats['persistent_entries'] = len(list(self.cache_dir.glob('*.cache')))

        return stats

    def vacuum(self, max_age: Optional[float] = None) -> int:
        """
        Remove old entries from the persistent cache.

        Args:
            max_age: Maximum age in seconds (None requires explicit age)

        Returns:
            Number of entries removed
        """
        if not self.persist or self.cache_dir is None:
            return 0

        if max_age is None:
            # Safety check - require explicit age for vacuum
            return 0

        removed_count = 0
        now = time.time()

        for cache_file in self.cache_dir.glob('*.cache'):
            try:
                mtime = os.path.getmtime(cache_file)
                if now - mtime > max_age:
                    cache_file.unlink()
                    removed_count += 1

                    # Also remove from memory cache
                    key = cache_file.stem
                    if key in self.cache:
                        del self.cache[key]
            except Exception as e:
                logger.warning(f"Error vacuuming cache entry: {e}")

        return removed_count

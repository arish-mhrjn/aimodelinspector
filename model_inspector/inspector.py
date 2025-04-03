from __future__ import annotations
import os
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
from contextlib import contextmanager

from .models.safety import SafetyLevel
from .models.confidence import ModelConfidence
from .models.info import ModelInfo
from .models.permissions import Permission, PermissionSet
from .config import InspectorConfig
from .utils.file_utils import scan_directory, get_file_extension
from .utils.caching import ModelCache, get_file_hash, get_content_hash, JSONSerializer
from .utils.progress import progress_iterator, ProgressCallback
from .utils.filtering import ModelFilter
from .sandbox import Sandbox, SandboxFactory
from .analyzers import get_analyzer_for_extension
from .exceptions import UnsupportedFormatError, UnsafeModelError, ModelAnalysisError, FileAccessError
from .exceptions import SecurityViolationError, SandboxError

# Register the new analyzers
from .analyzers.__init__ import register_analyzer
from .analyzers.coreml import CoreMLAnalyzer
from .analyzers.tensorrt import TensorRTAnalyzer

# Register new analyzers
register_analyzer(['.mlmodel'], CoreMLAnalyzer)
register_analyzer(['.engine', '.plan', '.trt'], TensorRTAnalyzer)


class ModelInspector:
    """
    Inspects directories for AI model files and identifies their types and metadata.

    This class can analyze various AI model formats including safetensors, GGUF,
    PyTorch, ONNX, CoreML, TensorRT, and more to determine their type and extract
    useful metadata.

    Attributes:
        base_dir (Path): The base directory to search for model files.
        config (InspectorConfig): Configuration for the inspector.
    """

    # Extensions we can potentially analyze
    SUPPORTED_EXTENSIONS = {
        '.safetensors', '.gguf', '.ggml', '.bin', '.pt', '.pth',
        '.onnx', '.pb', '.h5', '.hdf5', '.ckpt', '.mlmodel',
        '.engine', '.plan', '.trt'
    }

    # Formats that might be unsafe to load
    UNSAFE_FORMATS = {'.pb', '.ckpt', '.bin', '.pt', '.pth'}

    def __init__(
            self,
            base_dir: Union[str, Path],
            config: Optional[InspectorConfig] = None
    ):
        """
        Initialize the ModelInspector.

        Args:
            base_dir: Directory to scan for model files
            config: Configuration for the inspector
        """
        self.base_dir = Path(base_dir)
        self.config = config or InspectorConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize cache if enabled
        if self.config.enable_caching:
            self.cache = ModelCache(
                self.config.cache_size,
                persist=self.config.persistent_cache,
                cache_dir=self.config.cache_directory,
                serializer=JSONSerializer()  # Use JSON for better safety
            )
        else:
            self.cache = None

        # Initialize sandbox if enabled
        self.sandbox = None
        if self.config.enable_sandbox:
            self.sandbox = SandboxFactory.create_sandbox(
                self.config.sandbox_type,
                self.config.permissions,
                **self.config.sandbox_options
            )

    @contextmanager
    def _analyzer_for_file(self, file_path: str):
        """
        Context manager to get and configure an analyzer for a file.

        Args:
            file_path: Path to the file

        Yields:
            Configured analyzer for the file
        """
        extension = get_file_extension(file_path)

        try:
            analyzer = get_analyzer_for_extension(extension)

            # Configure the analyzer if there's specific configuration for it
            analyzer_type = analyzer.__class__.__name__
            if analyzer_type in self.config.analyzer_configs:
                analyzer.configure(self.config.analyzer_configs[analyzer_type])

            yield analyzer
        except UnsupportedFormatError:
            self.logger.warning(f"No analyzer available for {extension}")
            raise
        except Exception as e:
            self.logger.error(f"Error setting up analyzer for {file_path}: {e}")
            raise ModelAnalysisError(f"Error setting up analyzer for {file_path}: {e}")

    def directory_files(
            self,
            extensions: Optional[Set[str]] = None,
            min_size: Optional[int] = None,
            max_size: Optional[int] = None,
            show_progress: bool = False
    ) -> Iterator[str]:
        """
        Iterate over model files in the base directory.

        Args:
            extensions: Specific extensions to filter by (defaults to all supported)
            min_size: Minimum file size in bytes (defaults to config value)
            max_size: Maximum file size in bytes (defaults to config value)
            show_progress: Whether to show a progress bar

        Yields:
            Paths to model files meeting the criteria
        """
        if extensions is None:
            # Use supported extensions filtered by configuration
            extensions = {ext for ext in self.SUPPORTED_EXTENSIONS
                          if self.config.is_format_enabled(ext)}

        # Use config values if not specified
        if min_size is None:
            min_size = self.config.min_size

        if max_size is None:
            max_size = self.config.max_size

        # Use the scan_directory utility
        files = scan_directory(
            self.base_dir,
            extensions=extensions,
            recursive=self.config.recursive,
            min_size=min_size,
            max_size=max_size,
            exclude_patterns=self.config.exclude_patterns,
            include_patterns=self.config.include_patterns
        )

        # Wrap with progress bar if requested
        if show_progress:
            files = progress_iterator(
                files,
                desc="Scanning directory",
                config=self.config.get_progress_config()
            )

        yield from files

    def is_safe_to_load(self, file_path: str) -> bool:
        """
        Determine if a file is safe to load based on the current safety level.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file is safe to load, False otherwise
        """
        ext = get_file_extension(file_path)

        # If sandboxing is enabled, we can load any format
        if self.config.enable_sandbox:
            return True

        # Check safety level
        if self.config.safety_level == SafetyLevel.UNSAFE:
            return True

        # Check if it's an unsafe format
        if ext in self.UNSAFE_FORMATS:
            if self.config.safety_level == SafetyLevel.SAFE:
                return False
            else:  # WARN
                self.logger.warning(f"Loading potentially unsafe format: {ext} at {file_path}")
                return True

        return True

    def _check_permissions(self, file_path: str, analyzer: BaseAnalyzer) -> None:
        """
        Check if we have permission to analyze a file.

        Args:
            file_path: Path to the file
            analyzer: Analyzer to be used

        Raises:
            SecurityViolationError: If we don't have permission to analyze the file
        """
        # Skip checks if no permission system or sandbox is configured
        if not self.config.permissions or not self.sandbox:
            return

        # Get the file extension
        ext = get_file_extension(file_path)

        # Check basic file reading permission
        if not self.config.permissions.format_has_permission(ext, Permission.READ_FILE):
            raise SecurityViolationError(f"No permission to read files with extension: {ext}")

        # Check if analyzer needs to execute code
        if not analyzer.can_analyze_safely(file_path) and not self.config.permissions.format_has_permission(
                ext, Permission.EXECUTE_CODE
        ):
            raise SecurityViolationError(
                f"Analyzer for {ext} requires code execution but permission is denied"
            )

    def get_model_type(self, model_path: str) -> Optional[ModelInfo]:
        """
        Determine the type of model from the given path.

        Args:
            model_path: Path to the model file

        Returns:
            ModelInfo object containing model type and metadata, or None if unable to determine
        """
        path = Path(model_path)
        if not path.exists():
            raise FileAccessError(f"File not found: {model_path}")

        ext = get_file_extension(model_path)
        if ext not in self.SUPPORTED_EXTENSIONS or not self.config.is_format_enabled(ext):
            raise UnsupportedFormatError(f"Unsupported file extension: {ext}")

        if not self.is_safe_to_load(model_path):
            raise UnsafeModelError(f"File format is considered unsafe: {model_path}")

        # Check cache first if enabled
        if self.cache:
            file_hash = get_file_hash(model_path)
            cached_result = self.cache.get(file_hash)
            if cached_result:
                self.logger.debug(f"Using cached result for {model_path}")
                return cached_result

        try:
            with self._analyzer_for_file(model_path) as analyzer:
                # Check permissions
                self._check_permissions(model_path, analyzer)

                # Use sandbox if enabled and needed
                if self.sandbox and not analyzer.can_analyze_safely(model_path):
                    self.logger.debug(f"Using sandbox for {model_path}")

                    # Run analysis in sandbox
                    with self.sandbox.active_session():
                        model_type, confidence, metadata = analyzer.analyze(
                            model_path,
                            sandbox=self.sandbox
                        )
                else:
                    # Run analysis directly
                    model_type, confidence, metadata = analyzer.analyze(model_path)

                result = ModelInfo(
                    model_type=model_type,
                    confidence=confidence,
                    format=ext,
                    metadata=metadata,
                    file_path=model_path,
                    file_size=path.stat().st_size,
                    is_safe=ext not in self.UNSAFE_FORMATS
                )

                # Cache the result if caching is enabled
                if self.cache:
                    file_hash = get_file_hash(model_path)
                    self.cache.set(file_hash, result)

                return result

        except UnsafeModelError:
            # Re-raise safety-related exceptions
            raise
        except SecurityViolationError:
            # Re-raise security violations
            raise
        except SandboxError as e:
            self.logger.error(f"Sandbox error analyzing {model_path}: {e}")
            raise ModelAnalysisError(f"Sandbox error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error analyzing {model_path}: {e}")
            raise ModelAnalysisError(f"Error analyzing {model_path}: {e}")

    async def analyze_directory_async(
            self,
            extensions: Optional[Set[str]] = None,
            progress_callback: Optional[ProgressCallback] = None
    ) -> List[ModelInfo]:
        """
        Asynchronously analyze all files in the directory.

        Args:
            extensions: Specific extensions to filter by
            progress_callback: Callback for progress updates

        Returns:
            List of ModelInfo objects for all successfully analyzed files
        """
        results = []
        errors = []

        # Get list of files to process
        files = list(self.directory_files(extensions=extensions))
        total_files = len(files)

        # Notify of start
        if progress_callback:
            progress_callback.start()

        # Process in batches using a thread pool
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            loop = asyncio.get_event_loop()

            async def process_file(idx, file_path):
                """Process a single file and update progress."""
                try:
                    result = await loop.run_in_executor(
                        executor,
                        self.get_model_type,
                        file_path
                    )

                    # Update progress
                    if progress_callback:
                        progress_callback.progress(idx + 1, total_files)

                    return result
                except Exception as e:
                    errors.append((file_path, str(e)))

                    # Notify of error
                    if progress_callback:
                        progress_callback.error(e)

                    # Continue with other files
                    return None

            # Create and run tasks for all files
            tasks = [process_file(i, file_path) for i, file_path in enumerate(files)]
            for completed_task in await asyncio.gather(*tasks):
                if completed_task:
                    results.append(completed_task)

        # Notify of completion
        if progress_callback:
            progress_callback.complete(results)

        # Log any errors
        for file_path, error in errors:
            self.logger.error(f"Error processing {file_path}: {error}")

        return results

    def analyze_directory(
            self,
            extensions: Optional[Set[str]] = None,
            show_progress: bool = False
    ) -> List[ModelInfo]:
        """
        Analyze all files in the directory synchronously.

        Args:
            extensions: Specific extensions to filter by
            show_progress: Whether to show a progress bar

        Returns:
            List of ModelInfo objects for all successfully analyzed files
        """
        results = []
        errors = []

        # Get the file iterator
        files = self.directory_files(extensions=extensions)

        # Wrap with progress bar if requested
        if show_progress:
            files = progress_iterator(files, desc="Analyzing files")

        # Process each file
        for file_path in files:
            try:
                info = self.get_model_type(file_path)
                if info:
                    results.append(info)
            except Exception as e:
                errors.append((file_path, str(e)))
                self.logger.error(f"Error processing {file_path}: {e}")

        return results

    def __enter__(self) -> 'ModelInspector':
        """
        Enter the context manager.

        Returns:
            The ModelInspector instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        # Clean up resources
        if self.cache:
            # Don't clear the cache, just make sure any buffered writes are flushed
            pass

    def create_filtered_view(self, filter_function: Callable[[ModelInfo], bool]) -> 'InspectorFilterView':
        """
        Create a filtered view of the inspector's results.

        Args:
            filter_function: Function that takes a ModelInfo and returns a boolean

        Returns:
            An InspectorFilterView instance
        """
        return InspectorFilterView(self, filter_function)

    def apply_model_filter(self, model_filter: 'utils.filtering.ModelFilter') -> 'InspectorFilterView':
        """
        Apply a ModelFilter to create a filtered view.

        Args:
            model_filter: ModelFilter instance

        Returns:
            An InspectorFilterView instance
        """
        from .utils.filtering import ModelFilter

        # Create a filter function that converts ModelInfo to dict for the filter
        def filter_function(model_info: ModelInfo) -> bool:
            # Convert ModelInfo to dict for filtering
            model_dict = {
                'model_type': model_info.model_type,
                'confidence': model_info.confidence,
                'format': model_info.format,
                'metadata': model_info.metadata,
                'file_path': model_info.file_path,
                'file_size': model_info.file_size,
                'is_safe': model_info.is_safe
            }
            return model_filter.evaluate(model_dict)

        return InspectorFilterView(self, filter_function)

    def group_by_model_type(
            self,
            results: Optional[List[ModelInfo]] = None
    ) -> Dict[str, List[ModelInfo]]:
        """
        Group results by model type.

        Args:
            results: List of ModelInfo objects (uses analyze_directory() if None)

        Returns:
            Dictionary of model type to list of ModelInfo objects
        """
        if results is None:
            results = self.analyze_directory()

        grouped = {}
        for model_info in results:
            model_type = model_info.model_type
            if model_type not in grouped:
                grouped[model_type] = []
            grouped[model_type].append(model_info)

        return grouped

    def group_by_format(
            self,
            results: Optional[List[ModelInfo]] = None
    ) -> Dict[str, List[ModelInfo]]:
        """
        Group results by file format.

        Args:
            results: List of ModelInfo objects (uses analyze_directory() if None)

        Returns:
            Dictionary of format to list of ModelInfo objects
        """
        if results is None:
            results = self.analyze_directory()

        grouped = {}
        for model_info in results:
            format_name = model_info.format
            if format_name not in grouped:
                grouped[format_name] = []
            grouped[format_name].append(model_info)

        return grouped

    def get_analyzer_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the analyzers.

        Returns:
            Dictionary of analyzer statistics
        """
        stats = {}

        if self.cache:
            stats['cache'] = self.cache.get_stats()

        return stats

    @staticmethod
    def help(topic: Optional[str] = None) -> str:
        """
        Get help information about the ModelInspector.

        Args:
            topic: Optional specific topic to get help on

        Returns:
            Help text for the specified topic or general help
        """
        topics = {
            "general": """
                ModelInspector is a tool for identifying and extracting metadata from
                AI model files in various formats. It can recognize model architectures,
                training details, and other metadata.
            """,
            "formats": """
                Supported formats include:
                - safetensors: Safe format for ML models
                - GGUF/GGML: Quantized language models
                - PyTorch: .pt and .pth files
                - ONNX: Cross-platform neural network format
                - Checkpoint files: .ckpt
                - TensorFlow: .pb, SavedModel
                - HDF5: .h5 and .hdf5
            """,
            "safety": """
                Safety levels:
                - SAFE: Only load formats known to be completely safe
                - WARN: Load potentially unsafe formats with warnings
                - UNSAFE: Load all formats regardless of safety concerns

                Some formats may execute arbitrary code when loaded.
            """,
            "examples": """
                # Basic usage
                inspector = ModelInspector("/path/to/models")
                for file_path in inspector.directory_files():
                    model_info = inspector.get_model_type(file_path)
                    print(f"File: {file_path}, Type: {model_info.model_type}")

                # Async processing
                import asyncio
                inspector = ModelInspector("/path/to/models")
                results = asyncio.run(inspector.analyze_directory_async())
            """,
            "configuration": """
                Configuration options:

                # Create a configuration
                config = InspectorConfig(
                    recursive=True,              # Search subdirectories
                    max_workers=4,               # Number of worker threads
                    safety_level=SafetyLevel.SAFE,  # Safety level
                    enable_caching=True,         # Enable result caching
                    cache_size=1000              # Maximum cache entries
                )

                # Pass to inspector
                inspector = ModelInspector("/path/to/models", config=config)
            """,
            "progress": """
                # Using progress bars
                inspector = ModelInspector("/path/to/models")

                # Show progress during scanning
                for file_path in inspector.directory_files(show_progress=True):
                    print(file_path)

                # Show progress during analysis
                results = inspector.analyze_directory(show_progress=True)

                # Using callbacks for progress in async mode
                from model_inspector.utils.progress import ProgressCallback

                # Create a callback
                def on_progress(current, total):
                    print(f"Progress: {current}/{total}")

                callback = ProgressCallback(
                    on_progress=on_progress,
                    on_complete=lambda results: print(f"Completed with {len(results)} results")
                )

                # Use in async analysis
                results = await inspector.analyze_directory_async(
                    progress_callback=callback
                )
            """
        }

        if topic is None:
            # Return general help with available topics
            return (
                    "ModelInspector Help\n\n"
                    f"{topics['general'].strip()}\n\n"
                    "Available help topics: " + ", ".join(topics.keys()) + "\n"
                                                                           "Use ModelInspector.help(topic) to get specific help."
            )

        topic = topic.lower()
        if topic in topics:
            return f"Help for '{topic}':\n\n{topics[topic].strip()}"
        else:
            return f"Topic '{topic}' not found. Available topics: {', '.join(topics.keys())}"


class InspectorFilterView:
    """
    A filtered view of the ModelInspector's results.

    This class allows working with a subset of models that match specific
    criteria without modifying the original Inspector.
    """

    def __init__(self, inspector: ModelInspector, filter_function: Callable[[ModelInfo], bool]):
        """
        Initialize the filter view.

        Args:
            inspector: ModelInspector to create a view of
            filter_function: Function that determines which models to include
        """
        self.inspector = inspector
        self.filter_function = filter_function

    def directory_files(self, **kwargs) -> Iterator[str]:
        """
        Iterate over model files in the base directory, applying the filter.

        Args:
            **kwargs: Arguments passed to ModelInspector.directory_files()

        Yields:
            Paths to model files meeting the criteria
        """
        # This can't fully apply the filter since we don't have ModelInfo yet,
        # but it will use any extension filters, etc. from kwargs
        for file_path in self.inspector.directory_files(**kwargs):
            try:
                # Get model info to be able to apply the filter
                model_info = self.inspector.get_model_type(file_path)
                if model_info and self.filter_function(model_info):
                    yield file_path
            except Exception:
                # Skip files that cause errors
                continue

    def analyze_directory(self, **kwargs) -> List[ModelInfo]:
        """
        Analyze all files in the directory, returning only those that match the filter.

        Args:
            **kwargs: Arguments passed to ModelInspector.analyze_directory()

        Returns:
            List of ModelInfo objects that match the filter
        """
        results = self.inspector.analyze_directory(**kwargs)
        return [info for info in results if self.filter_function(info)]

    async def analyze_directory_async(self, **kwargs) -> List[ModelInfo]:
        """
        Asynchronously analyze the directory, returning only models that match the filter.

        Args:
            **kwargs: Arguments passed to ModelInspector.analyze_directory_async()

        Returns:
            List of ModelInfo objects that match the filter
        """
        results = await self.inspector.analyze_directory_async(**kwargs)
        return [info for info in results if self.filter_function(info)]

    def group_by_model_type(self) -> Dict[str, List[ModelInfo]]:
        """
        Group filtered results by model type.

        Returns:
            Dictionary of model type to list of ModelInfo objects
        """
        results = self.analyze_directory()
        return self.inspector.group_by_model_type(results)

    def group_by_format(self) -> Dict[str, List[ModelInfo]]:
        """
        Group filtered results by file format.

        Returns:
            Dictionary of format to list of ModelInfo objects
        """
        results = self.analyze_directory()
        return self.inspector.group_by_format(results)

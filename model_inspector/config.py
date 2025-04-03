from dataclasses import dataclass, field
from typing import Set, Optional, Dict, Any, List
from .models.safety import SafetyLevel
from .models.permissions import PermissionSet, Permission


@dataclass
class InspectorConfig:
    """Configuration for the ModelInspector."""

    # Basic configuration
    recursive: bool = True
    max_workers: int = 4
    safety_level: SafetyLevel = SafetyLevel.SAFE

    # Format handling
    enabled_formats: Optional[Set[str]] = None  # None means all supported formats
    disabled_formats: Set[str] = field(default_factory=set)

    # File filtering
    min_size: int = 0
    max_size: Optional[int] = None
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)

    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    persistent_cache: bool = False
    cache_directory: Optional[str] = None

    # Progress reporting
    progress_format: str = 'bar'  # 'bar', 'plain', 'pct', 'spinner'
    show_progress: bool = True

    # Sandbox configuration
    enable_sandbox: bool = False
    sandbox_type: str = 'inprocess'  # 'inprocess', 'process', 'container'
    sandbox_options: Dict[str, Any] = field(default_factory=dict)

    # Permissions
    permissions: Optional[PermissionSet] = None

    # Analyzer specific configurations
    analyzer_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if self.min_size < 0:
            raise ValueError("min_size cannot be negative")

        if self.max_size is not None and self.max_size < self.min_size:
            raise ValueError("max_size must be greater than or equal to min_size")

        if self.cache_size < 0:
            raise ValueError("cache_size cannot be negative")

        # Initialize permissions if not set
        if self.permissions is None:
            self.permissions = self._get_default_permissions()

        # Set sandbox options based on safety level
        if self.enable_sandbox:
            if self.sandbox_type not in ('inprocess', 'process', 'container'):
                raise ValueError(f"Invalid sandbox type: {self.sandbox_type}")

    def _get_default_permissions(self) -> PermissionSet:
        """
        Get the default PermissionSet based on safety level.

        Returns:
            PermissionSet with appropriate permissions
        """
        if self.safety_level == SafetyLevel.SAFE:
            return PermissionSet.create_safe_permissions()
        elif self.safety_level == SafetyLevel.WARN:
            # Default permissions with some restrictions
            permissions = PermissionSet()
            permissions.set_permission(Permission.EXECUTE_CODE, False)
            permissions.set_permission(Permission.CREATE_SUBPROCESS, False)
            return permissions
        else:  # UNSAFE
            return PermissionSet.create_unrestricted_permissions()

    def is_format_enabled(self, ext: str) -> bool:
        """
        Check if a format is enabled in this configuration.

        Args:
            ext: File extension to check

        Returns:
            True if the format is enabled, False otherwise
        """
        # Normalize extension
        if not ext.startswith('.'):
            ext = f'.{ext}'
        ext = ext.lower()

        # Check if explicitly disabled
        if ext in self.disabled_formats:
            return False

        # If enabled_formats is None, all formats are enabled unless explicitly disabled
        if self.enabled_formats is None:
            return True

        # Otherwise, check if format is in the enabled list
        return ext in self.enabled_formats

    def get_progress_config(self) -> 'utils.progress.ProgressConfig':
        """
        Get a ProgressConfig object from this configuration.

        Returns:
            ProgressConfig object
        """
        from .utils.progress import ProgressConfig, ProgressFormat

        # Convert string format to enum
        format_map = {
            'bar': ProgressFormat.BAR,
            'plain': ProgressFormat.PLAIN,
            'pct': ProgressFormat.PERCENTAGE,
            'spinner': ProgressFormat.SPINNER
        }

        progress_format = format_map.get(
            self.progress_format.lower(),
            ProgressFormat.BAR
        )

        return ProgressConfig(
            format=progress_format,
            # Other settings with defaults
        )
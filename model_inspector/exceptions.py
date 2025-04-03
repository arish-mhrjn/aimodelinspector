class ModelInspectorError(Exception):
    """Base exception for all model inspector errors."""
    pass


class UnsupportedFormatError(ModelInspectorError):
    """Raised when an unsupported format is encountered."""
    pass


class UnsafeModelError(ModelInspectorError):
    """Raised when an unsafe model is encountered with strict safety settings."""
    pass


class ModelAnalysisError(ModelInspectorError):
    """Raised when there's an error during model analysis."""
    pass


class InvalidConfigurationError(ModelInspectorError):
    """Raised when the configuration is invalid."""
    pass


class FileAccessError(ModelInspectorError):
    """Raised when a file cannot be accessed."""
    pass


class SecurityViolationError(ModelInspectorError):
    """Raised when an operation violates security policies or permissions."""
    pass


class SandboxError(ModelInspectorError):
    """Raised when there's an error within a sandbox environment."""
    pass


class PermissionError(ModelInspectorError):
    """Raised when an operation is attempted without required permissions."""
    pass


class CacheError(ModelInspectorError):
    """Raised when there's an error with the caching system."""
    pass

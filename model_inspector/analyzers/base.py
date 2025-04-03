# model_inspector/analyzers/base.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from ..models.confidence import ModelConfidence
import logging


class BaseAnalyzer(ABC):
    """Base class for all model analyzers."""

    def __init__(self):
        """Initialize the analyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a model file to determine its type and extract metadata.

        Args:
            file_path: Path to the model file

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        pass

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely without security risks.

        Args:
            file_path: Path to the model file

        Returns:
            True if the file can be analyzed safely, False otherwise
        """
        # By default, assume all files of supported formats can be analyzed safely
        # Subclasses should override this if they handle formats with security concerns
        return True

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions (including the dot)
        """
        # Subclasses should override this to return their supported extensions
        return set()

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the analyzer with specific settings.

        Args:
            config: Dictionary of configuration settings
        """
        # Override in subclasses to handle specific configuration
        pass

# model_inspector/models/info.py
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
from .confidence import ModelConfidence


@dataclass
class ModelInfo:
    """Contains information about an identified model."""
    model_type: str
    confidence: ModelConfidence
    format: str
    metadata: Dict[str, Any]
    file_path: str
    file_size: int
    is_safe: bool

    @property
    def filename(self) -> str:
        """Get just the filename without the path."""
        return Path(self.file_path).name

    @property
    def extension(self) -> str:
        """Get the file extension."""
        return Path(self.file_path).suffix.lower()

    @property
    def is_high_confidence(self) -> bool:
        """Return True if the identification has high confidence."""
        return self.confidence == ModelConfidence.HIGH

    def __str__(self) -> str:
        """String representation of the model info."""
        return f"{self.filename}: {self.model_type} ({self.confidence.name})"

# model_inspector/analyzers/diffusers.py
from typing import Dict, Any, Tuple, Optional, List, Set
import logging
from pathlib import Path

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer
from .diffusers_safetensors import DiffusersSafetensorsAnalyzer
from .diffusers_gguf import DiffusersGGUFAnalyzer
from .diffusers_bin import DiffusionBinAnalyzer


class DiffusersRouter(BaseAnalyzer):
    """
    Router analyzer that delegates to appropriate diffusion model analyzers.

    This analyzer automatically routes to the correct specialized analyzer
    based on file extension and content.
    """

    def __init__(self):
        """Initialize the Diffusers Router analyzer."""
        super().__init__()
        self.safetensors_analyzer = DiffusersSafetensorsAnalyzer()
        self.gguf_analyzer = DiffusersGGUFAnalyzer()
        self.bin_analyzer = DiffusionBinAnalyzer()

    def get_supported_extensions(self) -> set:
        """Get file extensions supported by this analyzer."""
        return self.safetensors_analyzer.get_supported_extensions() | \
               self.gguf_analyzer.get_supported_extensions() | \
               self.bin_analyzer.get_supported_extensions()

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Route the analysis to the appropriate specialized analyzer.

        Args:
            file_path: Path to the model file

        Returns:
            Tuple of (model_type, confidence, metadata)
        """
        ext = Path(file_path).suffix.lower()

        if ext == '.safetensors':
            return self.safetensors_analyzer.analyze(file_path)
        elif ext == '.gguf':
            return self.gguf_analyzer.analyze(file_path)
        elif ext in ['.pt', '.pth', '.ckpt', '.checkpoint']:
            return self.bin_analyzer.analyze(file_path)
        else:
            # Fallback to generic
            return "Unknown", ModelConfidence.LOW, {"error": "Unsupported file format"}


# For backward compatibility
DiffusersAnalyzer = DiffusersRouter

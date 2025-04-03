# model_inspector/__init__.py
from .inspector import ModelInspector
from .models.safety import SafetyLevel
from .models.confidence import ModelConfidence
from .models.info import ModelInfo
from .config import InspectorConfig
from .exceptions import *

__version__ = "0.1.0"

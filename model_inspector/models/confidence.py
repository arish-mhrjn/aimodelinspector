# model_inspector/models/confidence.py
from enum import Enum, IntEnum


class ModelConfidence(IntEnum):
    """
    Enum representing confidence in model type identification.

    Used to indicate how certain the analyzer is about the identified model type.
    """
    UNKNOWN = 0  # Unknown or couldn't determine
    LOW = 1  # Low confidence identification, guess based on limited info
    MEDIUM = 2  # Medium confidence based on some identifying features
    HIGH = 3  # High confidence based on clear identifying features

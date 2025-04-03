from enum import Enum, IntEnum


class SafetyLevel(IntEnum):
    """
    Enum representing the safety level for model inspection.

    This controls how strict the library is about loading potentially unsafe models.
    """
    SAFE = 0  # Only load formats known to be completely safe
    WARN = 1  # Load potentially unsafe formats but warn about them
    UNSAFE = 2  # Load all formats regardless of safety concerns

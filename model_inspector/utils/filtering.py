from typing import Dict, Any, Optional, List, Set, Callable, Union, TypeVar, Generic
from pathlib import Path
import os
import re
import fnmatch
from enum import Enum
from dataclasses import dataclass

T = TypeVar('T')


class FilterOperator(Enum):
    """Operators for filter conditions."""
    EQ = 'eq'  # Equal
    NE = 'ne'  # Not equal
    GT = 'gt'  # Greater than
    GE = 'ge'  # Greater than or equal
    LT = 'lt'  # Less than
    LE = 'le'  # Less than or equal
    CONTAINS = 'con'  # Contains substring
    STARTSWITH = 'sw'  # Starts with
    ENDSWITH = 'ew'  # Ends with
    MATCHES = 'match'  # Regex match
    IN = 'in'  # In list/set
    NOT_IN = 'not_in'  # Not in list/set


@dataclass
class FilterCondition:
    """
    A single filter condition.

    Examples:
        FilterCondition('file_size', FilterOperator.GT, 1024)
        FilterCondition('model_type', FilterOperator.EQ, 'BERT')
        FilterCondition('filename', FilterOperator.CONTAINS, 'model')
    """
    field: str
    operator: FilterOperator
    value: Any

    def evaluate(self, obj: Dict[str, Any]) -> bool:
        """
        Evaluate this condition against an object.

        Args:
            obj: Object to evaluate against

        Returns:
            True if the condition matches, False otherwise
        """
        # Get the field value, supporting nested paths with dot notation
        field_value = self._get_field_value(obj, self.field)

        # If field doesn't exist, condition doesn't match
        if field_value is None and self.field not in obj:
            return False

        # Evaluate based on operator
        if self.operator == FilterOperator.EQ:
            return field_value == self.value

        elif self.operator == FilterOperator.NE:
            return field_value != self.value

        elif self.operator == FilterOperator.GT:
            return field_value > self.value

        elif self.operator == FilterOperator.GE:
            return field_value >= self.value

        elif self.operator == FilterOperator.LT:
            return field_value < self.value

        elif self.operator == FilterOperator.LE:
            return field_value <= self.value

        elif self.operator == FilterOperator.CONTAINS:
            # Handle different container types
            if isinstance(field_value, str) and isinstance(self.value, str):
                return self.value in field_value
            elif hasattr(field_value, '__contains__'):
                return self.value in field_value
            return False

        elif self.operator == FilterOperator.STARTSWITH:
            if isinstance(field_value, str):
                return field_value.startswith(self.value)
            return False

        elif self.operator == FilterOperator.ENDSWITH:
            if isinstance(field_value, str):
                return field_value.endswith(self.value)
            return False

        elif self.operator == FilterOperator.MATCHES:
            if isinstance(field_value, str):
                return bool(re.search(self.value, field_value))
            return False

        elif self.operator == FilterOperator.IN:
            return field_value in self.value

        elif self.operator == FilterOperator.NOT_IN:
            return field_value not in self.value

        return False

    def _get_field_value(self, obj: Dict[str, Any], field: str) -> Any:
        """
        Get a field value, supporting nested paths with dot notation.

        Args:
            obj: Object to get value from
            field: Field path (e.g., "metadata.author")

        Returns:
            Field value or None if not found
        """
        if '.' not in field:
            return obj.get(field)

        parts = field.split('.')
        current = obj

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current


class Filter:
    """
    A filter that can combine multiple conditions.

    The filter can be configured with AND or OR logic and can contain
    multiple conditions.
    """

    def __init__(self, combine_with_and: bool = True):
        """
        Initialize the filter.

        Args:
            combine_with_and: If True, all conditions must match (AND logic)
                              If False, any condition can match (OR logic)
        """
        self.combine_with_and = combine_with_and
        self.conditions: List[FilterCondition] = []

    def add_condition(self, condition: FilterCondition) -> None:
        """
        Add a condition to the filter.

        Args:
            condition: FilterCondition to add
        """
        self.conditions.append(condition)

    def add(self, field: str, operator: FilterOperator, value: Any) -> None:
        """
        Add a condition using individual components.

        Args:
            field: Field name to filter on
            operator: FilterOperator to use
            value: Value to compare against
        """
        self.add_condition(FilterCondition(field, operator, value))

    def evaluate(self, obj: Dict[str, Any]) -> bool:
        """
        Evaluate this filter against an object.

        Args:
            obj: Object to evaluate against

        Returns:
            True if the filter matches, False otherwise
        """
        if not self.conditions:
            # Empty filter matches everything
            return True

        if self.combine_with_and:
            # All conditions must match
            return all(condition.evaluate(obj) for condition in self.conditions)
        else:
            # Any condition can match
            return any(condition.evaluate(obj) for condition in self.conditions)


class ModelFilter:
    """
    Filter specifically designed for filtering model files.

    This provides a more user-friendly interface for common filtering
    operations on model files and metadata.
    """

    def __init__(self):
        """Initialize the model filter."""
        self.filter = Filter(combine_with_and=True)

    def min_size(self, bytes_size: int) -> 'ModelFilter':
        """
        Filter by minimum file size.

        Args:
            bytes_size: Minimum size in bytes

        Returns:
            Self for method chaining
        """
        self.filter.add('file_size', FilterOperator.GE, bytes_size)
        return self

    def max_size(self, bytes_size: int) -> 'ModelFilter':
        """
        Filter by maximum file size.

        Args:
            bytes_size: Maximum size in bytes

        Returns:
            Self for method chaining
        """
        self.filter.add('file_size', FilterOperator.LE, bytes_size)
        return self

    def format(self, format_name: str) -> 'ModelFilter':
        """
        Filter by model format.

        Args:
            format_name: Format name (e.g., 'safetensors', 'onnx')

        Returns:
            Self for method chaining
        """
        # Handle format with or without dot
        if not format_name.startswith('.'):
            format_name = f'.{format_name}'

        self.filter.add('format', FilterOperator.EQ, format_name)
        return self

    def formats(self, format_names: List[str]) -> 'ModelFilter':
        """
        Filter by multiple model formats.

        Args:
            format_names: List of format names

        Returns:
            Self for method chaining
        """
        # Normalize formats
        normalized = [f'.{f}' if not f.startswith('.') else f for f in format_names]
        self.filter.add('format', FilterOperator.IN, normalized)
        return self

    def model_type(self, type_name: str, partial_match: bool = False) -> 'ModelFilter':
        """
        Filter by model type.

        Args:
            type_name: Model type name
            partial_match: If True, use contains instead of exact match

        Returns:
            Self for method chaining
        """
        if partial_match:
            self.filter.add('model_type', FilterOperator.CONTAINS, type_name)
        else:
            self.filter.add('model_type', FilterOperator.EQ, type_name)
        return self

    def path_contains(self, substring: str) -> 'ModelFilter':
        """
        Filter by path containing a substring.

        Args:
            substring: Substring to search for in path

        Returns:
            Self for method chaining
        """
        self.filter.add('file_path', FilterOperator.CONTAINS, substring)
        return self

    def path_matches(self, pattern: str) -> 'ModelFilter':
        """
        Filter by path matching a glob pattern.

        Args:
            pattern: Glob pattern to match path against

        Returns:
            Self for method chaining
        """
        self.filter.add_condition(PatternMatchCondition('file_path', pattern))
        return self

    def confidence(self, min_level: Union[str, Enum]) -> 'ModelFilter':
        """
        Filter by minimum confidence level.

        Args:
            min_level: Minimum confidence level (enum or name)

        Returns:
            Self for method chaining
        """
        # Handle both enum instances and string names
        from ..models.confidence import ModelConfidence

        if isinstance(min_level, str):
            min_level = ModelConfidence[min_level]

        # Convert to numeric value for comparison
        min_value = min_level.value
        self.filter.add_condition(ConfidenceCondition(min_value))
        return self

    def metadata_contains(self, key: str, value: Any) -> 'ModelFilter':
        """
        Filter by metadata containing a specific key-value pair.

        Args:
            key: Metadata key
            value: Value to match

        Returns:
            Self for method chaining
        """
        self.filter.add(f'metadata.{key}', FilterOperator.EQ, value)
        return self

    def metadata_exists(self, key: str) -> 'ModelFilter':
        """
        Filter by metadata containing a specific key.

        Args:
            key: Metadata key

        Returns:
            Self for method chaining
        """
        self.filter.add_condition(KeyExistsCondition(f'metadata.{key}'))
        return self

    def evaluate(self, model_info: Dict[str, Any]) -> bool:
        """
        Evaluate this filter against a model info object.

        Args:
            model_info: Model info object

        Returns:
            True if the filter matches, False otherwise
        """
        return self.filter.evaluate(model_info)


class PatternMatchCondition(FilterCondition):
    """Special condition for matching file paths against glob patterns."""

    def __init__(self, field: str, pattern: str):
        """
        Initialize the pattern match condition.

        Args:
            field: Field to match
            pattern: Glob pattern
        """
        super().__init__(field, FilterOperator.MATCHES, pattern)

    def evaluate(self, obj: Dict[str, Any]) -> bool:
        """
        Evaluate using fnmatch for glob pattern matching.

        Args:
            obj: Object to evaluate against

        Returns:
            True if the path matches the pattern
        """
        value = self._get_field_value(obj, self.field)
        if value is None:
            return False

        return fnmatch.fnmatch(value, self.value)


class KeyExistsCondition(FilterCondition):
    """Special condition that checks if a key exists in metadata."""

    def __init__(self, field: str):
        """
        Initialize the key exists condition.

        Args:
            field: Field to check existence of
        """
        super().__init__(field, FilterOperator.EQ, None)

    def evaluate(self, obj: Dict[str, Any]) -> bool:
        """
        Evaluate by checking if the field exists.

        Args:
            obj: Object to evaluate against

        Returns:
            True if the field exists
        """
        parts = self.field.split('.')
        current = obj

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False

        return True


class ConfidenceCondition(FilterCondition):
    """Special condition for comparing confidence levels."""

    def __init__(self, min_value: int):
        """
        Initialize the confidence condition.

        Args:
            min_value: Minimum confidence value
        """
        super().__init__('confidence', FilterOperator.GE, min_value)

    def evaluate(self, obj: Dict[str, Any]) -> bool:
        """
        Evaluate by comparing confidence values.

        Args:
            obj: Object to evaluate against

        Returns:
            True if the confidence is high enough
        """
        confidence = obj.get('confidence')

        # Handle both enum instances and raw values
        if confidence is None:
            return False

        if hasattr(confidence, 'value'):
            # It's an enum
            confidence_value = confidence.value
        else:
            # Assume it's already a value
            confidence_value = confidence

        return confidence_value >= self.value

from typing import Dict, Any, Tuple, Optional, List, Set
import struct
import logging
import json
from pathlib import Path
import re
from collections import defaultdict, Counter
import os

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class XGBoostAnalyzer(BaseAnalyzer):
    """
    Analyzer for XGBoost model files (.json, .ubj formats).

    This analyzer extracts information from XGBoost models, which are widely used
    for gradient boosting decision tree algorithms in machine learning tasks including
    classification, regression, and ranking.

    Potential improvements:
    - Support for binary (.model) format XGBoost files
    - Add parsing of detailed tree structure for model architecture visualization
    - Extract feature importance data when available
    - Support for newer XGBoost model versions as they're released
    - Improve detection of specific model variants (e.g., linear boosters vs tree boosters)
    """

    # XGBoost JSON expected keys
    EXPECTED_JSON_KEYS = ['version', 'learner', 'Config']

    # XGBoost objective function types and their human-readable descriptions
    OBJECTIVE_TYPES = {
        'reg:squarederror': 'Regression with squared error',
        'reg:squaredlogerror': 'Regression with squared log error',
        'reg:logistic': 'Logistic regression',
        'reg:pseudohubererror': 'Regression with Pseudo Huber error',
        'binary:logistic': 'Binary classification',
        'binary:logitraw': 'Binary classification (raw logits)',
        'binary:hinge': 'Binary classification with hinge loss',
        'count:poisson': 'Count data with Poisson regression',
        'survival:cox': 'Survival analysis with Cox regression',
        'survival:aft': 'Survival analysis with AFT loss',
        'multi:softmax': 'Multiclass classification',
        'multi:softprob': 'Multiclass classification with probabilities',
        'rank:pairwise': 'Learning to rank with pairwise loss',
        'rank:ndcg': 'Learning to rank with NDCG loss',
        'rank:map': 'Learning to rank with MAP loss',
        'reg:gamma': 'Gamma regression with log-link'
    }

    # Booster types
    BOOSTER_TYPES = {
        'gbtree': 'Gradient Boosted Tree',
        'gblinear': 'Generalized Linear Model',
        'dart': 'Dropout Additive Regression Tree'
    }

    def __init__(self):
        """Initialize the XGBoost model analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.json', '.ubj'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze an XGBoost model file to determine its model type and metadata.

        Args:
            file_path: Path to the XGBoost model file (.json/.ubj)

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid XGBoost model
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            metadata = self._parse_xgboost_model(file_path)
            model_type, confidence = self._determine_model_type(metadata)
            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing XGBoost file {file_path}: {e}")
            raise

    def _parse_xgboost_model(self, file_path: str) -> Dict[str, Any]:
        """
        Parse an XGBoost model file.

        Args:
            file_path: Path to the XGBoost model file

        Returns:
            Extracted metadata

        Raises:
            ValueError: If the file is not a valid XGBoost model
        """
        metadata = {
            'format': 'unknown',
            'file_size_bytes': os.path.getsize(file_path)
        }

        ext = os.path.splitext(file_path)[1].lower()

        try:
            model_data = None

            # Parse JSON format
            if ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                metadata['format'] = 'xgboost_json'

            # Parse Universal Binary JSON format
            elif ext == '.ubj':
                try:
                    import uberjson
                    with open(file_path, 'rb') as f:
                        model_data = uberjson.load(f)
                    metadata['format'] = 'xgboost_ubj'
                except ImportError:
                    self.logger.warning("uberjson module not available, cannot parse .ubj file")
                    return metadata

            # Validate that this is an XGBoost model
            if not self._is_valid_xgboost_model(model_data):
                raise ValueError("Not a valid XGBoost model file")

            # Extract metadata from the model
            metadata = self._extract_metadata(model_data, metadata)

            return metadata

        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")

    def _is_valid_xgboost_model(self, model_data: Dict) -> bool:
        """
        Check if the data appears to be a valid XGBoost model.

        Args:
            model_data: Parsed model data

        Returns:
            True if it appears to be a valid XGBoost model, False otherwise
        """
        if not model_data or not isinstance(model_data, dict):
            return False

        # Check for key XGBoost model components
        has_version = 'version' in model_data
        has_learner = 'learner' in model_data

        # Some XGBoost models don't have 'Config' at the top level
        return has_version and has_learner

    def _extract_metadata(self, model_data: Dict, metadata: Dict) -> Dict[str, Any]:
        """
        Extract metadata from an XGBoost model.

        Args:
            model_data: Parsed model data
            metadata: Initial metadata dictionary

        Returns:
            Updated metadata dictionary
        """
        # Extract basic model information
        metadata['version'] = model_data.get('version', None)

        learner = model_data.get('learner', {})
        attributes = learner.get('attributes', {})

        # Extract objective function
        objective = attributes.get('objective', None)
        if objective:
            metadata['objective'] = objective
            metadata['objective_description'] = self.OBJECTIVE_TYPES.get(objective, objective)

        # Extract booster type
        booster = attributes.get('booster', None)
        if booster:
            metadata['booster_type'] = booster
            metadata['booster_description'] = self.BOOSTER_TYPES.get(booster, booster)

        # Extract number of features
        try:
            n_features = int(attributes.get('feature_names', '').count(',') + 1)
            metadata['num_features'] = n_features
        except (ValueError, AttributeError):
            pass

        # Extract feature names if available
        feature_names = attributes.get('feature_names', '')
        if feature_names:
            try:
                metadata['feature_names'] = feature_names.split(',')
            except Exception:
                pass

        # Extract number of classes for classification tasks
        if 'num_class' in attributes:
            try:
                metadata['num_classes'] = int(attributes.get('num_class', 0))
            except ValueError:
                pass

        # Extract model parameters
        if 'learner_model_param' in learner:
            model_param = learner.get('learner_model_param', {})
            metadata['base_score'] = self._safe_float_parse(model_param.get('base_score', None))
            metadata['num_boosting_rounds'] = self._safe_int_parse(model_param.get('num_feature', None))

        # Extract gradient booster information
        gbm = learner.get('gradient_booster', {})

        # Extract tree information for gbtree/dart boosters
        if booster in ('gbtree', 'dart'):
            model = gbm.get('model', {})
            trees_data = model.get('trees', [])

            metadata['num_trees'] = len(trees_data)

            if len(trees_data) > 0:
                # Sample first tree to analyze structure
                first_tree = trees_data[0]
                metadata['tree_stats'] = self._analyze_tree_stats(trees_data)

        # Extract weights for linear booster
        elif booster == 'gblinear':
            model = gbm.get('model', {})
            weights = model.get('weights', [])
            metadata['num_weights'] = len(weights) if isinstance(weights, list) else 0

        return metadata

    def _safe_float_parse(self, value: Any) -> Optional[float]:
        """Safely parse a float value."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_int_parse(self, value: Any) -> Optional[int]:
        """Safely parse an int value."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _analyze_tree_stats(self, trees_data: List[Dict]) -> Dict[str, Any]:
        """
        Extract summary statistics about the trees.

        Args:
            trees_data: List of tree data dictionaries

        Returns:
            Dictionary of tree statistics
        """
        stats = {
            'average_depth': 0,
            'max_depth': 0,
            'total_nodes': 0,
        }

        if not trees_data:
            return stats

        depths = []
        node_counts = []

        for tree in trees_data:
            # Count nodes and depth by analyzing tree structure
            # For XGBoost JSON, we'd need to traverse the tree using 'left_children' and 'right_children'
            # as this information isn't directly available

            # For now, we'll use a simple estimate based on the size of the node arrays
            if 'left_children' in tree and isinstance(tree['left_children'], list):
                node_count = len(tree['left_children'])
                node_counts.append(node_count)

                # Estimate tree depth - in binary trees, depth is approximately log2(nodes)
                if node_count > 0:
                    depth = max(1, int(1.5 * (node_count + 1).bit_length()) - 1)  # Rough estimate
                    depths.append(depth)

        if depths:
            stats['average_depth'] = sum(depths) / len(depths)
            stats['max_depth'] = max(depths)

        if node_counts:
            stats['average_nodes'] = sum(node_counts) / len(node_counts)
            stats['total_nodes'] = sum(node_counts)

        return stats

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence level from metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # For invalid or unrecognized models
        if metadata.get('format') == 'unknown':
            return "Unknown", ModelConfidence.UNKNOWN

        # Get base type from booster
        booster_type = metadata.get('booster_type', '')
        objective = metadata.get('objective', '')

        # Determine base model type from booster and objective
        model_type = "XGBoost"
        confidence = ModelConfidence.HIGH

        # Add model variant based on booster type
        if booster_type in self.BOOSTER_TYPES:
            booster_name = self.BOOSTER_TYPES[booster_type]
            model_type = f"XGBoost-{booster_name}"

            # For specific task types, provide more detail
            if objective:
                # Classification
                if 'binary:' in objective:
                    model_type += " (Binary Classification)"
                elif 'multi:' in objective:
                    model_type += " (Multiclass Classification)"
                # Regression
                elif 'reg:' in objective:
                    model_type += " (Regression)"
                # Ranking
                elif 'rank:' in objective:
                    model_type += " (Ranking)"
                # Survival
                elif 'survival:' in objective:
                    model_type += " (Survival Analysis)"

        elif booster_type:
            # Unknown booster type
            model_type = f"XGBoost-{booster_type}"
            confidence = ModelConfidence.MEDIUM

        # If we can't determine the booster type but it's still an XGBoost model
        else:
            confidence = ModelConfidence.MEDIUM

        return model_type, confidence

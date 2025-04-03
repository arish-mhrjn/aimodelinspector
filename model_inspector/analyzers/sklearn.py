# model_inspector/analyzers/sklearn.py
from typing import Dict, Any, Tuple, List, Optional, Set
import os
import pickle
import logging
import re
import json
from pathlib import Path
import struct
from collections import defaultdict

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class SklearnAnalyzer(BaseAnalyzer):
    """
    Analyzer for Scikit-learn models (.joblib, .pkl).

    This analyzer can identify and extract metadata from Scikit-learn models
    saved via joblib or pickle, providing information about model type,
    hyperparameters, and other relevant details.
    """

    # Known model class patterns and their readable names
    MODEL_PATTERNS = {
        r'linear_model\..*': ('LinearModel', ModelConfidence.HIGH),
        r'ensemble\..*Random.*Forest': ('RandomForest', ModelConfidence.HIGH),
        r'ensemble\..*Boosting': ('GradientBoosting', ModelConfidence.HIGH),
        r'ensemble\..*AdaBoost': ('AdaBoost', ModelConfidence.HIGH),
        r'svm\..*SVM': ('SVM', ModelConfidence.HIGH),
        r'svm\..*SVR': ('SVR', ModelConfidence.HIGH),
        r'tree\..*Tree': ('DecisionTree', ModelConfidence.HIGH),
        r'cluster\..*Means': ('KMeans', ModelConfidence.HIGH),
        r'neighbors\..*Neighbor': ('NearestNeighbors', ModelConfidence.HIGH),
        r'naive_bayes\..*': ('NaiveBayes', ModelConfidence.HIGH),
        r'neural_network\..*MLP': ('NeuralNetwork', ModelConfidence.HIGH),
        r'decomposition\..*PCA': ('PCA', ModelConfidence.HIGH),
        r'preprocessing\..*': ('Preprocessor', ModelConfidence.HIGH),
        r'pipeline\..*': ('Pipeline', ModelConfidence.HIGH),
        r'.*XGBRegressor': ('XGBoost-Regressor', ModelConfidence.HIGH),
        r'.*XGBClassifier': ('XGBoost-Classifier', ModelConfidence.HIGH),
        r'.*LGBMRegressor': ('LightGBM-Regressor', ModelConfidence.HIGH),
        r'.*LGBMClassifier': ('LightGBM-Classifier', ModelConfidence.HIGH),
        r'.*CatBoostRegressor': ('CatBoost-Regressor', ModelConfidence.HIGH),
        r'.*CatBoostClassifier': ('CatBoost-Classifier', ModelConfidence.HIGH),
    }

    def __init__(self):
        """Initialize the Scikit-learn analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.joblib', '.pkl', '.pickle'}

    def can_analyze_safely(self, file_path: str) -> bool:
        """
        Check if the file can be analyzed safely.

        Pickle/joblib files can contain arbitrary code that executes on load.

        Args:
            file_path: Path to the file

        Returns:
            False as pickle/joblib files can execute arbitrary code
        """
        # Pickle and joblib files can contain arbitrary code
        return False

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a Scikit-learn model file.

        Args:
            file_path: Path to the model file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Determine file format (joblib or pickle)
            format_type = self._determine_file_format(file_path)

            # Extract metadata without loading the model
            metadata = self._extract_metadata_safely(file_path, format_type)

            # Add file format to metadata
            metadata['file_format'] = format_type

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing Scikit-learn model {file_path}: {e}")
            raise

    def _determine_file_format(self, file_path: str) -> str:
        """
        Determine if the file is a joblib or pickle format.

        Args:
            file_path: Path to the model file

        Returns:
            'joblib' or 'pickle'
        """
        # Check file extension first
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.joblib':
            return 'joblib'

        # Check file signature
        with open(file_path, 'rb') as f:
            header = f.read(256)

            # Joblib files start with specific markers
            if b'joblib' in header:
                return 'joblib'

            # Check for pickle protocol signatures
            if header.startswith(b'\x80\x03') or header.startswith(b'\x80\x04') or \
                    header.startswith(b'\x80\x02'):
                return 'pickle'

        # Default to pickle if unknown
        return 'pickle'

    def _extract_metadata_safely(self, file_path: str, format_type: str) -> Dict[str, Any]:
        """
        Extract metadata from the model file without executing code.

        Args:
            file_path: Path to the model file
            format_type: 'joblib' or 'pickle'

        Returns:
            Extracted metadata dictionary
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
        }

        # Scan file for signs of sklearn
        with open(file_path, 'rb') as f:
            content = f.read()

            # Look for sklearn package markers
            sklearn_markers = [b'sklearn', b'scikit-learn']
            for marker in sklearn_markers:
                if marker in content:
                    metadata['has_sklearn_marker'] = True
                    break

            # Look for popular external libraries
            external_libs = {
                'xgboost': b'xgboost',
                'lightgbm': b'lightgbm',
                'catboost': b'catboost',
                'tensorflow': b'tensorflow',
                'keras': b'keras',
                'pytorch': b'torch'
            }

            detected_libs = []
            for lib_name, marker in external_libs.items():
                if marker in content:
                    detected_libs.append(lib_name)

            if detected_libs:
                metadata['detected_libraries'] = detected_libs

            # Extract class name hints for model type detection
            class_markers = [
                (b'RandomForestClassifier', 'RandomForest-Classifier'),
                (b'RandomForestRegressor', 'RandomForest-Regressor'),
                (b'GradientBoostingClassifier', 'GradientBoosting-Classifier'),
                (b'GradientBoostingRegressor', 'GradientBoosting-Regressor'),
                (b'LogisticRegression', 'LogisticRegression'),
                (b'LinearRegression', 'LinearRegression'),
                (b'SVC', 'SupportVectorClassifier'),
                (b'SVR', 'SupportVectorRegressor'),
                (b'DecisionTreeClassifier', 'DecisionTree-Classifier'),
                (b'DecisionTreeRegressor', 'DecisionTree-Regressor'),
                (b'KMeans', 'KMeans-Clustering'),
                (b'DBSCAN', 'DBSCAN-Clustering'),
                (b'Pipeline', 'Pipeline'),
                (b'ColumnTransformer', 'ColumnTransformer'),
                (b'StandardScaler', 'StandardScaler'),
                (b'MinMaxScaler', 'MinMaxScaler'),
                (b'PCA', 'PCA'),
                (b'KernelPCA', 'KernelPCA'),
                (b'MLPClassifier', 'NeuralNetwork-Classifier'),
                (b'MLPRegressor', 'NeuralNetwork-Regressor'),
                (b'XGBClassifier', 'XGBoost-Classifier'),
                (b'XGBRegressor', 'XGBoost-Regressor'),
                (b'LGBMClassifier', 'LightGBM-Classifier'),
                (b'LGBMRegressor', 'LightGBM-Regressor'),
                (b'CatBoostClassifier', 'CatBoost-Classifier'),
                (b'CatBoostRegressor', 'CatBoost-Regressor'),
            ]

            model_type_hints = []
            for marker, model_type in class_markers:
                if marker in content:
                    model_type_hints.append(model_type)

            if model_type_hints:
                metadata['model_type_hints'] = model_type_hints

        # Look for metadata or config files in the same directory
        dir_path = Path(file_path).parent
        model_name = Path(file_path).stem

        # Check for any metadata files
        metadata_files = [
            f"{model_name}_metadata.json",
            f"{model_name}_params.json",
            f"{model_name}_config.json",
            "model_metadata.json",
            "model_config.json"
        ]

        for metadata_file in metadata_files:
            meta_path = dir_path / metadata_file
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        model_metadata = json.load(f)
                        metadata['external_metadata'] = model_metadata

                        # Try to extract hyperparameters
                        if 'hyperparameters' in model_metadata:
                            metadata['hyperparameters'] = model_metadata['hyperparameters']
                        elif 'params' in model_metadata:
                            metadata['hyperparameters'] = model_metadata['params']

                        break
                except (json.JSONDecodeError, IOError):
                    self.logger.warning(f"Failed to read metadata file: {metadata_file}")

        return metadata

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine the model type based on extracted metadata.

        Args:
            metadata: Extracted metadata

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check for direct model type hints first
        if 'model_type_hints' in metadata and metadata['model_type_hints']:
            # Use the first hint as the most likely model type
            return metadata['model_type_hints'][0], ModelConfidence.HIGH

        # Check for external metadata that might specify model type
        if 'external_metadata' in metadata:
            ext_meta = metadata['external_metadata']

            # Look for model type or class information
            for key in ['model_type', 'model_class', 'algorithm', 'type']:
                if key in ext_meta:
                    model_id = ext_meta[key]

                    # Check if it matches known patterns
                    for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
                        if re.search(pattern, model_id, re.IGNORECASE):
                            return model_type, confidence

                    # Return as is if not matching any pattern
                    return model_id, ModelConfidence.MEDIUM

        # Check for detected libraries
        if 'detected_libraries' in metadata:
            libs = metadata['detected_libraries']

            if 'xgboost' in libs:
                return "XGBoost", ModelConfidence.MEDIUM
            elif 'lightgbm' in libs:
                return "LightGBM", ModelConfidence.MEDIUM
            elif 'catboost' in libs:
                return "CatBoost", ModelConfidence.MEDIUM
            elif 'tensorflow' in libs or 'keras' in libs:
                return "TensorFlow/Keras", ModelConfidence.MEDIUM
            elif 'pytorch' in libs:
                return "PyTorch", ModelConfidence.MEDIUM

        # Check for sklearn marker
        if metadata.get('has_sklearn_marker', False):
            return "Scikit-learn-Model", ModelConfidence.MEDIUM

        # Default fallback
        return "Unknown-ML-Model", ModelConfidence.LOW

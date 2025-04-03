# cuml.py
from typing import Dict, Any, Tuple, Optional, List, Set
import struct
import logging
import json
from pathlib import Path
import re
from collections import defaultdict, Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer

"""
Analyzer module for RAPIDS cuML model files (.cuml).

This module provides functionality to analyze machine learning models saved in the 
RAPIDS cuML format. RAPIDS cuML is a suite of GPU-accelerated machine learning libraries
that provides implementations of various ML algorithms optimized for NVIDIA GPUs.

The analyzer extracts metadata about the model including algorithm type, hyperparameters,
feature information, and other model-specific details from .cuml files.

Potential improvements:
1. Deeper inspection of model internals based on algorithm-specific formats
2. Support for extracting performance metrics if stored with the model
3. Integration with RAPIDS compatibility database to identify version compatibility
4. Statistical analysis of model parameters for quick quality assessment
5. Support for analyzing ensemble models created with multiple cuML algorithms
6. Extraction of GPU-specific optimizations and requirements
7. Detection of quantization or reduced precision optimizations
8. Feature importance extraction for supported model types
9. Analysis of memory requirements for model inference
10. Detection of custom kernels or extensions used in the model
"""


class CuMLAnalyzer(BaseAnalyzer):
    """
    Analyzer for RAPIDS cuML machine learning model files.

    This analyzer extracts information from models saved in the RAPIDS cuML format (.cuml),
    which are GPU-accelerated machine learning models. It can determine the algorithm type,
    hyperparameters, size, and other model characteristics.

    The analyzer supports various cuML implementations including:
    - Supervised learning models (Random Forests, Linear/Logistic Regression, etc.)
    - Clustering models (K-Means, DBSCAN)
    - Dimensionality reduction (PCA, TSNE, UMAP)
    - Nearest Neighbors models
    - And other RAPIDS cuML algorithm implementations
    """

    # Known cuML algorithm types with their readable names
    ALGORITHM_TYPES = {
        # Supervised Learning
        "RandomForestClassifier": "Random Forest Classifier",
        "RandomForestRegressor": "Random Forest Regressor",
        "LinearRegression": "Linear Regression",
        "LogisticRegression": "Logistic Regression",
        "Ridge": "Ridge Regression",
        "Lasso": "Lasso Regression",
        "ElasticNet": "ElasticNet Regression",
        "CD": "Coordinate Descent",
        "MBSGDClassifier": "Mini-Batch SGD Classifier",
        "MBSGDRegressor": "Mini-Batch SGD Regressor",
        "KalmanFilter": "Kalman Filter",
        "SVC": "Support Vector Classifier",
        "SVR": "Support Vector Regressor",

        # Clustering
        "KMeans": "K-Means Clustering",
        "DBSCAN": "DBSCAN Clustering",
        "AgglomerativeClustering": "Agglomerative Clustering",
        "HDBSCAN": "HDBSCAN Clustering",

        # Dimensionality Reduction
        "PCA": "Principal Component Analysis",
        "IncrementalPCA": "Incremental PCA",
        "TruncatedSVD": "Truncated SVD",
        "TSNE": "t-SNE",
        "UMAP": "UMAP",

        # Neighbors
        "NearestNeighbors": "Nearest Neighbors",
        "KNeighborsClassifier": "K-Neighbors Classifier",
        "KNeighborsRegressor": "K-Neighbors Regressor",

        # Manifold Learning
        "TSNE": "t-SNE",
        "MDS": "Multidimensional Scaling",

        # Time Series
        "ARIMA": "ARIMA Time Series Model",

        # Other
        "SummarizerXGBoost": "XGBoost Summarizer",
        "FIL": "Forest Inference Library"
    }

    # Common model categories for grouping
    MODEL_CATEGORIES = {
        "classifier": ["RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier",
                       "MBSGDClassifier", "SVC"],
        "regressor": ["RandomForestRegressor", "LinearRegression", "Ridge", "Lasso",
                      "ElasticNet", "KNeighborsRegressor", "MBSGDRegressor", "SVR"],
        "clustering": ["KMeans", "DBSCAN", "AgglomerativeClustering", "HDBSCAN"],
        "dimensionality_reduction": ["PCA", "IncrementalPCA", "TruncatedSVD", "TSNE", "UMAP"],
    }

    def __init__(self):
        """Initialize the RAPIDS cuML analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.cuml'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a RAPIDS cuML model file to determine its model type and metadata.

        Args:
            file_path: Path to the cuML model file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid cuML model file
            Exception: For other issues during analysis
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Parse the cuML file and extract metadata
            metadata = self._parse_cuml_model(file_path)

            # Determine model type from metadata
            model_type, confidence = self._determine_model_type(metadata)

            return model_type, confidence, metadata

        except Exception as e:
            self.logger.error(f"Error analyzing cuML model file {file_path}: {e}")
            raise

    def _parse_cuml_model(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a RAPIDS cuML model file to extract metadata.

        Args:
            file_path: Path to the cuML model file

        Returns:
            Dictionary containing extracted metadata

        Raises:
            ValueError: If the file is not a valid cuML model file
        """
        metadata = {
            "format": "rapids_cuml",
            "file_path": file_path,
            "file_size_bytes": Path(file_path).stat().st_size
        }

        try:
            # Try to use cuML to load the model if available
            import cuml
            from cuml.common.import_utils import has_rapids

            if not has_rapids():
                raise ImportError("RAPIDS is not installed or unavailable")

            # Try to load the model using cuML's serialization utilities
            model = cuml.common.serialization.load(file_path)

            # Extract basic model information
            metadata["model_class"] = model.__class__.__name__

            # Extract model parameters
            params = {}
            for key, value in model.get_params().items():
                # Convert non-serializable objects to strings
                if hasattr(value, '__class__'):
                    params[key] = f"{value.__class__.__name__}: {str(value)}"
                else:
                    params[key] = value

            metadata["parameters"] = params

            # Extract feature information if available
            if hasattr(model, 'n_features_in_'):
                metadata["n_features"] = model.n_features_in_

            # Extract additional attributes based on model type
            if hasattr(model, 'n_classes_'):
                metadata["n_classes"] = model.n_classes_

            if hasattr(model, 'classes_'):
                try:
                    metadata["classes"] = model.classes_.tolist()
                except:
                    metadata["classes"] = str(model.classes_)

            # Extract model-specific attributes
            self._extract_model_specific_metadata(model, metadata)

            # Compute model size
            metadata["size_mb"] = metadata["file_size_bytes"] / (1024 * 1024)

        except ImportError:
            # If cuML is not available, use lower-level file inspection
            self.logger.warning("RAPIDS cuML library not available, using limited file analysis")
            metadata["limited_analysis"] = True

            # Basic file inspection
            try:
                # cuML models are typically saved as JSON or binary format
                with open(file_path, 'rb') as f:
                    header = f.read(20)  # Read first 20 bytes

                    # Check if it's a JSON file
                    if header.startswith(b'{'):
                        f.seek(0)
                        try:
                            json_content = json.load(f)
                            metadata["format_details"] = "JSON-based cuML model"

                            # Try to extract model type from JSON
                            if "type" in json_content:
                                metadata["model_class"] = json_content["type"]
                            elif "model_type" in json_content:
                                metadata["model_class"] = json_content["model_type"]

                            # Extract any parameters found in the JSON
                            if "params" in json_content:
                                metadata["parameters"] = json_content["params"]

                        except json.JSONDecodeError:
                            metadata["format_details"] = "Binary cuML model (with JSON-like header)"
                    else:
                        metadata["format_details"] = "Binary cuML model"

                        # Try to find model type in binary data
                        f.seek(0)
                        file_content = f.read()

                        # Search for algorithm names in the binary content
                        for algo_name in self.ALGORITHM_TYPES.keys():
                            if algo_name.encode() in file_content:
                                metadata["detected_algorithm"] = algo_name
                                break

            except Exception as e:
                self.logger.error(f"Error during limited file inspection: {e}")
                metadata["inspection_error"] = str(e)

        return metadata

    def _extract_model_specific_metadata(self, model: Any, metadata: Dict[str, Any]) -> None:
        """
        Extract model-specific metadata based on the algorithm type.

        Args:
            model: The loaded cuML model object
            metadata: Dictionary to update with extracted metadata
        """
        model_class = model.__class__.__name__

        # Random Forest specific
        if "RandomForest" in model_class:
            if hasattr(model, 'n_estimators'):
                metadata["n_estimators"] = model.n_estimators
            if hasattr(model, 'max_depth'):
                metadata["max_depth"] = model.max_depth
            if hasattr(model, 'max_features'):
                metadata["max_features"] = model.max_features

        # Linear/Logistic Regression specific
        elif "Regression" in model_class:
            if hasattr(model, 'coef_'):
                try:
                    metadata["coef_shape"] = model.coef_.shape
                except:
                    pass
            if hasattr(model, 'intercept_'):
                try:
                    if hasattr(model.intercept_, 'shape'):
                        metadata["intercept_shape"] = model.intercept_.shape
                except:
                    pass

        # K-Means specific
        elif model_class == "KMeans":
            if hasattr(model, 'n_clusters'):
                metadata["n_clusters"] = model.n_clusters
            if hasattr(model, 'cluster_centers_'):
                try:
                    metadata["cluster_centers_shape"] = model.cluster_centers_.shape
                except:
                    pass

        # DBSCAN specific
        elif model_class == "DBSCAN":
            if hasattr(model, 'eps'):
                metadata["eps"] = model.eps
            if hasattr(model, 'min_samples'):
                metadata["min_samples"] = model.min_samples

        # PCA specific
        elif model_class == "PCA" or model_class == "IncrementalPCA":
            if hasattr(model, 'n_components'):
                metadata["n_components"] = model.n_components
            if hasattr(model, 'explained_variance_ratio_'):
                try:
                    metadata["explained_variance_ratio_shape"] = model.explained_variance_ratio_.shape
                except:
                    pass

        # UMAP specific
        elif model_class == "UMAP":
            if hasattr(model, 'n_neighbors'):
                metadata["n_neighbors"] = model.n_neighbors
            if hasattr(model, 'min_dist'):
                metadata["min_dist"] = model.min_dist

        # Nearest Neighbors specific
        elif "Neighbors" in model_class:
            if hasattr(model, 'n_neighbors'):
                metadata["n_neighbors"] = model.n_neighbors
            if hasattr(model, 'algorithm'):
                metadata["algorithm"] = model.algorithm

        # SVM specific
        elif model_class in ["SVC", "SVR"]:
            if hasattr(model, 'kernel'):
                metadata["kernel"] = model.kernel
            if hasattr(model, 'C'):
                metadata["C"] = model.C

        # Add GPU-specific information if available
        try:
            import cupy
            metadata["gpu_info"] = {
                "cuda_version": cupy.cuda.runtime.runtimeGetVersion(),
                "device_count": cupy.cuda.runtime.getDeviceCount(),
            }
        except:
            pass

        # Add memory usage estimation if possible
        try:
            import sys
            metadata["estimated_memory_size_bytes"] = sys.getsizeof(model)
        except:
            pass

    def _determine_model_type(self, metadata: Dict[str, Any]) -> Tuple[str, ModelConfidence]:
        """
        Determine model type and confidence from metadata.

        Args:
            metadata: Extracted metadata from model file

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check if we have direct model class information
        if "model_class" in metadata:
            model_class = metadata["model_class"]

            # Check if it's a known algorithm type
            if model_class in self.ALGORITHM_TYPES:
                readable_name = self.ALGORITHM_TYPES[model_class]

                # Try to enhance the model type with additional details
                if model_class == "RandomForestClassifier" and "n_classes" in metadata:
                    return f"{readable_name} ({metadata['n_classes']} classes)", ModelConfidence.HIGH

                if model_class == "KMeans" and "n_clusters" in metadata:
                    return f"{readable_name} ({metadata['n_clusters']} clusters)", ModelConfidence.HIGH

                if "n_features" in metadata:
                    return f"{readable_name} ({metadata['n_features']} features)", ModelConfidence.HIGH

                # Default case for known algorithm
                return readable_name, ModelConfidence.HIGH

        # Fall back to detected algorithm from binary inspection
        if "detected_algorithm" in metadata:
            algo_name = metadata["detected_algorithm"]
            if algo_name in self.ALGORITHM_TYPES:
                return self.ALGORITHM_TYPES[algo_name], ModelConfidence.MEDIUM

        # If we have parameters but no clear algorithm identifier
        if "parameters" in metadata and metadata["parameters"]:
            # Make an educated guess based on parameters
            params = metadata["parameters"]

            if "n_estimators" in params and "max_depth" in params:
                return "Random Forest Model", ModelConfidence.MEDIUM

            if "C" in params and "kernel" in params:
                return "Support Vector Machine", ModelConfidence.MEDIUM

            if "n_clusters" in params:
                return "Clustering Model", ModelConfidence.MEDIUM

            if "n_components" in params:
                return "Dimensionality Reduction", ModelConfidence.MEDIUM

            # Generic ML model if we have some parameters
            return "RAPIDS cuML Model", ModelConfidence.LOW

        # If we have format details from limited analysis
        if "format_details" in metadata:
            return "RAPIDS cuML Model", ModelConfidence.LOW

        # If nothing else works
        return "Unknown Model", ModelConfidence.UNKNOWN

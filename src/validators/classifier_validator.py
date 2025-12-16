from typing import Any
import numpy as np
import pandas as pd

class ClassifierValidator:
    # --- Small check functions ---
    @staticmethod
    def check_not_empty(X: Any, y: Any) -> None:
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("X or y data is invalid (None or empty).")

    @staticmethod
    def check_length_match(X: Any, y: Any) -> None:
        if len(X) != len(y):
            raise ValueError("Number of samples in X and y do not match.")

    @staticmethod
    def check_shape(X: Any) -> None:
        if hasattr(X, "shape") and len(X.shape) != 2:
            raise ValueError("X must have shape (n_samples, n_features).")

    @staticmethod
    def check_model_exists(model: Any) -> None:
        if model is None:
            raise ValueError("Model has not been trained or loaded.")

    @staticmethod
    def check_features_match(model: Any, X: Any) -> None:
        if hasattr(X, "shape") and X.shape[1] != getattr(model, "n_features_", X.shape[1]):
            raise ValueError("Number of features in X does not match the model.")

    @staticmethod
    def validate_train_input(X: Any, y: Any) -> None:
        ClassifierValidator.check_not_empty(X, y)
        ClassifierValidator.check_length_match(X, y)
        ClassifierValidator.check_shape(X)

    @staticmethod
    def validate_predict_input(model: Any, X: Any) -> None:
        ClassifierValidator.check_model_exists(model)
        if X is None or len(X) == 0:
            raise ValueError("X data is invalid for prediction.")
        ClassifierValidator.check_features_match(model, X)

    @staticmethod
    def validate_evaluate_input(model: Any, X: Any, y: Any) -> None:
        ClassifierValidator.check_model_exists(model)
        ClassifierValidator.check_not_empty(X, y)
        ClassifierValidator.check_length_match(X, y)

    @staticmethod
    def validate_sample_weight(sample_weight, n_samples: int):
        if sample_weight is None:
            return
        if not isinstance(sample_weight, (np.ndarray, list)):
            raise ValueError("sample_weight must be a numpy array or list.")
        if len(sample_weight) != n_samples:
            raise ValueError(
                f"sample_weight length must be {n_samples}, but got {len(sample_weight)}."
            )
        if np.any(np.array(sample_weight) < 0):
            raise ValueError("sample_weight must not contain negative values.")
        if np.any(np.isnan(sample_weight)):
            raise ValueError("sample_weight must not contain NaN values.")

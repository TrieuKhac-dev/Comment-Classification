import numpy as np
from typing import Any, Optional, List, Dict
from sklearn.linear_model import LogisticRegression

from src.interfaces.classifier import IClassifier
from src.interfaces.repositories import IClassifierRepository
from src.validators import ClassifierValidator


class LogisticRegressionClassifier(IClassifier):
    """
    Classifier dùng Logistic Regression (multinomial), tương thích với pipeline hiện tại.
    """

    def __init__(self):
        self.model: Optional[LogisticRegression] = None

    def train(
        self,
        X: Any,
        y: Any,
        config: Optional[Dict] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        ClassifierValidator.validate_train_input(X, y)
        ClassifierValidator.validate_sample_weight(sample_weight, len(y))

        config = config or {}
        # Sử dụng multi_class="auto" để tự xử lý binary / multi-class
        config = {"multi_class": "auto", **config}

        self.model = LogisticRegression(**config)
        self.model.fit(X, y.ravel(), sample_weight=sample_weight)

    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = False,
    ) -> Optional[np.ndarray]:
        ClassifierValidator.validate_predict_input(self.model, X)
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        if return_proba:
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None,
        average: str = "weighted",
    ) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        if self.model is None:
            raise ValueError("Model does not exist.")

        ClassifierValidator.validate_evaluate_input(self.model, X, y)

        y_pred = self.model.predict(X)
        metric_funcs = {
            "accuracy": lambda yt, yp: accuracy_score(yt, yp),
            "f1": lambda yt, yp: f1_score(yt, yp, average=average),
            "precision": lambda yt, yp: precision_score(yt, yp, average=average),
            "recall": lambda yt, yp: recall_score(yt, yp, average=average),
        }

        metrics = metrics or ["accuracy"]
        results: Dict[str, Any] = {}

        for m in metrics:
            if m in metric_funcs:
                results[m] = metric_funcs[m](y, y_pred)

        return results

    def save(
        self,
        repository: IClassifierRepository,
        model_path: str,
        save_type: str = "joblib",
    ) -> None:
        if repository is None or model_path is None:
            raise ValueError("Missing repository or model path when saving.")
        if self.model is None:
            raise ValueError("No model to save.")
        repository.save(self.model, model_path, save_type)

    def load(
        self,
        repository: IClassifierRepository,
        model_path: str,
        load_type: str = "joblib",
    ) -> None:
        if repository is None or model_path is None:
            raise ValueError("Missing repository or model path when loading.")

        self.model = repository.load(model_path, load_type)
        if self.model is None:
            raise ValueError("Failed to load model.")


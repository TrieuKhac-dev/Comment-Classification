import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict

from src.interfaces.repositories.i_classifier_repository import IClassifierRepository

class IClassifier(ABC):
    @abstractmethod
    def train(self, X: Any, y: Any, config: Optional[Dict] = None, sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Huấn luyện mô hình với dữ liệu đầu vào đã được xử lý.
        """
        pass

    @abstractmethod
    def predict(self, X: Any, return_proba: bool = False) -> Optional[Any]:
        """
        Dự đoán nhãn hoặc xác suất cho dữ liệu đầu vào đã được xử lý.
        """
        pass

    @abstractmethod
    def evaluate(self, X: Any, y: Any, metrics: Optional[List[str]] = None, average: str = "weighted") -> Dict[str, Any]:
        """
        Đánh giá mô hình với dữ liệu đã được xử lý và trả về các chỉ số theo yêu cầu.
        """
        pass

    @abstractmethod
    def save(self, repository: IClassifierRepository, model_path: str, save_type: str = "joblib") -> None:
        """
        Lưu mô hình vào repository với kiểu lưu tương ứng.
        """
        pass

    @abstractmethod
    def load(self, repository: IClassifierRepository, model_path: str, load_type: str = "joblib") -> None:
        """
        Load mô hình từ repository với kiểu load tương ứng.
        """
        pass
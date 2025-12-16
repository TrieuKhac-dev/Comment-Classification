from abc import ABC, abstractmethod
from typing import Any

class IClassifierRepository(ABC):
    @abstractmethod
    def save(self, model: Any, path: str, save_type: str = "joblib") -> None:
        pass

    @abstractmethod
    def load(self, path: str, load_type: str = "joblib") -> Any:
        pass
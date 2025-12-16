from typing import Protocol, Any
from abc import abstractmethod, ABC

class IClassifierRepoProvider(ABC):
    @abstractmethod
    def save(self, model: Any, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> Any:
        pass
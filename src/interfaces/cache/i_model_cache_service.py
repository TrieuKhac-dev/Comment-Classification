from typing import Any, Optional
from abc import ABC, abstractmethod

class IModelCacheService(ABC):
    @abstractmethod
    def get(self, key: str, copy: bool = False) -> Optional[Any]:
        pass
    @abstractmethod
    def set(self, key: str, model: Any) -> None:
        pass
    @abstractmethod
    def has(self, key: str) -> bool:
        pass
    @abstractmethod
    def clear(self) -> None:
        pass
    @abstractmethod
    def remove(self, key: str) -> None:
        pass
    @abstractmethod
    def get_cached_keys(self) -> list[str]:
        pass
    @abstractmethod
    def cache_size(self) -> int:
        pass
    @staticmethod
    @abstractmethod
    def normalize_path(path: str) -> str:
        pass

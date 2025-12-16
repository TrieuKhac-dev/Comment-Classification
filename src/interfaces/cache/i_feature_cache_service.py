from typing import Any, Optional, List, Dict
from abc import ABC, abstractmethod

class IFeatureCacheService(ABC):
    @abstractmethod
    def exists(self, key: str) -> bool:
        pass
    @abstractmethod
    def load(self, key: str) -> Any:
        pass
    @abstractmethod
    def save(self, key: str, features: Any):
        pass
    @abstractmethod
    def clear(self):
        pass
    @abstractmethod
    def make_cache_key(self, texts: List[str], configs: Optional[Dict[str, Any]] = None) -> str:
        pass

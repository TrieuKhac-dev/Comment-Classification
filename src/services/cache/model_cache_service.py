from typing import Any, Dict, Optional
import copy as _copy
from pathlib import Path

class ModelCacheService:
    
    _instance: Optional['ModelCacheService'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ModelCacheService._initialized:
            return
        
        self._cache: Dict[str, Any] = {}
        ModelCacheService._initialized = True
    
    def get(self, key: str, copy: bool = False) -> Optional[Any]:
        val = self._cache.get(key)
        if val is None:
            return None
        if copy:
            return _copy.deepcopy(val)
        return val
    
    def set(self, key: str, model: Any) -> None:
        self._cache[key] = model
    
    def has(self, key: str) -> bool:
        return key in self._cache
    
    def clear(self) -> None:
        self._cache.clear()
    
    def remove(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
    
    def get_cached_keys(self) -> list[str]:
        """Lấy danh sách các keys đã được cache"""
        return list(self._cache.keys())
    
    def cache_size(self) -> int:
        """Số lượng models đã cache"""
        return len(self._cache)
    
    @staticmethod
    def normalize_path(path: str) -> str:
        return str(Path(path).resolve())


# Global service instance
model_cache_service = ModelCacheService()

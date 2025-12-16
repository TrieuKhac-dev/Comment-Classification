from typing import Any

from .ml_loader_factory import ML_LoaderFactory
from src.interfaces.cache import IModelCacheService

class LoaderService:
    def __init__(self, cache_service: IModelCacheService):
        self.cache_service = cache_service

    def load(self, model_type: str, model_path: str, **kwargs) -> Any:
        cache_key = self.cache_service.normalize_path(model_path)
        # Check cache first
        if self.cache_service.has(cache_key):
            print(f"[ModelLoaderService] Loaded {model_type} from cache: {model_path}")
            return self.cache_service.get(cache_key)
        # Load from disk via child loader
        loader_cls = ML_LoaderFactory.get(model_type, **kwargs)
        loader = loader_cls()
        model = loader.load(model_path)
        # Save to cache
        self.cache_service.set(cache_key, model)
        print(f"[ModelLoaderService] Đã load và cache {model_type} thành công.")
        return model
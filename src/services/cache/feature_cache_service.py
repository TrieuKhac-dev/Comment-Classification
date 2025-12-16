import os
import hashlib
import joblib
from typing import Any, Dict, List, Optional

from src.interfaces.cache import IFeatureCacheService
from config.core.paths import PathConfig

class FeatureCacheService(IFeatureCacheService):
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or PathConfig.FEATURES_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    #feature_{prefix}_{dataset_type}_{key_hash}.pkl
    def _get_cache_path(self, key: str, prefix: Optional[str] = None, dataset_type: Optional[str] = None) -> str:
        # Sinh tên file cache từ key (hash để tránh trùng lặp)
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        if prefix:
            if dataset_type:
                filename = f"feature_{prefix}_{dataset_type}_{key_hash}.pkl"
            else:
                filename = f"feature_{prefix}_{key_hash}.pkl"
        else:
            if dataset_type:
                filename = f"feature_{dataset_type}_{key_hash}.pkl"
            else:
                filename = f"feature_{key_hash}.pkl"
        return os.path.join(self.cache_dir, filename)

    def save(self, key: str, features: Any, prefix: Optional[str] = None, dataset_type: Optional[str] = None):
        path = self._get_cache_path(key, prefix, dataset_type)
        try:
            joblib.dump(features, path)
            print(f"[FeatureCacheService] Feature cache saved successfully: {path}")
        except Exception as e:
            print(f"[FeatureCacheService] Error saving cache at {path}: {e}")

    def load(self, key: str, prefix: Optional[str] = None, dataset_type: Optional[str] = None) -> Any:
        path = self._get_cache_path(key, prefix, dataset_type)
        if os.path.exists(path):
            return joblib.load(path)
        return None

    def exists(self, key: str, prefix: Optional[str] = None, dataset_type: Optional[str] = None) -> bool:
        return os.path.exists(self._get_cache_path(key, prefix, dataset_type))

    def clear(self):
        for fname in os.listdir(self.cache_dir):
            if fname.startswith("feature_") and fname.endswith(".pkl"):
                os.remove(os.path.join(self.cache_dir, fname))

    def make_cache_key(self, texts: List[str], configs: Optional[Dict[str, Any]] = None) -> str:
        import hashlib, json
        raw = json.dumps({
            'texts': texts,
            'configs': configs
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode('utf-8')).hexdigest()

import numpy as np
from typing import List, Dict, Optional
from ...interfaces.cache import IFeatureCacheService
from ...interfaces.machine_learning import IFeatureExtractor

class ExtractorService:
    def __init__(self, extractor: IFeatureExtractor, cache_service: IFeatureCacheService):
        self.extractor = extractor
        self.cache_service = cache_service

    def extract(self, texts: List[str], 
                configs: Optional[Dict[str, dict]] = None, 
                use_cache: bool = True,
                cache_prefix: Optional[str] = None,
                dataset_type: Optional[str] = None):
        if not self.extractor.is_loaded():
            raise Exception("Extractor model is not loaded.")
        
        if not texts:
            raise ValueError("Input texts list is empty.")
        
        if use_cache and self.cache_service is None:
            raise ValueError("Cache service must be provided when use_cache is True.")
        
        key = self.cache_service.make_cache_key(texts, configs)
        if use_cache and self.cache_service.exists(key, prefix=cache_prefix, dataset_type=dataset_type):
            return self.cache_service.load(key, prefix=cache_prefix, dataset_type=dataset_type)

        output = self.extractor.extract(
            texts=texts,
            use_cache=False,
            cache_prefix=None,
            configs=configs
        )
        if not isinstance(output, np.ndarray):
            output = np.array(output)
        if use_cache:
            self.cache_service.save(key, output, prefix=cache_prefix, dataset_type=dataset_type)
        return output

import numpy as np
from typing import List, Optional, Dict
from collections import OrderedDict

from src.interfaces.machine_learning import IFeatureExtractor, ModelNotLoadedError
from src.validators import CompositeExtractorValidator


class CompositeExtractor(IFeatureExtractor):

    def __init__(
        self,
        extractors: Dict[str, IFeatureExtractor],
        weights: Optional[Dict[str, float]] = None,
        combine_mode: str = 'concat'
    ):
        # Kiểm tra và chuẩn hóa extractors thành OrderedDict[name->extractor]
        CompositeExtractorValidator.validate_extractors(extractors)
        self.extractors = OrderedDict(extractors)

        # Kiểm tra và chuẩn hóa weights thành dict mapping name->float
        CompositeExtractorValidator.validate_weights(self.extractors, weights)
        self.weights_map = weights or {}

        self.combine_mode = combine_mode

    def extract(self, texts: List[str], use_cache: bool, cache_prefix: Optional[str] = None, configs: Optional[Dict[str, dict]] = None) -> np.ndarray:
        # Kiểm tra xem tất cả extractor đã load chưa
        if not self.is_loaded():
            raise ModelNotLoadedError("One or more extractors are not loaded")

        features_list = self._collect_features(texts, use_cache, cache_prefix, configs)
        CompositeExtractorValidator.validate_row_counts(features_list)
        return self._combine_features(features_list)

    def _collect_features(self, texts: List[str], use_cache: bool, cache_prefix: Optional[str], configs: Optional[Dict[str, dict]]) -> list:
        """Thu thập và xử lý feature từ từng extractor."""
        features_list = []
        for name, ext in self.extractors.items():
            ext_kwargs = configs.get(name, {}) if configs else {}
            
            # Không truyền cache_prefix vào extractors con
            # vì caching được xử lý ở layer cao hơn (ExtractorService)
            
            try:
                features = ext.extract(texts, **ext_kwargs)
            except TypeError as e:
                raise ValueError(
                    f"Extractor '{name}' does not accept some of the provided parameters: {list(ext_kwargs.keys())}. "
                    f"Original error: {e}"
                )
            
            weight = self.weights_map.get(name, 1.0) if self.weights_map is not None else 1.0
            try:
                features = features * float(weight)
            except Exception:
                raise ValueError("Weight must be convertible to float")
            features_list.append(features)
        return features_list

    def _combine_features(self, features_list: list) -> np.ndarray:
        """Kết hợp các feature lại theo combine_mode."""
        if self.combine_mode == 'concat':
            # Ghép vector theo chiều ngang (hstack)
            return np.hstack(features_list)
        
        if self.combine_mode == 'mean':
            # Kiểm tra dimension trước khi tính trung bình
            CompositeExtractorValidator.validate_dimensions(self.extractors, self.combine_mode)
            return np.mean(np.stack(features_list, axis=2), axis=2)
        raise ValueError(f"Unsupported combine mode: {self.combine_mode}")

    def get_dimension(self) -> int:
        # Kiểm tra extractor đã load chưa
        if not self.is_loaded():
            raise ModelNotLoadedError("Extractors not loaded")
        if self.combine_mode == 'concat':
            # Trả về tổng số chiều của tất cả extractor
            return sum(ext.get_dimension() for ext in self.extractors.values())
        elif self.combine_mode == 'mean':
            # Trả về số chiều trung bình (yêu cầu tất cả extractor có cùng dimension)
            dims = [ext.get_dimension() for ext in self.extractors.values()]
            if len(set(dims)) != 1:
                raise ValueError("For 'mean' mode, all extractors must have the same dimension")
            return dims[0]
        else:
            raise ValueError(f"Unsupported combine mode: {self.combine_mode}")

    def is_loaded(self) -> bool:
        # Kiểm tra tất cả extractor đã load chưa
        return all(ext is not None and ext.is_loaded() for ext in self.extractors.values())

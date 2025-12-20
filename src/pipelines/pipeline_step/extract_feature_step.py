import numpy as np
from typing import Optional, Dict, List

from src.interfaces.pipeline import IPipelineStep
from src.models.classifier.base_classifier_context import BaseClassifierContext

class ExtractFeatureStep(IPipelineStep):
    def __init__(
        self,
        configs: Optional[Dict[str, dict]] = None,
        use_cache: bool = True,
        extract_pred: bool = True,
        extract_train: bool = True,
        extract_test: bool = True,
        extract_val: bool = True,
    ):
        self.configs = configs or {}
        self.use_cache = use_cache
        self.extract_pred = extract_pred
        self.extract_train = extract_train
        self.extract_test = extract_test
        self.extract_val = extract_val

    def _extract_and_assign(self, texts: Optional[List[str]], assign_attr: str, context: BaseClassifierContext):
        """Helper để extract features từ texts và gán vào context"""
        if texts is None:
            return
        
        # Nếu texts empty, set empty array và skip extract
        if not texts:
            setattr(context, assign_attr, np.array([]))
            return
        
        # Lấy prefix từ context (nếu có)
        cache_prefix = getattr(context, 'feature_cache_prefix', None)
        
        # Xác định dataset type từ assign_attr (X_train_features -> train)
        dataset_type = None
        if 'train' in assign_attr:
            dataset_type = 'train'
        elif 'test' in assign_attr:
            dataset_type = 'test'
        elif 'val' in assign_attr:
            dataset_type = 'val'
        elif 'pred' in assign_attr:
            dataset_type = 'pred'
        
        # Thử extract với cache_prefix và dataset_type
        try:
            features = context.extractor_service.extract(
                texts=texts,
                use_cache=self.use_cache,
                configs=self.configs,
                cache_prefix=cache_prefix,
                dataset_type=dataset_type
            )
        except TypeError as e:
            # Nếu extractor không hỗ trợ các tham số mới, thử lại không có chúng
            if 'cache_prefix' in str(e) or 'dataset_type' in str(e):
                features = context.extractor_service.extract(
                    texts=texts,
                    use_cache=self.use_cache,
                    configs=self.configs
                )
            else:
                raise
        
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        setattr(context, assign_attr, features)

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if context.extractor_service is None:
            raise ValueError("ExtractorService is not provided")
        
        if self.extract_pred:
            self._extract_and_assign(context.X_pred_processed, "X_pred_features", context)

        if self.extract_train:
            self._extract_and_assign(context.X_train_processed, "X_train_features", context)

        if self.extract_test:
            self._extract_and_assign(context.X_test_processed, "X_test_features", context)

        if self.extract_val:
            self._extract_and_assign(context.X_val_processed, "X_val_features", context)

        if context.logger_service:
            context.logger_service.info(
                f"ExtractFeatureStep | Extracted features "
                f"predict={context.X_pred_features.shape if context.X_pred_features is not None else None}, "
                f"train={context.X_train_features.shape if context.X_train_features is not None else None}, "
                f"test={context.X_test_features.shape if context.X_test_features is not None else None}, "
                f"val={context.X_val_features.shape if context.X_val_features is not None else None}"
            )

        return context

from typing import Dict, Type

from src.interfaces.machine_learning import IFeatureExtractor
from src.utils.factory import GenericFactory

class FeatureExtractorFactory(GenericFactory[IFeatureExtractor]):
    @classmethod
    def create(cls, extractor_type: str, model, **kwargs) -> IFeatureExtractor:
        extractor_cls = cls.get(extractor_type)
        return extractor_cls(model=model, **kwargs)

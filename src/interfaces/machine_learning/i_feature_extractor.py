from abc import ABC, abstractmethod
import numpy as np


class ModelNotLoadedError(Exception):
    pass


class IFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, texts: list[str], usecache: bool, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        pass
    
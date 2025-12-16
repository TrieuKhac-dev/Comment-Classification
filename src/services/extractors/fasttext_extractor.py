import numpy as np
from fasttext import FastText
from typing import List, Optional

from src.interfaces.machine_learning import IFeatureExtractor, ModelNotLoadedError


class FastTextExtractor(IFeatureExtractor):
    def __init__(self, model: Optional[FastText._FastText] = None):
        self._model = model
    
    def set_model(self, model: FastText._FastText) -> None:
        self._model = model
    
    def extract(self, texts: List[str], pooling: str = 'mean') -> np.ndarray:
        if not self.is_loaded():
            raise ModelNotLoadedError(
                "FastText model not loaded. Please load model first using FastTextLoader."
            )
        
        features = []
        for text in texts:
            words = text.split()
            if not words:
                features.append(np.zeros(self._model.get_dimension()))
                continue
            
            word_vectors = [self._model.get_word_vector(word) for word in words]
            word_vectors = np.array(word_vectors)
            
            if pooling == 'mean':
                features.append(np.mean(word_vectors, axis=0))
            elif pooling == 'max':
                features.append(np.max(word_vectors, axis=0))
            elif pooling == 'sum':
                features.append(np.sum(word_vectors, axis=0))
            else:
                features.append(np.mean(word_vectors, axis=0))
        
        return np.array(features)
    
    def get_dimension(self) -> int:
        if not self.is_loaded():
            raise ModelNotLoadedError("Model not loaded")
        return self._model.get_dimension()
    
    def is_loaded(self) -> bool:
        return self._model is not None

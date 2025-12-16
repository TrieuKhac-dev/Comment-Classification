import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional

from src.interfaces.machine_learning import IFeatureExtractor, ModelNotLoadedError


class SBERTExtractor(IFeatureExtractor):
    def __init__(self, model: Optional[SentenceTransformer] = None):
        self._model = model
    
    def set_model(self, model: SentenceTransformer) -> None:
        self._model = model
    
    def extract(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False
    ) -> np.ndarray:
        if not self.is_loaded():
            raise ModelNotLoadedError(
                "SBERT model not loaded. Please load model first using SBERTLoader."
            )
        
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        if not self.is_loaded():
            raise ModelNotLoadedError("Model not loaded")
        return self._model.get_sentence_embedding_dimension()
    
    def is_loaded(self) -> bool:
        return self._model is not None

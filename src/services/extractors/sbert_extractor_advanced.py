import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional

from src.interfaces.machine_learning import IFeatureExtractor, ModelNotLoadedError


class SBERTExtractorAdvanced(IFeatureExtractor):
    def __init__(self, model: Optional[SentenceTransformer] = None):
        self._model = model
        self._pool = None  # Multi-process pool
    
    def set_model(self, model: SentenceTransformer) -> None:
        self._model = model
        # Cleanup old pool if exists
        if self._pool is not None:
            self._stop_pool()
    
    def extract(
        self,
        texts: List[str],
        batch_size: int = 200,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        num_workers: int = 4,
        chunk_size: int = 2000,
        **kwargs
    ) -> np.ndarray:
        """
        Extract features với hỗ trợ multi-processing
        
        Args:
            texts: Danh sách văn bản
            batch_size: Kích thước batch
            show_progress_bar: Hiển thị thanh tiến trình
            normalize_embeddings: Chuẩn hóa embeddings về unit length
            convert_to_numpy: Chuyển sang numpy array
            num_workers: Số workers cho multi-processing (0 = single process)
            chunk_size: Số câu mỗi chunk khi dùng pool
            **kwargs: Tham số bổ sung cho SBERT encode:
                - convert_to_tensor: bool = False
                - output_value: str = 'sentence_embedding'
                - precision: str = 'float32'
                - device: str = None (auto-detect)
        
        Returns:
            np.ndarray: Feature embeddings với shape (len(texts), dimension)
        """
        if not self.is_loaded():
            raise ModelNotLoadedError(
                "SBERT model not loaded. Please load model first using SBERTLoader."
            )
        
        # Multi-processing mode: cần cả num_workers và chunk_size hợp lệ
        if self.validate_num_workers(num_workers) and self.validate_chunk_size(chunk_size):
            return self._extract_with_pool(
                texts=texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=convert_to_numpy,
                num_workers=num_workers,
                chunk_size=chunk_size,
                **kwargs
            )
        else:
            # Single-process mode
            if num_workers > 0 and not self.validate_chunk_size(chunk_size):
                self._warn_chunk_size()
            
            return self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=convert_to_numpy,
                **kwargs
            )
    
    def _extract_with_pool(
        self,
        texts: List[str],
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        num_workers: int,
        chunk_size: int,
        **kwargs
    ) -> np.ndarray:
        # Tạo pool với CPU devices
        devices = ["cpu"] * num_workers
        pool = self._model.start_multi_process_pool(devices)
        
        try:
            result = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=convert_to_numpy,
                pool=pool,
                chunk_size=chunk_size,
                **kwargs
            )
            
            return result
            
        finally:
            # Đảm bảo cleanup pool sau khi xong
            self._model.stop_multi_process_pool(pool)
    
    def similarity_vectors(
        self, 
        v1: np.ndarray, 
        v2: np.ndarray, 
        normalize: bool = True
    ) -> float:
        if normalize:
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1 / norm1, v2 / norm2))
        
        # Không normalize: tính theo công thức chuẩn
        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denominator == 0:
            return 0.0
        return float(np.dot(v1, v2) / denominator)
    
    def similarity_texts(
        self, 
        text1: str, 
        text2: str, 
        normalize: bool = True
    ) -> float:
        if not self.is_loaded():
            raise ModelNotLoadedError("Model not loaded")
        
        # Encode cả hai texts cùng lúc
        vecs = self._model.encode(
            [text1, text2],
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True
        )
        
        return self.similarity_vectors(vecs[0], vecs[1], normalize=normalize)
    
    def get_dimension(self) -> int:
        if not self.is_loaded():
            raise ModelNotLoadedError("Model not loaded")
        
        try:
            return self._model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback: encode một text dummy để suy ra dimension
            vec = self._model.encode(
                ["dummy"],
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return int(vec.shape[1])
    
    def is_loaded(self) -> bool:
        """Kiểm tra model đã được load chưa"""
        return self._model is not None
    
    def validate_num_workers(self, num_workers: int) -> bool:
        return isinstance(num_workers, int) and num_workers > 0
    
    def validate_chunk_size(self, chunk_size: int) -> bool:
        return isinstance(chunk_size, int) and chunk_size > 0
    
    def _warn_chunk_size(self):
        print(
            "[SBERT] CẢNH BÁO: Khi dùng pool (num_workers > 0), "
            "bạn nên truyền chunk_size > 0 để tối ưu hiệu năng và tránh lỗi bộ nhớ!"
        )
    
    def _stop_pool(self):
        if self._pool is not None:
            try:
                self._model.stop_multi_process_pool(self._pool)
            except:
                pass
            finally:
                self._pool = None
    
    def __del__(self):
        self._stop_pool()

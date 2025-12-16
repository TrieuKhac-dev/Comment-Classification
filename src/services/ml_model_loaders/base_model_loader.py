from abc import abstractmethod
from pathlib import Path
from typing import Any

from src.interfaces.machine_learning import IModelLoader, ModelNotFoundError

class BaseModelLoader(IModelLoader):
    def load(self, model_path: str) -> Any:
        #Kiểm tra model đã tải
        if not self.is_model_exists(model_path):
            raise ModelNotFoundError(
                f"{self._get_model_type()} model not found at: {model_path}\n"
                f"{self._get_download_instruction()}"
            )
        
        #Load từ disk
        try:
            print(f"Loading {self._get_model_type()} model from disk: {Path(model_path).name}...")
            model = self._load_from_disk(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load {self._get_model_type()} model: {e}")
    
    @abstractmethod
    def _load_from_disk(self, model_path: str) -> Any:
        """
        Load model từ disk (subclass implement)
        
        Args:
            model_path: Đường dẫn tới model
        
        Returns:
            Model instance
        
        Raises:
            Exception: Nếu load thất bại
        """
        pass
    
    @abstractmethod
    def _get_model_type(self) -> str:
        """
        Trả về tên loại model (cho logging)
        
        Returns:
            Model type name (e.g., "FastText", "SBERT")
        """
        pass
    
    @abstractmethod
    def _get_download_instruction(self) -> str:
        """
        Trả về hướng dẫn download model
        
        Returns:
            Download instruction string
        """
        pass

    @abstractmethod
    def is_model_exists(self, model_path: str) -> bool:
        pass
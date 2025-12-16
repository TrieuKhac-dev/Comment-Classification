from abc import ABC, abstractmethod
from typing import Any


class ModelNotFoundError(Exception):
    pass


class IModelLoader(ABC):
    @abstractmethod
    def load(self, model_path: str) -> Any:
        """
        Load model từ đường dẫn local.
        
        Args:
            model_path: Đường dẫn đến model trên local
            
        Returns:
            Any: Model object đã được load
            
        Raises:
            ModelNotFoundError: Nếu model không tồn tại tại đường dẫn
            RuntimeError: Nếu có lỗi khi load model
        """
        pass
    
    @abstractmethod
    def is_model_exists(self, model_path: str) -> bool:
        """
        Kiểm tra model có tồn tại tại đường dẫn hay không.
        
        Args:
            model_path: Đường dẫn đến model
            
        Returns:
            bool: True nếu model tồn tại và hợp lệ
        """
        pass

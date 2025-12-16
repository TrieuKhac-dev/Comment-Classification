from abc import ABC, abstractmethod


class IPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """
        Tiền xử lý một văn bản đơn lẻ.
        
        Args:
            text: Văn bản cần xử lý
            
        Returns:
            str: Văn bản đã được tiền xử lý
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """
        Tiền xử lý một danh sách các văn bản.
        
        Args:
            texts: Danh sách các văn bản cần xử lý
            
        Returns:
            list[str]: Danh sách các văn bản đã được tiền xử lý
        """
        pass
    
    @abstractmethod
    def get_config_summary(self) -> dict:
        """
        Lấy thông tin tóm tắt về cấu hình preprocessing hiện tại.
        
        Returns:
            dict: Dictionary chứa các config đang được sử dụng
        """
        pass

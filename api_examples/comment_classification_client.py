import requests
from typing import Dict, List


class CommentClassificationClient:
    """Client để tương tác với Comment Classification API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Khởi tạo client
        
        Args:
            base_url: URL của API server (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> Dict:
        """
        Kiểm tra trạng thái server
        
        Returns:
            Dict chứa thông tin health check
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict(self, comments: Dict[str, str]) -> Dict:
        """
        Phân loại bình luận (endpoint chính)
        
        Args:
            comments: Dictionary {id: comment_text}
            
        Returns:
            Dict chứa kết quả phân loại
            
        Example:
            >>> client = CommentClassificationClient()
            >>> result = client.predict({
            ...     "1": "Sản phẩm tốt",
            ...     "2": "Đồ rác"
            ... })
            >>> print(result)
        """
        response = requests.post(
            f"{self.base_url}/predict",
            json={"comments": comments},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def predict_simple(self, comments: List[str]) -> Dict:
        """
        Phân loại bình luận (endpoint đơn giản)
        
        Args:
            comments: List các comment text
            
        Returns:
            Dict chứa predictions
        """
        response = requests.post(
            f"{self.base_url}/predict/simple",
            json=comments,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
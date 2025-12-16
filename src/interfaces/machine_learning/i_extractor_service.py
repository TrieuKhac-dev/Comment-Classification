from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class IExtractorService(ABC):
    @abstractmethod
    def extract(
        self,
        texts: List[str],
        configs: Optional[Dict[str, dict]] = None,
        use_cache: bool = True
    ) -> Any:
        """
        Trích xuất đặc trưng từ danh sách văn bản đầu vào, có thể sử dụng cache.
        """
        pass
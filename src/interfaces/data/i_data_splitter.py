from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np


class IDataSplitterService(ABC):
    @abstractmethod
    def split(
        self,
        texts: list[str],
        labels: Union[pd.DataFrame, np.ndarray],
        test_ratio: float = 0.2,
        val_ratio: float = 0.0,
        random_seed: int = 42,
    ) -> dict:
        """
        Tách dữ liệu thành các tập train/test/val.
        
        Args:
            texts: Danh sách các văn bản
            labels: Nhãn dưới dạng DataFrame hoặc numpy array
            test_ratio: Tỷ lệ tập test (mặc định: 0.2)
            val_ratio: Tỷ lệ tập validation (mặc định: 0.0)
            random_seed: Seed ngẫu nhiên để tái tạo kết quả (mặc định: 42)
            
        Returns:
            dict: Dictionary với keys 'train', 'test', và 'val' (nếu có),
                  mỗi key chứa 'texts' (list) và 'labels' (np.ndarray)
        """
        pass

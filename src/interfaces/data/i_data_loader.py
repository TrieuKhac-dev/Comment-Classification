from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
import pandas as pd


class IDataLoaderService(ABC):
    @abstractmethod
    def load_csv(
        self,
        filepath: Union[str, Path],
        text_column: str,
        label_columns: list[str]
    ) -> pd.DataFrame:
        """
        Đọc file CSV và validate dữ liệu.
        
        Args:
            filepath: Đường dẫn đến file CSV
            text_column: Tên cột chứa văn bản
            label_columns: Danh sách tên các cột nhãn
            
        Returns:
            pd.DataFrame: DataFrame đã được làm sạch và validate
            
        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu validation thất bại (thiếu cột, có NA, nhãn không binary, tham số rỗng)
        """
        pass
    
    @abstractmethod
    def get_text_and_labels(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_columns: list[str]
    ) -> tuple[list[str], pd.DataFrame]:
        """
        Trích xuất texts và labels từ DataFrame.
        
        Args:
            df: DataFrame đầu vào
            text_column: Tên cột chứa văn bản
            label_columns: Danh sách tên các cột nhãn
            
        Returns:
            tuple: (danh sách texts, DataFrame chứa labels)
            
        Raises:
            ValueError: Nếu thiếu các cột bắt buộc
        """
        pass

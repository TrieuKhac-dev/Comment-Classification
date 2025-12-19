"""
Pipeline step để load processed data từ cache.
Thay thế LoadDataStep + PreprocessTextStep khi có sẵn processed data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.interfaces.pipeline import IPipelineStep
from src.models.classifier.base_classifier_context import BaseClassifierContext
from src.services.cache.processed_data_cache_service import ProcessedDataCacheService
from config.training.data import data_config


class LoadProcessedDataStep(IPipelineStep):
    """
    Load dữ liệu đã qua preprocessing từ cache file (CSV UTF-8).
    Đưa vào context như prediction data (X_pred_texts, y_pred).
    """
    
    def __init__(self, cache_file: str):
        """
        Khởi tạo step.
        
        Args:
            cache_file: Đường dẫn đến file processed cache (.csv)
                       Ví dụ: data/processed/processed_lightgbm_20231215_test_abc123.csv
        """
        self.cache_file = Path(cache_file)
        if not self.cache_file.exists():
            raise FileNotFoundError(f"Processed cache file not found: {cache_file}")
        
        self.cache_service = ProcessedDataCacheService()
    
    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        """
        Load processed data từ file và đưa vào context.
        
        Args:
            context: Context để load data vào
        
        Returns:
            Context với X_pred_processed và y_pred đã được load
        """
        try:
            # Load CSV file với UTF-8 encoding
            df = pd.read_csv(self.cache_file, encoding='utf-8')
            
            # Lấy tên cột text và label columns từ config
            text_column = data_config.TEXT_COLUMN
            label_columns = data_config.LABEL_COLUMNS
            
            # Kiểm tra các cột bắt buộc
            missing_cols = [col for col in [text_column] + label_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Processed cache thiếu các cột: {missing_cols}. "
                    f"Cột có trong file: {df.columns.tolist()}"
                )
            
            # Lấy processed texts (cột text theo config) và đảm bảo kiểu string
            # Một số dòng có thể là NaN/float -> chuyển về chuỗi an toàn để tránh lỗi .split()
            processed_series = df[text_column].fillna("").astype(str)
            processed_texts = processed_series.tolist()
            
            # Lấy labels (các cột label theo config)
            labels_df = df[label_columns]
            
            # Đưa vào context như prediction data (đã qua preprocessing)
            context.X_pred_processed = processed_texts
            context.y_pred = labels_df
            
            if context.logger_service:
                context.logger_service.info(
                    f"LoadProcessedDataStep | Loaded {len(context.X_pred_processed)} "
                    f"samples from {self.cache_file.name} (text_col={text_column}, label_cols={label_columns})"
                )
            
            return context
            
        except Exception as e:
            raise RuntimeError(f"Failed to load processed data from {self.cache_file}: {e}")

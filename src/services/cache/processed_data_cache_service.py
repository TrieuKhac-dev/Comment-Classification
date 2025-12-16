"""
Service để lưu và load dữ liệu đã qua preprocessing.
Lưu dữ liệu vào data/processed/ với format: processed_{prefix}_{dataset_type}_{hash}.csv (UTF-8)
"""
import os
import hashlib
import pandas as pd
from typing import Any, List, Optional
from pathlib import Path

from config.core.paths import PathConfig


class ProcessedDataCacheService:
    """
    Service để cache dữ liệu text đã qua preprocessing.
    Format file: processed_{prefix}_{dataset_type}_{hash}.csv (UTF-8)
    
    Ví dụ: processed_lightgbm_20231215_train_abc123.csv
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Khởi tạo service với thư mục cache.
        
        Args:
            cache_dir: Thư mục lưu cache. Mặc định: data/processed/
        """
        self.cache_dir = Path(cache_dir) if cache_dir else PathConfig.DATA_DIR / "processed"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str, prefix: str, dataset_type: str) -> Path:
        """
        Sinh đường dẫn file cache từ key, prefix và dataset_type.
        
        Args:
            key: Hash key từ dữ liệu gốc
            prefix: Prefix để phân biệt các phiên (vd: lightgbm_20231215_143025)
            dataset_type: Loại tập dữ liệu (train/test/val/predict)
        
        Returns:
            Đường dẫn đến file cache
        """
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        filename = f"processed_{prefix}_{dataset_type}_{key_hash}.csv"
        return self.cache_dir / filename

    def save(
        self,
        processed_texts: List[str],
        labels_df: Any,
        key: str,
        prefix: str,
        dataset_type: str,
        text_column: str = "processed_text",
        label_columns: Optional[List[str]] = None
    ):
        """
        Lưu dữ liệu đã preprocess vào cache (CSV UTF-8).
        
        Args:
            processed_texts: Danh sách text đã qua preprocessing
            labels_df: DataFrame chứa labels
            key: Hash key từ dữ liệu gốc
            prefix: Prefix để phân biệt các phiên
            dataset_type: Loại tập dữ liệu (train/test/val/predict)
            text_column: Tên cột cho processed text (mặc định: processed_text)
            label_columns: Danh sách tên cột labels (mặc định: None sẽ dùng label/label_0/label_1...)
        """
        path = self._get_cache_path(key, prefix, dataset_type)
        try:
            # Tạo DataFrame kết hợp processed texts và labels
            df = pd.DataFrame({
                text_column: processed_texts
            })
            
            # Thêm các cột labels
            if labels_df is not None:
                if isinstance(labels_df, pd.DataFrame):
                    df = pd.concat([df, labels_df.reset_index(drop=True)], axis=1)
                elif isinstance(labels_df, pd.Series):
                    col_name = label_columns[0] if label_columns and len(label_columns) > 0 else 'label'
                    df[col_name] = labels_df.reset_index(drop=True)
                else:
                    # Xử lý numpy array hoặc list
                    import numpy as np
                    if isinstance(labels_df, (np.ndarray, list)):
                        labels_array = np.array(labels_df)
                        # Nếu là 2D array (multi-label)
                        if len(labels_array.shape) > 1 and labels_array.shape[1] > 1:
                            for i in range(labels_array.shape[1]):
                                col_name = label_columns[i] if label_columns and i < len(label_columns) else f'label_{i}'
                                df[col_name] = labels_array[:, i]
                        else:
                            # 1D array hoặc 2D với 1 cột
                            col_name = label_columns[0] if label_columns and len(label_columns) > 0 else 'label'
                            df[col_name] = labels_array.flatten()
            
            # Lưu với encoding UTF-8
            df.to_csv(path, index=False, encoding='utf-8')
            print(f"[ProcessedDataCacheService] Processed data saved: {path}")
        except Exception as e:
            print(f"[ProcessedDataCacheService] Error saving at {path}: {e}")

    def load(self, key: str, prefix: str, dataset_type: str, text_column: str = "processed_text") -> Optional[dict]:
        """
        Load dữ liệu đã preprocess từ cache (CSV UTF-8).
        
        Args:
            key: Hash key từ dữ liệu gốc
            prefix: Prefix để phân biệt các phiên
            dataset_type: Loại tập dữ liệu
            text_column: Tên cột chứa processed text (mặc định: processed_text)
        
        Returns:
            Dict chứa {'processed_texts': List[str], 'labels_df': DataFrame} hoặc None
        """
        path = self._get_cache_path(key, prefix, dataset_type)
        if path.exists():
            try:
                df = pd.read_csv(path, encoding='utf-8')
                
                # Kiểm tra cột text có tồn tại không
                if text_column not in df.columns:
                    # Thử tìm cột text mặc định
                    if 'processed_text' in df.columns:
                        text_column = 'processed_text'
                    elif 'comment' in df.columns:
                        text_column = 'comment'
                    else:
                        print(f"[ProcessedDataCacheService] Text column '{text_column}' not found in {path}")
                        return None
                
                # Tách processed texts và labels
                processed_texts = df[text_column].tolist()
                
                # Các cột còn lại là labels
                label_columns = [col for col in df.columns if col != text_column]
                labels_df = df[label_columns] if label_columns else None
                
                return {
                    'processed_texts': processed_texts,
                    'labels_df': labels_df
                }
            except Exception as e:
                print(f"[ProcessedDataCacheService] Error loading {path}: {e}")
                return None
        return None

    def exists(self, key: str, prefix: str, dataset_type: str) -> bool:
        """
        Kiểm tra file cache có tồn tại hay không.
        
        Args:
            key: Hash key từ dữ liệu gốc
            prefix: Prefix để phân biệt các phiên
            dataset_type: Loại tập dữ liệu
        
        Returns:
            True nếu file cache tồn tại
        """
        return self._get_cache_path(key, prefix, dataset_type).exists()

    def make_cache_key(self, texts: List[str]) -> str:
        """
        Tạo cache key từ danh sách texts gốc.
        
        Args:
            texts: Danh sách text gốc (chưa preprocess)
        
        Returns:
            Hash key (MD5)
        """
        import json
        raw = json.dumps({'texts': texts}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode('utf-8')).hexdigest()

    def list_cached_files(self, prefix: Optional[str] = None) -> List[Path]:
        """
        Liệt kê các file cache trong thư mục.
        
        Args:
            prefix: Lọc theo prefix (optional)
        
        Returns:
            Danh sách đường dẫn file cache
        """
        pattern = f"processed_{prefix}_*" if prefix else "processed_*"
        return list(self.cache_dir.glob(f"{pattern}.csv"))

    def clear(self, prefix: Optional[str] = None):
        """
        Xóa các file cache.
        
        Args:
            prefix: Nếu có, chỉ xóa file có prefix này. Nếu None, xóa tất cả.
        """
        files = self.list_cached_files(prefix)
        for file in files:
            try:
                file.unlink()
                print(f"[ProcessedDataCacheService] Deleted: {file}")
            except Exception as e:
                print(f"[ProcessedDataCacheService] Error deleting {file}: {e}")

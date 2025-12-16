import pandas as pd
from pathlib import Path

class DataLoaderService:
    def load_csv(self, filepath: str, text_column: str, label_columns: list[str]) -> pd.DataFrame:
        return self._load_file(filepath, text_column, label_columns)

    def load_xlsx(self, filepath: str, text_column: str, label_columns: list[str]) -> pd.DataFrame:
        return self._load_file(filepath, text_column, label_columns, is_excel=True)

    def _load_file(self, filepath: str, text_column: str, label_columns: list[str], is_excel: bool = False) -> pd.DataFrame:
        df = self._read_file(filepath, is_excel)

        self._check_missing_columns(df, text_column, label_columns)
        df = self._remove_incomplete_rows(df, text_column, label_columns)
        df = self._remove_duplicates(df, text_column, label_columns)
        self._check_na_values(df, text_column, label_columns)
        self._check_binary_labels(df, label_columns)

        return df.reset_index(drop=True)

    # Bước 1: đọc file
    def _read_file(self, filepath: str, is_excel: bool) -> pd.DataFrame:
        try:
            if is_excel or str(filepath).lower().endswith('.xlsx'):
                return pd.read_excel(filepath)
            else:
                return pd.read_csv(filepath)
        except Exception as e:
            raise RuntimeError(f"Error reading data file: {e}")

    # Bước 2: kiểm tra tồn tại cột văn bản và nhãn
    def _check_missing_columns(self, df: pd.DataFrame, text_column: str, label_columns: list[str]):
        missing_cols = [col for col in [text_column] + label_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data file: {missing_cols}")

    # Bước 3: loại bỏ các dòng không đủ dữ liệu
    def _remove_incomplete_rows(self, df: pd.DataFrame, text_column: str, label_columns: list[str]) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=[text_column] + label_columns)
        after = len(df)
        if before - after > 0:
            print(f"Removed {before - after} incomplete rows")
        return df

    # Bước 4: loại bỏ các dòng trùng lặp
    def _remove_duplicates(self, df: pd.DataFrame, text_column: str, label_columns: list[str]) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=[text_column] + label_columns)
        after = len(df)
        if before - after > 0:
            print(f"Removed {before - after} duplicate rows")
        return df

    # Bước 5: kiểm tra giá trị NA
    def _check_na_values(self, df: pd.DataFrame, text_column: str, label_columns: list[str]):
        if df[[text_column] + label_columns].isnull().any().any():
            raise ValueError("Data contains NA values in text or label columns")

    # Bước 6: kiểm tra nhãn nhị phân
    def _check_binary_labels(self, df: pd.DataFrame, label_columns: list[str]):
        for col in label_columns:
            unique_vals = set(df[col].unique())
            if not unique_vals.issubset({0, 1}):
                raise ValueError(f"Label column {col} contains non-binary values: {unique_vals}")

    def get_text_and_labels(self, df: pd.DataFrame, text_column: str, label_columns: list[str]):
        texts = df[text_column].tolist()
        labels_df = df[label_columns]
        return texts, labels_df

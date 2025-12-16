import pandas as pd
from pathlib import Path

class DataLoaderService:
    def load_csv(self, filepath: str, text_column: str, label_columns: list[str]) -> pd.DataFrame:
        return self._load_file(filepath, text_column, label_columns)

    def load_xlsx(self, filepath: str, text_column: str, label_columns: list[str]) -> pd.DataFrame:
        return self._load_file(filepath, text_column, label_columns, is_excel=True)

    def _load_file(self, filepath: str, text_column: str, label_columns: list[str], is_excel: bool = False) -> pd.DataFrame:
        try:
            if is_excel or str(filepath).lower().endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
        except Exception as e:
            raise RuntimeError(f"Error reading data file: {e}")

        # Kiểm tra tồn tại cột văn bản và nhãn
        missing_cols = [col for col in [text_column] + label_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data file: {missing_cols}")

        # Loại bỏ các dòng trùng lặp dựa trên văn bản và nhãn
        before = len(df)
        df = df.drop_duplicates(subset=[text_column] + label_columns)
        after = len(df)
        if before - after > 0:
            print(f"Removed {before - after} duplicate rows")

        df = df.reset_index(drop=True)

        # Kiểm tra giá trị NA
        if df[[text_column] + label_columns].isnull().any().any():
            raise ValueError("Data contains NA values in text or label columns")

        # Kiểm tra nhãn nhị phân
        for col in label_columns:
            unique_vals = set(df[col].unique())
            if not unique_vals.issubset({0, 1}):
                raise ValueError(f"Label column {col} contains non-binary values: {unique_vals}")

        return df

    def get_text_and_labels(self, df: pd.DataFrame, text_column: str, label_columns: list[str]):
        texts = df[text_column].tolist()
        labels_df = df[label_columns]
        return texts, labels_df
import pandas as pd

class DataValidator:
    
    @staticmethod
    def is_columns_exist(
        df: pd.DataFrame, 
        text_column: str, 
        label_columns: list[str]
    ) -> bool:
        if text_column not in df.columns:
            return False
        
        missing_cols = set(label_columns) - set(df.columns)
        return len(missing_cols) == 0
    
    @staticmethod
    def has_na_values(
        df: pd.DataFrame,
        text_column: str,
        label_columns: list[str]
    ) -> bool:
        missing_cols = [col for col in [text_column] + label_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Các cột sau không tồn tại trong dữ liệu: {missing_cols}")
        return df[[text_column] + label_columns].isna().any().any()

    @staticmethod
    def has_duplicates(
        df: pd.DataFrame,
        text_column: str
    ) -> bool:
        return df[text_column].duplicated().any()
    
    @staticmethod
    def is_text_length_valid(
        df: pd.DataFrame,
        text_column: str,
        min_length: int = 1,
        max_length: int = 10000
    ) -> bool:
        text_lengths = df[text_column].str.len()
        return ((text_lengths >= min_length) & (text_lengths <= max_length)).all()
    
    @staticmethod
    def is_labels_binary(
        df: pd.DataFrame,
        label_columns: list[str]
    ) -> bool:
        for col in label_columns:
            if not df[col].isin([0, 1]).all():
                return False
        return True

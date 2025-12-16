from typing import Union
import pandas as pd
import numpy as np

class DataSplitterValidator:

    @staticmethod
    def validate_texts_and_labels(
        texts: list[str],
        labels: Union[pd.DataFrame, np.ndarray]
    ) -> None:
        if texts is None or labels is None:
            raise ValueError("Cả 'texts' và 'labels_df' đều không được để trống.")

    @staticmethod
    def validate_ratios_non_negative(
        test_ratio: float,
        val_ratio: float
    ) -> None:
        if test_ratio < 0 or val_ratio < 0:
            raise ValueError("Tỉ lệ phải không âm.")

    @staticmethod
    def validate_ratios_not_exceed_one(
        test_ratio: float,
        val_ratio: float
    ) -> None:
        if test_ratio > 1 or val_ratio > 1:
            raise ValueError("Tỉ lệ không được vượt quá 1.")

    @staticmethod
    def validate_sum_ratios(
        test_ratio: float,
        val_ratio: float
    ) -> None:
        if test_ratio + val_ratio > 1:
            raise ValueError("Tổng test_ratio và val_ratio không được vượt quá 1.")

    @classmethod
    def validate_split_params(
        cls,
        texts: list[str],
        labels: Union[pd.DataFrame, np.ndarray],
        test_ratio: float,
        val_ratio: float
    ) -> None:
        cls.validate_texts_and_labels(texts, labels)
        cls.validate_ratios_non_negative(test_ratio, val_ratio)
        cls.validate_ratios_not_exceed_one(test_ratio, val_ratio)
        cls.validate_sum_ratios(test_ratio, val_ratio)
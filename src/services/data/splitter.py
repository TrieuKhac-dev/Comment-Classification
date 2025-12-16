import numpy as np
import pandas as pd
from typing import Union
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

from src.interfaces.data import IDataSplitterService
from src.validators import DataSplitterValidator

class DataSplitterService(IDataSplitterService):
    def _convert_labels_to_array(self, labels: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(labels, pd.DataFrame):
            labels_array = labels.values
        else:
            labels_array = np.array(labels)
        if labels_array.ndim == 1:
            labels_array = labels_array.reshape(-1, 1)
        return labels_array

    def _perform_stratified_split(
        self,
        texts: list[str],
        labels_array: np.ndarray,
        test_size: float,
        random_seed: int
    ) -> tuple[list[int], list[int]]:
        # Nếu chỉ có 1 cột nhãn (binary/classification), dùng StratifiedShuffleSplit
        if labels_array.shape[1] == 1:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_seed
            )
            # labels_array cần là vector 1 chiều cho StratifiedShuffleSplit
            train_idx, test_idx = next(splitter.split(texts, labels_array.ravel()))
        else:
            # Multi-label: dùng MultilabelStratifiedShuffleSplit
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_seed
            )
            train_idx, test_idx = next(msss.split(texts, labels_array))
        return train_idx, test_idx

    def _extract_subset(
        self,
        texts: list[str],
        labels_array: np.ndarray,
        indices: list[int]
    ) -> tuple[list[str], np.ndarray]:
        subset_texts = [texts[i] for i in indices]
        subset_labels = labels_array[indices]
        return subset_texts, subset_labels

    def _calculate_val_size(self, val_ratio: float, test_ratio: float) -> float:
        return val_ratio / (1 - test_ratio)

    def _split_with_validation(
        self,
        train_texts: list[str],
        train_labels: np.ndarray,
        val_ratio: float,
        test_ratio: float,
        random_seed: int
    ) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
        val_size = self._calculate_val_size(val_ratio, test_ratio)
        train_idx, val_idx = self._perform_stratified_split(
            train_texts,
            train_labels,
            val_size,
            random_seed
        )
        final_train_texts, final_train_labels = self._extract_subset(
            train_texts,
            train_labels,
            train_idx
        )
        val_texts, val_labels = self._extract_subset(
            train_texts,
            train_labels,
            val_idx
        )
        return final_train_texts, final_train_labels, val_texts, val_labels

    def split(
        self, 
        texts: list[str], 
        labels: Union[pd.DataFrame, np.ndarray],
        test_ratio: float = 0.2,
        val_ratio: float = 0.0,
        random_seed: int = 42,
    ) -> dict:
        DataSplitterValidator.validate_split_params(texts, labels, test_ratio, val_ratio)
        labels_array = self._convert_labels_to_array(labels)
        train_idx, test_idx = self._perform_stratified_split(
            texts,
            labels_array,
            test_ratio,
            random_seed
        )
        train_texts, train_labels = self._extract_subset(texts, labels_array, train_idx)
        test_texts, test_labels = self._extract_subset(texts, labels_array, test_idx)
        result = {
            'train': {
                'texts': train_texts,
                'labels': train_labels
            },
            'test': {
                'texts': test_texts,
                'labels': test_labels
            }
        }
        if val_ratio > 0:
            train_texts, train_labels, val_texts, val_labels = self._split_with_validation(
                train_texts,
                train_labels,
                val_ratio,
                test_ratio,
                random_seed
            )
            result['train'] = {
                'texts': train_texts,
                'labels': train_labels
            }
            result['val'] = {
                'texts': val_texts,
                'labels': val_labels
            }
        return result

    def get_split_info(self, split_data: dict) -> str:
        info = []
        for split_name in ['train', 'val', 'test']:
            if split_name in split_data:
                n_samples = len(split_data[split_name]['texts'])
                info.append(f"{split_name}: {n_samples} samples")
        return " | ".join(info)
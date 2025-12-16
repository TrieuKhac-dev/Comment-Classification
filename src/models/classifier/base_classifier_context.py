from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np
import pandas as pd

from src.interfaces.classifier import IClassifier
from src.interfaces.repositories import IClassifierRepository
from src.interfaces.cache import IFeatureCacheService
from src.interfaces.logging import ILoggerService
from src.interfaces.machine_learning import IFeatureExtractor
from src.interfaces.data import IDataLoaderService, IDataSplitterService
from src.interfaces.processing import IPreprocessor

# type alias
LabelsType = Union[pd.DataFrame, pd.Series]

@dataclass
class BaseClassifierContext:
    #Info
    classifier_name: Optional[str] = None
    dataset_name: Optional[str] = None
    model_save_path: Optional[str] = None
    feature_cache_prefix: Optional[str] = None
    
    # Data load
    texts: Optional[List[str]] = None
    labels_df: Optional[LabelsType] = None

    # Split 
    X_train_texts: Optional[List[str]] = None
    X_test_texts: Optional[List[str]] = None
    X_val_texts: Optional[List[str]] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None

    # Preprocess 
    X_train_processed: Optional[List[str]] = None
    X_test_processed: Optional[List[str]] = None
    X_val_processed: Optional[List[str]] = None
    X_pred_processed: Optional[List[str]] = None

    # Features 
    X_train_features: Optional[np.ndarray] = None
    X_test_features: Optional[np.ndarray] = None
    X_val_features: Optional[np.ndarray] = None
    X_pred_features: Optional[np.ndarray] = None

    # Model 
    classifier: Optional[IClassifier] = None
    metrics: Optional[dict] = field(default_factory=dict)

    # Infracture
    logger_service: Optional[ILoggerService] = None
    feature_cache_service: Optional[IFeatureCacheService] = None
    model_repository: Optional[IClassifierRepository] = None
    extractor_service: Optional[IFeatureExtractor] = None
    data_loader_service: Optional[IDataLoaderService] = None
    splitter_service: Optional[IDataSplitterService] = None
    preprocessor_service: Optional[IPreprocessor] = None

    #predict
    X_pred_texts: Optional[List[str]] = None
    y_pred: Optional[np.ndarray] = None

class BaseClassifierContextBuilder:
    def __init__(self):
        self._context = {}

    # Info
    def set_dataset_name(self, name: str):
        self._context["dataset_name"] = name
        return self

    def set_model_save_path(self, path: str):
        self._context["model_save_path"] = path
        return self

    def set_feature_cache_prefix(self, prefix: str):
        self._context["feature_cache_prefix"] = prefix
        return self

    # Data 
    def set_texts(self, texts: List[str]):
        self._context['texts'] = texts
        return self

    def set_labels(self, labels: np.ndarray):
        self._context['labels_df'] = labels
        return self

    # Split 
    def set_X_train_texts(self, texts: List[str]):
        self._context['X_train_texts'] = texts
        return self

    def set_X_test_texts(self, texts: List[str]):
        self._context['X_test_texts'] = texts
        return self

    def set_y_train(self, y: np.ndarray):
        self._context['y_train'] = y
        return self

    def set_y_test(self, y: np.ndarray):
        self._context['y_test'] = y
        return self

    # Preprocess 
    def set_X_train_processed(self, texts: List[str]):
        self._context['X_train_processed'] = texts
        return self

    def set_X_test_processed(self, texts: List[str]):
        self._context['X_test_processed'] = texts
        return self

    # Features 
    def set_X_train_features(self, features: np.ndarray):
        self._context['X_train_features'] = features
        return self

    def set_X_test_features(self, features: np.ndarray):
        self._context['X_test_features'] = features
        return self

    # Model 
    def set_classifier(self, classifier: IClassifier):
        self._context['classifier'] = classifier
        return self

    def set_metrics(self, metrics: dict):
        self._context['metrics'] = metrics
        return self

    # Infracture
    def set_logger_service(self, logger: ILoggerService):
        self._context['logger_service'] = logger
        return self

    def set_feature_cache_service(self, cache: IFeatureCacheService):
        self._context['feature_cache_service'] = cache
        return self

    def set_model_repository(self, repo: IClassifierRepository):
        self._context['model_repository'] = repo
        return self

    def set_extractor_service(self, extractor: IFeatureExtractor):
        self._context['extractor_service'] = extractor
        return self
    
    def set_data_loader_service(self, loader: IDataLoaderService):
        self._context["data_loader_service"] = loader
        return self

    def set_splitter_service(self, splitter: IDataSplitterService):
        self._context["splitter_service"] = splitter
        return self

    def set_preprocessor_service(self, preprocessor: IPreprocessor):
        self._context["preprocessor_service"] = preprocessor
        return self

    def build(self) -> BaseClassifierContext:
        return BaseClassifierContext(**self._context)
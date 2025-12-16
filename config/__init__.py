"""
Package config chứa toàn bộ cấu hình ứng dụng.
Export các config instance để sử dụng trong toàn bộ ứng dụng.
"""

from .core.paths import paths
from .core.settings import settings

# Model config
from .model.embedding import embedding_config
from .model.classifier import classifier_config

# Training config
from .training.data import data_config
from .training.preprocessing import preprocessing_config
from .training.cache import cache_config
from .training.evaluation import evaluation_config
from .training.logging import logging_config
from .training.trainer import trainer_config 
__all__ = [
    "paths",
    "settings",
    "embedding_config",
    "classifier_config",
    "data_config",
    "preprocessing_config",
    "cache_config",
    "evaluation_config",
    "logging_config",
    "trainer_config",
]
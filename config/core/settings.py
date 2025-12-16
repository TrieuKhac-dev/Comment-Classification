from pathlib import Path
from config.core.paths import paths

from config.model.embedding import embedding_config
from config.model.classifier import classifier_config
from config.training.data import data_config
from config.training.preprocessing import preprocessing_config
from config.training.cache import cache_config
from config.training.evaluation import evaluation_config
from config.training.logging import logging_config
from config.training.trainer import trainer_config

class Settings:
    """
    Singleton chứa toàn bộ cấu hình ứng dụng.
    Đóng vai trò như cổng giao tiếp giữa folder config và các module bên ngoài.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Core config
        self.paths = paths
        
        # Model config - Sử dụng các instance đã được tạo sẵn
        self.embedding = embedding_config
        self.classifier = classifier_config
        
        # Training config - Sử dụng các instance đã được tạo sẵn
        self.data = data_config
        self.preprocessing = preprocessing_config
        self.cache = cache_config
        self.evaluation = evaluation_config
        self.logging = logging_config
        self.trainer = trainer_config

# Singleton instance để sử dụng trong toàn bộ ứng dụng
settings = Settings()
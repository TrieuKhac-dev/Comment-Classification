from pathlib import Path


class PathConfig:

    BASE_DIR = Path(__file__).resolve().parents[2]
    CONFIG_DIR = BASE_DIR / "config"
    SRC_DIR = BASE_DIR / "src"
    LOG_DIR = BASE_DIR / "logs"
    LOGS_DIR = BASE_DIR / "logs"  # Alias for backward compatibility
    TEMP_DIR = BASE_DIR / "temp"
    DATA_DIR = BASE_DIR / "data"

    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Thư mục cho processed cache

    ML_MODEL_DIR = BASE_DIR / "ml_model_storage"
    PRETRAINED_DIR = ML_MODEL_DIR / "pretrained"
    TRAINED_DIR = ML_MODEL_DIR / "trained"

    FASTTEXT_DIR = PRETRAINED_DIR / "fasttext"
    SBERT_DIR = PRETRAINED_DIR / "sbert"
    LIGHTGBM_DIR = TRAINED_DIR / "lightgbm"

    FEATURES_CACHE_DIR = TEMP_DIR / "feature_cache"
    MODEL_DOWNLOAD_TEMP_DIR = TEMP_DIR / "model_downloads"

    def __init__(self):
        self._create_directories()

    def _create_directories(self):
        dirs = [
            self.LOGS_DIR,
            self.TEMP_DIR,
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.ML_MODEL_DIR,
            self.PRETRAINED_DIR,
            self.TRAINED_DIR,
            self.FASTTEXT_DIR,
            self.SBERT_DIR,
            self.LIGHTGBM_DIR,
            self.FEATURES_CACHE_DIR,
            self.MODEL_DOWNLOAD_TEMP_DIR,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


paths = PathConfig()

from config.core.settings import settings
from src.containers import Container

from src.services.ml_model_loaders import (
    ML_LoaderFactory,
    FastTextLoader,
    SBERTLoader,
)
from src.services.ml_model_loaders.loader_service import LoaderService
from src.services.cache.model_cache_service import ModelCacheService
from src.services.cache.feature_cache_service import FeatureCacheService
from src.services.logging.logger_service import LoggerService
from src.services.logging.listeners import FileLoggerListener
from src.repositories.classifier import (
    ClassifierRepositoryFactory,
    JoblibRepositoryProvider,
    ClassifierRepository,
)
from src.services.extractors import (
    FeatureExtractorFactory,
    FastTextExtractor,
    SBERTExtractor,
)
from src.services.extractors.composite_extractor_builder import CompositeExtractorBuilder
from src.services.preprocessors.text_preprocessor_builder import TextPreprocessorBuilder
from src.services.data import DataLoaderService, DataSplitterService
from src.classifiers.lightgbm_classifier import LightGBMClassifier

def register_all(container: Container):
    # Đăng ký các Factory và Provider
    # Đăng ký provider cho kiểu lưu 'joblib'
    ClassifierRepositoryFactory.register("joblib", JoblibRepositoryProvider())

    # Đăng ký Loader vào Factory
    loader_registry = {
        "fasttext": FastTextLoader,
        "sbert": SBERTLoader,
    }
    for name, cls in loader_registry.items():
        ML_LoaderFactory.register(name, cls)

    # Đăng ký Extractor vào Factory
    extractor_registry = {
        "fasttext_only": FastTextExtractor,
        "sbert_only": SBERTExtractor,
    }
    for name, cls in extractor_registry.items():
        FeatureExtractorFactory.register(name, cls)

    # Khởi tạo các Service dùng chung (chỉ 1 lần)
    
    model_paths = {
        "fasttext": settings.paths.FASTTEXT_DIR / settings.embedding.FASTTEXT_MODEL_NAME,
        "sbert": settings.paths.SBERT_DIR / settings.embedding.SBERT_MODEL_NAME,
    }

    # Khởi tạo các service dùng chung
    model_cache_service = ModelCacheService()
    loader_service = LoaderService(model_cache_service)
    feature_cache_service = FeatureCacheService()
    logger_service = LoggerService()
    
    # Attach FileLoggerListener để ghi log vào file
    log_file_path = settings.paths.LOG_DIR / "application.log"
    file_listener = FileLoggerListener(str(log_file_path))
    logger_service.attach(file_listener)
    
    data_loader_service = DataLoaderService()
    splitter_service = DataSplitterService()
    repository = ClassifierRepository()
    lightgbm_classifier = LightGBMClassifier()

    # Đăng ký các Service vào DI Container
    container.register("model_cache_service", lambda: model_cache_service)
    container.register("feature_cache_service", lambda: feature_cache_service)
    container.register("logger_service", lambda: logger_service)
    container.register("data_loader_service", lambda: data_loader_service)
    container.register("splitter_service", lambda: splitter_service)
    container.register("repository", lambda: repository)
    container.register("lightgbm_classifier", lambda: lightgbm_classifier)

    # Đăng ký Extractor vào DI Container
    for key in ["fasttext", "sbert"]:
        extractor_key = f"{key}_only"
        # Đăng ký extractor với key đầy đủ (ví dụ: "fasttext_extractor")
        container.register(
            f"{key}_extractor",
            lambda k=key, ek=extractor_key: FeatureExtractorFactory.create(
                ek,
                model=loader_service.load(
                    k,
                    str(model_paths[k])
                )
            ),
        )
        # Đăng ký alias để dùng trong composite (ví dụ: "sbert", "fasttext")
        container.register(
            key,
            lambda k=key: container.resolve(f"{k}_extractor")
        )


    # Đăng ký Composite Extractor Builder và Preprocessor Builder
    container.register(
        "composite_extractor_builder",
        lambda: CompositeExtractorBuilder(container)
    )
    container.register(
        "preprocessor_builder",
        lambda: TextPreprocessorBuilder()
    )

    # Đăng ký extractor_service (dùng composite extractor)
    from src.services.extractors.extractor_service import ExtractorService
    from config.core.settings import settings as _settings
    def _extractor_service():
        composite_extractor = (
            container.resolve("composite_extractor_builder")
            .add_extractor('sbert', weight=1.0)
            .add_extractor('fasttext', weight=1.0)
            .set_combine_mode('concat')
            .build()
        )
        cache_service = container.resolve("feature_cache_service")
        return ExtractorService(composite_extractor, cache_service)
    container.register("extractor_service", _extractor_service)

    # Đăng ký preprocessor_service (add step trực tiếp, không dùng config ngoài)
    def _preprocessor_service():
        return (
            container.resolve("preprocessor_builder")
            .with_unicode_normalize(enabled=True, form="NFC")
            .with_lowercase(enabled=True)
            .with_emoji_removal(enabled=True)
            .with_special_chars_removal(enabled=True)
            .with_whitespace_normalization(enabled=True)
            .with_remove_non_Vietnamese_chars(enabled=True)
            .build()
        )
    container.register("preprocessor_service", _preprocessor_service)

# Khởi tạo container và đăng ký tất cả
container = Container()
register_all(container)
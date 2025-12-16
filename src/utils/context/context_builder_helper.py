from config.core.settings import settings
from src.app_setup import container
from src.services.extractors.extractor_service import ExtractorService

def build_extractor_service():
    # Tạo CompositeExtractor
    composite_extractor = (
        container.resolve("composite_extractor_builder")
        .add_extractor('sbert', weight=1.0)
        .add_extractor('fasttext', weight=1.0)
        .set_combine_mode('concat')
        .build()
    )
    
    cache_service = container.resolve("feature_cache_service")
    return ExtractorService(composite_extractor, cache_service)

def build_preprocessor_service():
    preprocessor_builder = container.resolve("preprocessor_builder")
    preprocessor_config = settings.preprocessing.PREPROCESSOR_CONFIG
    return preprocessor_builder.build(preprocessor_config)
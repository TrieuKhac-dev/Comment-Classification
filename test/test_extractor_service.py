import numpy as np
from src.services.extractors.extractor_service import ExtractorService
from src.services.extractors.fasttext_extractor import FastTextExtractor
from src.services.extractors.sbert_extractor import SBERTExtractor
from src.services.extractors.composite_extractor import CompositeExtractor
from src.services.cache.feature_cache_service import FeatureCacheService
from src.services.ml_model_loaders import ML_LoaderFactory, FastTextLoader, SBERTLoader
from config.core.settings import settings

ML_LoaderFactory.register("fasttext", FastTextLoader)
ML_LoaderFactory.register("sbert", SBERTLoader)

if __name__ == "__main__":
    print("[TEST] ExtractorService với FastTextExtractor (thật):")
    fasttext_model_path = settings.paths.FASTTEXT_DIR / settings.embedding.FASTTEXT_MODEL_NAME
    fasttext_model = ML_LoaderFactory.get_loader("fasttext").load(str(fasttext_model_path))
    fasttext_extractor = FastTextExtractor(fasttext_model)
    cache = FeatureCacheService()
    service = ExtractorService(fasttext_extractor, cache)
    texts = ["xin chào", "bạn khỏe không"]
    features = service.extract(texts, use_cache=False)
    print("Kết quả FastText:", features)
    print("Shape:", features.shape)

    print("\n[TEST] ExtractorService với SBERTExtractor (thật):")
    sbert_model_name = settings.embedding.SBERT_MODEL_NAME.split('/')[-1]
    sbert_model_path = settings.paths.SBERT_DIR / sbert_model_name
    sbert_model = ML_LoaderFactory.get_loader("sbert").load(str(sbert_model_path))
    sbert_extractor = SBERTExtractor(sbert_model)
    cache = FeatureCacheService()
    service = ExtractorService(sbert_extractor, cache)
    texts = ["xin chào", "bạn khỏe không"]
    features = service.extract(texts, use_cache=False)
    print("Kết quả SBERT:", features)
    print("Shape:", features.shape)

    print("\n[TEST] ExtractorService với cache (FastText):")
    cache = FeatureCacheService()
    service = ExtractorService(fasttext_extractor, cache)
    features1 = service.extract(texts, use_cache=True)
    features2 = service.extract(texts, use_cache=True)
    print("Kết quả lần 1:", features1)
    print("Kết quả lần 2 (từ cache):", features2)
    print("Giống nhau:", np.array_equal(features1, features2))
    # cache.clear()
    print("Tồn tại cache:", cache.exists(cache.make_cache_key(texts)))

    print("\n[TEST] ExtractorService với CompositeExtractor (FastText + SBERT):")
    # Chuẩn bị extractor thật
    fasttext_model_path = settings.paths.FASTTEXT_DIR / settings.embedding.FASTTEXT_MODEL_NAME
    fasttext_model = ML_LoaderFactory.get_loader("fasttext").load(str(fasttext_model_path))
    fasttext_extractor = FastTextExtractor(fasttext_model)
    sbert_model_name = settings.embedding.SBERT_MODEL_NAME.split('/')[-1]
    sbert_model_path = settings.paths.SBERT_DIR / sbert_model_name
    sbert_model = ML_LoaderFactory.get_loader("sbert").load(str(sbert_model_path))
    sbert_extractor = SBERTExtractor(sbert_model)
    # Kết hợp bằng CompositeExtractor
    composite = CompositeExtractor({
        "fasttext": fasttext_extractor,
        "sbert": sbert_extractor
    })
    cache = FeatureCacheService()
    service = ExtractorService(composite, cache)
    texts = ["xin chào", "bạn khỏe không"]
    features = service.extract(texts, use_cache=False)
    print("Kết quả Composite (FastText + SBERT):", features)
    print("Shape:", features.shape)

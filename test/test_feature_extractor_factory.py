"""
Unit tests for FeatureExtractorFactory and CompositeExtractor
"""
import sys
from pathlib import Path
import numpy as np

# Ensure project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.extractors.feature_extractor_factory import FeatureExtractorFactory
from src.services.extractors.fasttext_extractor import FastTextExtractor
from src.services.extractors.sbert_extractor import SBERTExtractor
from src.services.extractors.composite_extractor import CompositeExtractor
from src.services.extractors.composite_extractor_builder import CompositeExtractorBuilder
from src.containers.container import Container
from config.core.settings import settings
from src.services.ml_model_loaders import ML_LoaderFactory, FastTextLoader, SBERTLoader

ML_LoaderFactory.register("fasttext", FastTextLoader)
ML_LoaderFactory.register("sbert", SBERTLoader)

class FakeFastTextModel:
    def __init__(self, dim=2):
        self._dim = dim

    def get_dimension(self):
        return self._dim

    def get_word_vector(self, word: str):
        # deterministic vector per word
        return np.ones(self._dim)


class FakeSBERTModel:
    def __init__(self, dim=3):
        self._dim = dim

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=False, convert_to_numpy=True):
        # return deterministic embeddings
        n = len(texts)
        embeddings = np.ones((n, self._dim))
        return embeddings


# Đăng ký extractor cho test
FeatureExtractorFactory.register("fasttext_only", FastTextExtractor)
FeatureExtractorFactory.register("sbert_only", SBERTExtractor)
FeatureExtractorFactory.register("combined", CompositeExtractor)


def test_factory_fasttext_only():

        # Sử dụng model thật
        fasttext_model_path = settings.paths.FASTTEXT_DIR / settings.embedding.FASTTEXT_MODEL_NAME
        fasttext_model = ML_LoaderFactory.get_loader("fasttext").load(str(fasttext_model_path))
        container = Container()
        FeatureExtractorFactory.register("fasttext_only", FastTextExtractor)
        container.register(
            "fasttext_only_extractor",
            lambda: FeatureExtractorFactory.create('fasttext_only', model=fasttext_model)
        )
        ext = container.resolve("fasttext_only_extractor")
        print("[FastTextExtractor] get_dimension:", ext.get_dimension())
        texts = ["hello world", "abc"]
        features = ext.extract(texts, pooling='mean')
        print("[FastTextExtractor] features:", features)


def test_factory_sbert_only():

        # Sử dụng model thật
        sbert_model_name = settings.embedding.SBERT_MODEL_NAME.split('/')[-1]
        sbert_model_path = settings.paths.SBERT_DIR / sbert_model_name
        sbert_model = ML_LoaderFactory.get_loader("sbert").load(str(sbert_model_path))
        container = Container()
        FeatureExtractorFactory.register("sbert_only", SBERTExtractor)
        container.register(
            "sbert_only_extractor",
            lambda: FeatureExtractorFactory.create('sbert_only', model=sbert_model)
        )
        ext = container.resolve("sbert_only_extractor")
        print("[SBERTExtractor] get_dimension:", ext.get_dimension())
        texts = ["hello world", "abc"]
        features = ext.extract(texts, batch_size=16)
        print("[SBERTExtractor] features:", features)


def test_factory_combined_and_composite_extract():

        # Load model thật từ config
        fasttext_model_path = settings.paths.FASTTEXT_DIR / settings.embedding.FASTTEXT_MODEL_NAME
        fasttext_model = ML_LoaderFactory.get_loader("fasttext").load(str(fasttext_model_path))
        sbert_model_name = settings.embedding.SBERT_MODEL_NAME.split('/')[-1]
        sbert_model_path = settings.paths.SBERT_DIR / sbert_model_name
        sbert_model = ML_LoaderFactory.get_loader("sbert").load(str(sbert_model_path))

        container = Container()
        FeatureExtractorFactory.register("fasttext_only", FastTextExtractor)
        FeatureExtractorFactory.register("sbert_only", SBERTExtractor)
        container.register(
            "fasttext_only_extractor",
            lambda: FeatureExtractorFactory.create('fasttext_only', model=fasttext_model)
        )
        container.register(
            "sbert_only_extractor",
            lambda: FeatureExtractorFactory.create('sbert_only', model=sbert_model)
        )
        # Kết hợp bằng CompositeExtractorBuilder
        builder = CompositeExtractorBuilder(container)
        builder.add_extractor("fasttext_only_extractor").add_extractor("sbert_only_extractor")
        composite = builder.build()
        print("[CompositeExtractor] get_dimension:", composite.get_dimension())
        texts = ["hello world", "abc"]
        features = composite.extract(
            texts,
            configs={
                'fasttext_only_extractor': {'pooling': 'mean'},
                'sbert_only_extractor': {'batch_size': 16, 'show_progress_bar': True}
            }
        )
        print("[CompositeExtractor] features:", features)


def test_composite_kwargs_dispatch_and_weights_mismatch():

        # Sử dụng model thật
        fasttext_model_path = settings.paths.FASTTEXT_DIR / settings.embedding.FASTTEXT_MODEL_NAME
        fasttext_model = ML_LoaderFactory.get_loader("fasttext").load(str(fasttext_model_path))
        sbert_model_name = settings.embedding.SBERT_MODEL_NAME.split('/')[-1]
        sbert_model_path = settings.paths.SBERT_DIR / sbert_model_name
        sbert_model = ML_LoaderFactory.get_loader("sbert").load(str(sbert_model_path))
        ft_ext = FastTextExtractor(fasttext_model)
        sb_ext = SBERTExtractor(sbert_model)
        composite = CompositeExtractor({'fasttext': ft_ext, 'sbert': sb_ext})

        # Pass configs for each extractor; ensure dispatch to correct extractors
        texts = ["hello world"]
        features = composite.extract(texts, configs={'fasttext': {'pooling': 'mean'}, 'sbert': {'batch_size': 16}})
        print("[CompositeExtractor] features:", features)

        # Test weights mismatch: 2 extractors nhưng chỉ truyền 1 weight
        try:
            CompositeExtractor({'fasttext': ft_ext, 'sbert': sb_ext}, weights={'fasttext': 1.0})
            print("❌ Không phát sinh lỗi khi số lượng weights không khớp số extractor (cần kiểm tra lại validate)")
        except ValueError as e:
            print("✅ Đã phát sinh ValueError như mong đợi:", str(e))


def test_factory_fasttext_only_real():
    container = Container()
    FeatureExtractorFactory.register("fasttext_only", FastTextExtractor)
    # Load model thật từ config
    fasttext_model_path = settings.paths.FASTTEXT_DIR / settings.embedding.FASTTEXT_MODEL_NAME
    fasttext_model = ML_LoaderFactory.get_loader("fasttext").load(str(fasttext_model_path))
    container.register(
        "fasttext_only_extractor",
        lambda: FeatureExtractorFactory.create('fasttext_only', model=fasttext_model)
    )
    ext = container.resolve("fasttext_only_extractor")
    print("[FastTextExtractor] get_dimension:", ext.get_dimension())
    texts = ["hello world", "abc"]
    features = ext.extract(texts, pooling='mean')
    print("[FastTextExtractor] features:", features)


def test_factory_sbert_only_real():
    container = Container()
    FeatureExtractorFactory.register("sbert_only", SBERTExtractor)
    # Load model thật từ config
    from pathlib import Path
    sbert_model_name = settings.embedding.SBERT_MODEL_NAME.split('/')[-1]
    sbert_model_path = settings.paths.SBERT_DIR / sbert_model_name
    sbert_model = ML_LoaderFactory.get_loader("sbert").load(str(sbert_model_path))
    container.register(
        "sbert_only_extractor",
        lambda: FeatureExtractorFactory.create('sbert_only', model=sbert_model)
    )
    ext = container.resolve("sbert_only_extractor")
    print("[SBERTExtractor] get_dimension:", ext.get_dimension())
    texts = ["hello world", "abc"]
    features = ext.extract(texts, batch_size=16)
    print("[SBERTExtractor] features:", features)


if __name__ == '__main__':
    try:
        print("\n[TEST] FastTextExtractor (FAKE):")
        test_factory_fasttext_only()
        print("\n[TEST] SBERTExtractor (FAKE):")
        test_factory_sbert_only()
        print("\n[TEST] FastTextExtractor (REAL):")
        test_factory_fasttext_only_real()
        print("\n[TEST] SBERTExtractor (REAL):")
        test_factory_sbert_only_real()
        print("\n[TEST] CompositeExtractor (Builder):")
        test_factory_combined_and_composite_extract()
        print("\n[TEST] CompositeExtractor (Direct):")
        test_composite_kwargs_dispatch_and_weights_mismatch()
        print('\n--- Kết quả kiểm thử thực tế ---')
        print('Nếu không có lỗi, các extractor đã hoạt động đúng. Kiểm tra output phía trên để xem kết quả thực tế.')
        sys.exit(0)
    except Exception as e:
        print('TEST FAILED:', e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

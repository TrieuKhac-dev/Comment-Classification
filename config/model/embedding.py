class EmbeddingModelConfig:
    """Config cho FastText & SBERT."""

    # FastText
    FASTTEXT_MODEL_NAME = "cc.vi.300.bin"
    FASTTEXT_DIMENSION = 300
    FASTTEXT_POOLING = "mean"

    # SBERT
    # Use full repo id on Hugging Face (include namespace) to avoid 401/404 issues
    SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    SBERT_DIMENSION = 384
    SBERT_BATCH_SIZE = 200
    SBERT_MAX_LENGTH = 512
    SBERT_NORMALIZE_EMBEDDINGS = True

    # Feature Extractor options
    COMBINE_EMBEDDINGS = True
    USE_CACHE = True
    CACHE_TTL = None


embedding_config = EmbeddingModelConfig()

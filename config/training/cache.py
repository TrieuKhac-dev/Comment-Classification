class CacheConfig:
    FEATURE_CACHE_ENABLED = True
    MODEL_CACHE_ENABLED = True
    CACHE_DIR = "cache"
    CACHE_TTL = None  # None = không hết hạn

# Export singleton instance
cache_config = CacheConfig()

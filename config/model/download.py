class ModelDownloadConfig:
    FASTTEXT_BASE_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl"

    FASTTEXT_ALTERNATIVE_URLS = {
        "cc.vi.300.bin": [
            "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz"
        ]
    }

    SBERT_MARKER_FILE = ".download_complete"
    SBERT_PROGRESS_UPDATE_INTERVAL = 0.5

    CHUNK_SIZE = 8192
    EXTRACTION_CHUNK_SIZE = 8192

    MAX_RETRIES = 3
    RETRY_DELAY = 5
    DOWNLOAD_TIMEOUT = 3600
    CONNECT_TIMEOUT = 30


download_config = ModelDownloadConfig()
import os
from pathlib import Path
from typing import Any
from sentence_transformers import SentenceTransformer

from src.interfaces.machine_learning import ModelNotFoundError
from src.services.ml_model_loaders import BaseModelLoader

class SBERTLoader(BaseModelLoader):
    
    MARKER_FILE = ".download_complete"
    
    def _load_from_disk(self, model_path: str) -> Any:
        return SentenceTransformer(str(model_path))
    
    def _get_model_type(self) -> str:
        return "SBERT"
    
    def _get_download_instruction(self) -> str:
        return (
            "Please download the model first using the downloader utility.\n"
            "Example: python -m src.utils.download_sbert --model keepitreal/vietnamese-sbert"
        )
    
    def is_model_exists(self, model_path: str) -> bool:
        path = Path(model_path)
        marker_path = path / self.MARKER_FILE
        
        return path.exists() and path.is_dir() and marker_path.exists()

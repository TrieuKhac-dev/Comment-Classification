import os
import fasttext
from pathlib import Path
from typing import Any

from src.interfaces.machine_learning import ModelNotFoundError
from src.services.ml_model_loaders import BaseModelLoader

class FastTextLoader(BaseModelLoader):
    def _load_from_disk(self, model_path: str) -> Any:
        return fasttext.load_model(str(model_path))
    
    def _get_model_type(self) -> str:
        return "FastText"
    
    def _get_download_instruction(self) -> str:
        return (
            "Please download the model first using the downloader utility.\n"
            "Example: python -m src.utils.download_fasttext --model cc.vi.300"
        )
    
    def is_model_exists(self, model_path: str) -> bool:
        path = Path(model_path)
        return path.exists() and path.is_file()

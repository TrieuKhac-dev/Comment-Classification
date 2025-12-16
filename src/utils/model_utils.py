import os
from pathlib import Path
from typing import Optional


def get_latest_model_path(model_dir: Path, model_prefix: str = "lightgbm", extension: str = ".joblib") -> Optional[str]:
    if not model_dir.exists():
        return None
    
    # Tìm tất cả các file model phù hợp
    model_files = []
    for file in model_dir.iterdir():
        if file.is_file() and file.name.startswith(model_prefix) and file.name.endswith(extension):
            model_files.append(file)
    
    if not model_files:
        return None
    
    # Sắp xếp theo thời gian modify (mới nhất trước)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return str(model_files[0])


def get_all_models(model_dir: Path, model_prefix: str = "lightgbm", extension: str = ".joblib") -> list[str]:
    if not model_dir.exists():
        return []
    
    model_files = []
    for file in model_dir.iterdir():
        if file.is_file() and file.name.startswith(model_prefix) and file.name.endswith(extension):
            model_files.append(file)
    
    # Sắp xếp theo thời gian modify (mới nhất trước)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return [str(f) for f in model_files]

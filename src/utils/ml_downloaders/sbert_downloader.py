import time
import threading
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download

from config.core.paths import PathConfig
from config.model.download import download_config
from config.model.embedding import embedding_config


def _get_sbert_paths(model_name: str = None, output_path: str = None, marker_filename: str = None):
    """
    Get SBERT model paths (output directory and marker) consistently.
    
    Args:
        model_name: Model name on HuggingFace. Uses config default if None.
        output_path: Output directory path. Uses config default if None.
        marker_filename: Marker filename. Uses config default if None.
        
    Returns:
        tuple: (output_path, marker_path) as Path objects
    """
    model_name = model_name or embedding_config.SBERT_MODEL_NAME
    marker_filename = marker_filename or download_config.SBERT_MARKER_FILE
    
    if output_path is None:
        # Extract model name from HuggingFace path
        model_dirname = model_name.split('/')[-1]
        output_path = PathConfig.SBERT_DIR / model_dirname
    output_path = Path(output_path)
    marker_path = output_path / marker_filename
    
    return output_path, marker_path


def download_sbert_model(
    model_name: str = None,
    output_path: str = None,
    marker_filename: str = None,
    progress_interval: float = None,
    force: bool = False
) -> bool:
    
    model_name = model_name or embedding_config.SBERT_MODEL_NAME
    progress_interval = progress_interval or download_config.SBERT_PROGRESS_UPDATE_INTERVAL
    
    # Use helper function to get consistent paths
    output_path, marker_path = _get_sbert_paths(model_name, output_path, marker_filename)
    
    # Kiểm tra model đã tồn tại
    if marker_path.exists() and not force:
        print(f"[SBERT] Model đã tồn tại tại: {output_path}")
        print(f"[SBERT] Sử dụng --force để tải lại")
        return True
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[SBERT] Downloading: {model_name}")
    print(f"[SBERT] Output: {output_path}")
    print("[SBERT] Press Ctrl+C to cancel at any time")
    
    pbar = tqdm(unit='B', unit_scale=True, desc="[SBERT] Downloading")
    monitor_stop = {"flag": False}
    
    def progress_monitor():
        while not monitor_stop["flag"]:
            try:
                size = _get_folder_size(output_path)
                if size > pbar.n:
                    pbar.update(size - pbar.n)
                time.sleep(progress_interval)
            except:
                break
    
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=str(output_path)
        )
        
        monitor_stop["flag"] = True
        monitor_thread.join(timeout=2)
        pbar.close()
        
        _create_marker(marker_path)
        print(f"[SBERT] Download complete: {output_path}")
        return True
        
    except KeyboardInterrupt:
        print("\n[SBERT] Download cancelled by user")
        monitor_stop["flag"] = True
        pbar.close()
        return False
    except Exception as e:
        print(f"[SBERT] Error: {e}")
        monitor_stop["flag"] = True
        pbar.close()
        clear_sbert_model(output_path, marker_path)
        return False



def clear_sbert_model(output_path: Path = None, marker_path: Path = None):
    import shutil
    
    if output_path is None or marker_path is None:
        default_output, default_marker = _get_sbert_paths()
        output_path = output_path or default_output
        marker_path = marker_path or default_marker
    
    output_path = Path(output_path)
    marker_path = Path(marker_path)
    
    removed = []
    
    # Remove marker file first
    if marker_path and marker_path.exists():
        try:
            marker_path.unlink()
            removed.append(str(marker_path))
            print(f"[SBERT] Removed marker: {marker_path.name}")
        except Exception as e:
            print(f"[SBERT] Warning: Could not remove marker {marker_path}: {e}")
    
    # Remove entire directory
    if output_path.exists() and output_path.is_dir():
        try:
            shutil.rmtree(output_path)
            removed.append(str(output_path))
            print(f"[SBERT] Removed directory: {output_path}")
        except Exception as e:
            print(f"[SBERT] Warning: Could not remove directory {output_path}: {e}")
    
    if not removed:
        print("[SBERT] No files to remove")
    
    return len(removed) > 0


def _get_folder_size(folder: Path) -> int:
    total = 0
    try:
        for entry in folder.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except:
        pass
    return total


def _create_marker(marker_path: Path):
    temp_path = marker_path.with_suffix('.tmp')
    temp_path.write_text("ok")
    temp_path.replace(marker_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download SBERT model")
    parser.add_argument("--model", help="Model name on HuggingFace. Uses config default if not specified.")
    parser.add_argument("--output", help="Output directory. Uses config default if not specified.")
    parser.add_argument("--marker-filename", help="Completion marker filename. Uses config default if not specified.")
    parser.add_argument("--progress-interval", type=float, help="Progress update interval in seconds. Uses config default if not specified.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if model exists.")
    
    args = parser.parse_args()
    
    success = download_sbert_model(
        model_name=args.model,
        output_path=args.output,
        marker_filename=args.marker_filename,
        progress_interval=args.progress_interval,
        force=args.force
    )
    exit(0 if success else 1)

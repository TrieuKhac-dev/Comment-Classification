import gzip
import urllib.request
from pathlib import Path
from tqdm import tqdm

from config.core.paths import PathConfig
from config.model.download import download_config
from config.model.embedding import embedding_config


def _get_fasttext_paths(model_name: str = None, output_path: str = None):
    model_name = model_name or embedding_config.FASTTEXT_MODEL_NAME
    
    if output_path is None:
        output_path = PathConfig.FASTTEXT_DIR / model_name
    output_path = Path(output_path)
    
    # Store .gz file in temp directory, not alongside the final .bin file
    gz_filename = Path(model_name).with_suffix('.bin.gz').name
    gz_path = PathConfig.MODEL_DOWNLOAD_TEMP_DIR / gz_filename
    return output_path, gz_path


def download_fasttext_model(
    model_name: str = None,
    output_path: str = None,
    base_url: str = None,
    chunk_size: int = None,
    force: bool = False
) -> bool:
    
    model_name = model_name or embedding_config.FASTTEXT_MODEL_NAME
    base_url = base_url or download_config.FASTTEXT_BASE_URL
    chunk_size = chunk_size or download_config.EXTRACTION_CHUNK_SIZE
    
    output_path, gz_path = _get_fasttext_paths(model_name, output_path)
    
    # Kiểm tra model đã tồn tại
    if output_path.exists() and not force:
        print(f"[FastText] Model đã tồn tại tại: {output_path}")
        print(f"[FastText] Sử dụng --force để tải lại")
        return True
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gz_path.parent.mkdir(parents=True, exist_ok=True)  
    
    url = f"{base_url}/{model_name}.gz"
    
    print(f"[FastText] Downloading from: {url}")
    print(f"[FastText] Output: {output_path}")
    print("[FastText] Press Ctrl+C to cancel at any time")
    
    try:
        _download_with_progress(url, gz_path)
        _extract_gz(gz_path, output_path, chunk_size)
        gz_path.unlink()
        print(f"[FastText] Download complete: {output_path}")
        return True
    except KeyboardInterrupt:
        print("\n[FastText] Download cancelled by user")
        clear_fasttext_model(gz_path, output_path)
        return False
    except Exception as e:
        print(f"[FastText] Error: {e}")
        clear_fasttext_model(gz_path, output_path)
        return False

def clear_fasttext_model(gz_path: Path = None, output_path: Path = None):
    if gz_path is None or output_path is None:
        default_output, default_gz = _get_fasttext_paths()
        gz_path = gz_path or default_gz
        output_path = output_path or default_output
    
    gz_path = Path(gz_path)
    output_path = Path(output_path)
    
    removed = []
    
    if gz_path.exists():
        try:
            gz_path.unlink()
            removed.append(str(gz_path))
            print(f"[FastText] Removed: {gz_path.name}")
        except Exception as e:
            print(f"[FastText] Warning: Could not remove {gz_path}: {e}")
    
    if output_path.exists():
        try:
            output_path.unlink()
            removed.append(str(output_path))
            print(f"[FastText] Removed: {output_path.name}")
        except Exception as e:
            print(f"[FastText] Warning: Could not remove {output_path}: {e}")
    
    if not removed:
        print("[FastText] No files to remove")
    
    return len(removed) > 0


def _download_with_progress(url: str, output_path: Path):
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r[FastText] Downloading: {percent:.2f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
    
    urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
    print()


def _extract_gz(gz_path: Path, output_path: Path, chunk_size: int = 8192):
    print(f"[FastText] Extracting: {gz_path.name}")
    total_size = gz_path.stat().st_size
    
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download FastText model")
    parser.add_argument("--model", help="Model name. Uses config default if not specified.")
    parser.add_argument("--output", help="Output path. Uses config default if not specified.")
    parser.add_argument("--base-url", help="Base URL for download. Uses config default if not specified.")
    parser.add_argument("--chunk-size", type=int, help="Chunk size for extraction. Uses config default if not specified.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if model exists.")
    
    args = parser.parse_args()
    
    success = download_fasttext_model(
        model_name=args.model,
        output_path=args.output,
        base_url=args.base_url,
        chunk_size=args.chunk_size,
        force=args.force
    )
    exit(0 if success else 1)

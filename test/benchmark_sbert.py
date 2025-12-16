"""
Benchmark script để đo tốc độ extract features của SBERT model
"""
import sys
import time
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.core.paths import PathConfig
from config.model.embedding import embedding_config
from src.services.ml_model_loaders.sbert_loader import SBERTLoader
from src.services.extractors.sbert_extractor import SBERTExtractor


def benchmark_sbert_extract(
    num_texts: int = 100000,
    batch_size: int = 32,
    normalize_embeddings: bool = False,
    show_progress: bool = True,
    repeat: int = 3
):
    """
    Benchmark tốc độ extract features của SBERT
    
    Args:
        num_texts: Số lượng text để test
        batch_size: Batch size cho encoding
        normalize_embeddings: Có normalize embeddings không
        show_progress: Hiển thị progress bar không
        repeat: Số lần lặp lại để tính trung bình
    """
    
    print("=" * 80)
    print("BENCHMARK: SBERT Feature Extraction Speed")
    print("=" * 80)
    
    # 1. Load model
    print(f"\n[1] Loading SBERT model...")
    model_dirname = embedding_config.SBERT_MODEL_NAME.split('/')[-1]
    model_path = PathConfig.SBERT_DIR / model_dirname
    
    if not model_path.exists():
        print(f"  Model không tồn tại tại: {model_path}")
        print(f"  Hãy chạy downloader trước:")
        print(f"  python -m src.utils.ml_downloaders.sbert_downloader")
        return None
    
    marker_path = model_path / ".download_complete"
    if not marker_path.exists():
        print(f"  Marker file không tồn tại: {marker_path}")
        print(f"  Model có thể chưa download hoàn chỉnh")
        return None
    
    start_load = time.time()
    loader = SBERTLoader()
    model = loader.load(str(model_path))
    load_time = time.time() - start_load
    
    print(f"   Model loaded in {load_time:.2f}s")
    print(f"   - Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"   - Max sequence length: {model.max_seq_length}")
    print(f"   - Model type: {type(model).__name__}")
    
    # 2. Khởi tạo extractor
    extractor = SBERTExtractor(model)
    
    # 3. Tạo dữ liệu test
    print(f"\n[2] Generating {num_texts:,} test texts...")
    texts = [f"Đây là câu số {i} để test tốc độ embedding" for i in range(num_texts)]
    print(f"   Generated {len(texts):,} texts")
    
    # 4. Benchmark extraction
    print(f"\n[3] Benchmarking extraction...")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Normalize: {normalize_embeddings}")
    print(f"   - Show progress: {show_progress}")
    print(f"   - Repeat: {repeat}")
    
    times = []
    for i in range(repeat):
        print(f"\n   Run {i+1}/{repeat}:")
        
        start = time.time()
        embeddings = extractor.extract(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize_embeddings
        )
        elapsed = time.time() - start
        
        times.append(elapsed)
        
        texts_per_sec = num_texts / elapsed
        
        print(f"      - Elapsed: {elapsed:.4f}s")
        print(f"      - Speed: {texts_per_sec:,.0f} texts/sec")
        print(f"      - Shape: {embeddings.shape}")
        print(f"      - Dtype: {embeddings.dtype}")
        
        if normalize_embeddings and i == 0:
            import numpy as np
            norms = [np.linalg.norm(vec) for vec in embeddings[:5]]
            print(f"      - Sample norms (first 5): {[f'{n:.4f}' for n in norms]}")
    
    # 5. Kết quả tổng hợp
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_speed = num_texts / avg_time
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Số texts: {num_texts:,}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Normalize: {normalize_embeddings}")
    print(f"  - Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  - Max seq length: {model.max_seq_length}")
    print(f"  - Số lần chạy: {repeat}")
    
    print(f"\nTiming:")
    print(f"  - Average: {avg_time:.4f}s")
    print(f"  - Min: {min_time:.4f}s")
    print(f"  - Max: {max_time:.4f}s")
    
    print(f"\nThroughput:")
    print(f"  - Average speed: {avg_speed:,.0f} texts/sec")
    print(f"  - Min speed: {num_texts/max_time:,.0f} texts/sec")
    print(f"  - Max speed: {num_texts/min_time:,.0f} texts/sec")
    
    # Batch processing efficiency
    batches = (num_texts + batch_size - 1) // batch_size
    time_per_batch = avg_time / batches
    print(f"\nBatch Processing:")
    print(f"  - Total batches: {batches}")
    print(f"  - Time per batch: {time_per_batch*1000:.2f}ms")
    print(f"  - Texts per batch: {batch_size}")
    
    print("=" * 80)
    
    return {
        'num_texts': num_texts,
        'batch_size': batch_size,
        'normalize_embeddings': normalize_embeddings,
        'dimension': model.get_sentence_embedding_dimension(),
        'repeat': repeat,
        'times': times,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_speed': avg_speed,
        'shape': embeddings.shape
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark SBERT extraction speed")
    parser.add_argument("--num-texts", type=int, default=100, 
                       help="Number of texts to benchmark (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for encoding (default: 32)")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize embeddings to unit length")
    parser.add_argument("--no-progress", action="store_true",
                       help="Hide progress bar")
    parser.add_argument("--repeat", type=int, default=3,
                       help="Number of times to repeat benchmark (default: 3)")
    
    args = parser.parse_args()
    
    try:
        result = benchmark_sbert_extract(
            num_texts=args.num_texts,
            batch_size=args.batch_size,
            normalize_embeddings=args.normalize,
            show_progress=not args.no_progress,
            repeat=args.repeat
        )
        
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n    Benchmark cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n   FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

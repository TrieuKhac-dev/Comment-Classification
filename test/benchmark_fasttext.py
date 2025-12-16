"""
Benchmark script để đo tốc độ extract features của FastText model
"""
import sys
import time
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.core.paths import PathConfig
from config.model.embedding import embedding_config
from src.services.ml_model_loaders.fasttext_loader import FastTextLoader
from src.services.extractors.fasttext_extractor import FastTextExtractor


def benchmark_fasttext_extract(
    num_texts: int = 100000,
    pooling: str = "mean",
    repeat: int = 3
):
    """
    Benchmark tốc độ extract features của FastText
    
    Args:
        num_texts: Số lượng text để test
        pooling: Phương pháp pooling ('mean', 'max', 'sum')
        repeat: Số lần lặp lại để tính trung bình
    """
    
    print("=" * 80)
    print("BENCHMARK: FastText Feature Extraction Speed")
    print("=" * 80)
    
    # 1. Load model
    print(f"\n[1] Loading FastText model...")
    model_path = PathConfig.FASTTEXT_DIR / embedding_config.FASTTEXT_MODEL_NAME
    
    if not model_path.exists():
        print(f"   Model không tồn tại tại: {model_path}")
        print(f"   Hãy chạy downloader trước:")
        print(f"   python -m src.utils.ml_downloaders.fasttext_downloader")
        return None
    
    start_load = time.time()
    loader = FastTextLoader()
    model = loader.load(str(model_path))
    load_time = time.time() - start_load
    
    print(f"✅ Model loaded in {load_time:.2f}s")
    print(f"   - Dimension: {model.get_dimension()}")
    print(f"   - Model type: {type(model).__name__}")
    
    # 2. Khởi tạo extractor
    extractor = FastTextExtractor(model)
    
    # 3. Tạo dữ liệu test
    print(f"\n[2] Generating {num_texts:,} test texts...")
    texts = [f"Đây là câu số {i} để test tốc độ embedding" for i in range(num_texts)]
    print(f"✅ Generated {len(texts):,} texts")
    
    # 4. Benchmark extraction
    print(f"\n[3] Benchmarking extraction (pooling={pooling}, repeat={repeat})...")
    
    times = []
    for i in range(repeat):
        print(f"\n   Run {i+1}/{repeat}:")
        
        start = time.time()
        embeddings = extractor.extract(texts, pooling=pooling)
        elapsed = time.time() - start
        
        times.append(elapsed)
        
        texts_per_sec = num_texts / elapsed
        
        print(f"      - Elapsed: {elapsed:.4f}s")
        print(f"      - Speed: {texts_per_sec:,.0f} texts/sec")
        print(f"      - Shape: {embeddings.shape}")
        print(f"      - Dtype: {embeddings.dtype}")
    
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
    print(f"  - Pooling: {pooling}")
    print(f"  - Dimension: {model.get_dimension()}")
    print(f"  - Số lần chạy: {repeat}")
    
    print(f"\nTiming:")
    print(f"  - Average: {avg_time:.4f}s")
    print(f"  - Min: {min_time:.4f}s")
    print(f"  - Max: {max_time:.4f}s")
    
    print(f"\nThroughput:")
    print(f"  - Average speed: {avg_speed:,.0f} texts/sec")
    print(f"  - Min speed: {num_texts/max_time:,.0f} texts/sec")
    print(f"  - Max speed: {num_texts/min_time:,.0f} texts/sec")
    
    print("=" * 80)
    
    return {
        'num_texts': num_texts,
        'pooling': pooling,
        'dimension': model.get_dimension(),
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
    
    parser = argparse.ArgumentParser(description="Benchmark FastText extraction speed")
    parser.add_argument("--num-texts", type=int, default=100000, 
                       help="Number of texts to benchmark (default: 100000)")
    parser.add_argument("--pooling", choices=['mean', 'max', 'sum'], default='mean',
                       help="Pooling strategy (default: mean)")
    parser.add_argument("--repeat", type=int, default=3,
                       help="Number of times to repeat benchmark (default: 3)")
    
    args = parser.parse_args()
    
    try:
        result = benchmark_fasttext_extract(
            num_texts=args.num_texts,
            pooling=args.pooling,
            repeat=args.repeat
        )
        
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

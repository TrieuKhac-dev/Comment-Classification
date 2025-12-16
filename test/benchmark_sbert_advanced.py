"""
Benchmark script để đo tốc độ extract features của SBERT Advanced với multi-processing
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
from src.services.extractors.sbert_extractor_advanced import SBERTExtractorAdvanced


def benchmark_sbert_advanced(
    num_texts: int = 100000,
    batch_size: int = 200,
    normalize_embeddings: bool = False,
    show_progress: bool = True,
    num_workers: int = 4,
    chunk_size: int = 2000,
    repeat: int = 3
):
    """
    Benchmark tốc độ extract features của SBERT Advanced
    
    Args:
        num_texts: Số lượng text để test
        batch_size: Batch size cho encoding
        normalize_embeddings: Có normalize embeddings không
        show_progress: Hiển thị progress bar không
        num_workers: Số workers cho multi-processing (0 = single process)
        chunk_size: Chunk size khi dùng multi-processing
        repeat: Số lần lặp lại để tính trung bình
    """
    
    print("=" * 80)
    print("BENCHMARK: SBERT Advanced Feature Extraction Speed")
    print("=" * 80)
    
    # 1. Load model
    print(f"\n[1] Loading SBERT model...")
    model_dirname = embedding_config.SBERT_MODEL_NAME.split('/')[-1]
    model_path = PathConfig.SBERT_DIR / model_dirname
    
    if not model_path.exists():
        print(f"  Model không tồn tại tại: {model_path}")
        print(f"  Hãy chạy downloader trước:")
        print(f"   python -m src.utils.ml_downloaders.sbert_downloader")
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
    
    print(f"  Model loaded in {load_time:.2f}s")
    print(f"   - Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"   - Max sequence length: {model.max_seq_length}")
    print(f"   - Model type: {type(model).__name__}")
    
    # 2. Khởi tạo extractor
    extractor = SBERTExtractorAdvanced(model)
    
    # 3. Tạo dữ liệu test
    print(f"\n[2] Generating {num_texts:,} test texts...")
    texts = [f"Đây là câu số {i} để test tốc độ embedding với SBERT Advanced" for i in range(num_texts)]
    print(f"  Generated {len(texts):,} texts")
    
    # 4. Xác định chế độ processing
    use_multiprocessing = extractor.validate_num_workers(num_workers) and extractor.validate_chunk_size(chunk_size)
    processing_mode = "Multi-processing" if use_multiprocessing else "Single-process"
    
    # 5. Benchmark extraction
    print(f"\n[3] Benchmarking extraction ({processing_mode})...")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Normalize: {normalize_embeddings}")
    print(f"   - Show progress: {show_progress}")
    if use_multiprocessing:
        print(f"   - Num workers: {num_workers}")
        print(f"   - Chunk size: {chunk_size}")
    print(f"   - Repeat: {repeat}")
    
    times = []
    embeddings = None
    
    for i in range(repeat):
        print(f"\n   Run {i+1}/{repeat}:")
        
        start = time.time()
        embeddings = extractor.extract(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize_embeddings,
            num_workers=num_workers,
            chunk_size=chunk_size
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
        
        # Giải phóng memory nếu không phải run cuối
        if i < repeat - 1:
            import gc
            del embeddings
            embeddings = None
            gc.collect()
    
    # 6. Test similarity functions
    print(f"\n[4] Testing similarity functions...")
    try:
        import numpy as np
        
        # Test similarity_texts
        text1 = "Sản phẩm tốt"
        text2 = "Hàng chất lượng"
        sim = extractor.similarity_texts(text1, text2, normalize=True)
        print(f"   - similarity_texts('{text1}', '{text2}'): {sim:.4f}")
        
        # Test similarity_vectors
        vec1 = np.random.randn(extractor.get_dimension())
        vec2 = np.random.randn(extractor.get_dimension())
        sim_vec = extractor.similarity_vectors(vec1, vec2, normalize=True)
        print(f"   - similarity_vectors (random): {sim_vec:.4f}")
        
    except Exception as e:
        print(f"      Similarity test error: {e}")
    
    # 7. Kết quả tổng hợp
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_speed = num_texts / avg_time
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Processing mode: {processing_mode}")
    print(f"  - Số texts: {num_texts:,}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Normalize: {normalize_embeddings}")
    if use_multiprocessing:
        print(f"  - Num workers: {num_workers}")
        print(f"  - Chunk size: {chunk_size}")
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
    
    # Multi-processing efficiency (nếu dùng)
    if use_multiprocessing:
        chunks = (num_texts + chunk_size - 1) // chunk_size
        time_per_chunk = avg_time / chunks
        print(f"\nMulti-processing:")
        print(f"  - Total chunks: {chunks}")
        print(f"  - Time per chunk: {time_per_chunk*1000:.2f}ms")
        print(f"  - Texts per chunk: {chunk_size}")
        print(f"  - Theoretical speedup: ~{num_workers}x")
    
    print("=" * 80)
    
    return {
        'num_texts': num_texts,
        'batch_size': batch_size,
        'normalize_embeddings': normalize_embeddings,
        'num_workers': num_workers,
        'chunk_size': chunk_size,
        'use_multiprocessing': use_multiprocessing,
        'dimension': model.get_sentence_embedding_dimension(),
        'repeat': repeat,
        'times': times,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_speed': avg_speed
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark SBERT Advanced extraction speed with multi-processing support"
    )
    parser.add_argument("--num-texts", type=int, default=100, 
                       help="Number of texts to benchmark (default: 100)")
    parser.add_argument("--batch-size", type=int, default=200,
                       help="Batch size for encoding (default: 200)")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize embeddings to unit length")
    parser.add_argument("--no-progress", action="store_true",
                       help="Hide progress bar")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of workers for multi-processing (default: 4)")
    parser.add_argument("--chunk-size", type=int, default=2000,
                       help="Chunk size for multi-processing (default: 0 = auto)")
    parser.add_argument("--repeat", type=int, default=3,
                       help="Number of times to repeat benchmark (default: 3)")
    
    args = parser.parse_args()
    
    try:
        result = benchmark_sbert_advanced(
            num_texts=args.num_texts,
            batch_size=args.batch_size,
            normalize_embeddings=args.normalize,
            show_progress=not args.no_progress,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            repeat=args.repeat
        )
        
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n   Benchmark cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n  FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

import sys
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.core.paths import PathConfig
from config.model.embedding import embedding_config
from src.services.ml_model_loaders import FastTextLoader, SBERTLoader
from src.services.cache import model_cache_service


def test_cache_key_consistency():
    """Kiểm thử cache key consistency - tất cả loaders dùng cùng logic"""
    
    print("=" * 80)
    print("TEST: Cache Key Consistency Across Loaders")
    print("=" * 80)
    
    # Test 1: Cache key generation
    print("\n[1] Testing cache key generation...")
    print("-" * 80)
    
    ft_loader = FastTextLoader()
    sbert_loader = SBERTLoader()
    
    # Test với cùng 1 path
    test_path = "/some/path/to/model.bin"
    
    ft_key = ft_loader._get_cache_key(test_path)
    sbert_key = sbert_loader._get_cache_key(test_path)
    cache_key = model_cache_service.normalize_path(test_path)
    
    print(f"  FastText cache key: {ft_key}")
    print(f"  SBERT cache key:    {sbert_key}")
    print(f"  Direct cache key:   {cache_key}")
    
    if ft_key == sbert_key == cache_key:
        print(f"  PASS: Tất cả cache keys giống nhau")
    else:
        print(f"  FAIL: Cache keys khác nhau!")
        return False
    
    # Test 2: Relative vs absolute path
    print("\n[2] Testing path normalization...")
    print("-" * 80)
    
    rel_path = "models/fasttext/cc.vi.300.bin"
    abs_path = (Path.cwd() / rel_path).resolve()
    
    key1 = ft_loader._get_cache_key(rel_path)
    key2 = ft_loader._get_cache_key(str(abs_path))
    
    print(f"  Relative path: {rel_path}")
    print(f"  Cache key:     {key1}")
    print(f"  Absolute path: {abs_path}")
    print(f"  Cache key:     {key2}")
    
    if key1 == key2:
        print(f"  PASS: Relative và absolute paths normalize về cùng key")
    else:
        print(f"  FAIL: Paths normalize về keys khác nhau!")
        return False
    
    # Test 3: Load real models và verify cache
    print("\n[3] Testing real model loading với cache consistency...")
    print("-" * 80)
    
    # Clear cache trước
    model_cache_service.clear()
    print(f"  Cleared cache, size: {model_cache_service.cache_size()}")
    
    # Load FastText nếu có
    ft_model_path = PathConfig.FASTTEXT_DIR / embedding_config.FASTTEXT_MODEL_NAME
    if ft_model_path.exists():
        print(f"\n  [3.1] Loading FastText model...")
        ft_loader1 = FastTextLoader()
        ft_loader2 = FastTextLoader()  # Instance khác
        
        model1 = ft_loader1.load(str(ft_model_path))
        print(f"    Cache size after load 1: {model_cache_service.cache_size()}")
        
        model2 = ft_loader2.load(str(ft_model_path))
        print(f"    Cache size after load 2: {model_cache_service.cache_size()}")
        
        if model_cache_service.cache_size() == 1:
            print(f"    PASS: Chỉ có 1 entry trong cache (2 loaders dùng chung key)")
        else:
            print(f"    FAIL: Cache có {model_cache_service.cache_size()} entries!")
            return False
        
        if model1 is model2:
            print(f"    PASS: model1 is model2 (same instance from cache)")
        else:
            print(f"    FAIL: model1 và model2 là instances khác nhau!")
            return False
    else:
           print(f"  FastText model not found, skipping")
    
    # Load SBERT nếu có
    sbert_dirname = embedding_config.SBERT_MODEL_NAME.split('/')[-1]
    sbert_model_path = PathConfig.SBERT_DIR / sbert_dirname
    
    if sbert_model_path.exists():
        print(f"\n  [3.2] Loading SBERT model...")
        sbert_loader1 = SBERTLoader()
        sbert_loader2 = SBERTLoader()  # Instance khác

        prev_size = model_cache_service.cache_size()
        model1 = sbert_loader1.load(str(sbert_model_path))
        print(f"    Cache size after load 1: {model_cache_service.cache_size()}")
        
        model2 = sbert_loader2.load(str(sbert_model_path))
        print(f"    Cache size after load 2: {model_cache_service.cache_size()}")
        
        if model_cache_service.cache_size() == prev_size + 1:
            print(f"    PASS: Chỉ thêm 1 entry (2 loaders dùng chung key)")
        else:
            print(f"    FAIL: Cache tăng {model_cache_service.cache_size() - prev_size} entries!")
            return False
        
        if model1 is model2:
            print(f"    PASS: model1 is model2 (same instance from cache)")
        else:
            print(f"    FAIL: model1 và model2 là instances khác nhau!")
            return False
    else:
           print(f"  SBERT model not found, skipping")
    
    # Test 4: Verify BaseModelLoader methods
    print("\n[4] Testing BaseModelLoader methods...")
    print("-" * 80)
    
    print(f"  FastText model type: {ft_loader._get_model_type()}")
    print(f"  SBERT model type:    {sbert_loader._get_model_type()}")
    
    if ft_loader._get_model_type() == "FastText":
        print(f"  PASS: FastText model type correct")
    else:
        print(f"  FAIL: FastText model type incorrect!")
        return False
    
    if sbert_loader._get_model_type() == "SBERT":
        print(f"  PASS: SBERT model type correct")
    else:
        print(f"  FAIL: SBERT model type incorrect!")
        return False
    
    # Test 5: Cache statistics
    print("\n[5] Final cache statistics...")
    print("-" * 80)
    print(f"  Total cached models: {model_cache_service.cache_size()}")
    print(f"  Cached keys:")
    for key in model_cache_service.get_cached_keys():
        print(f"    - {Path(key).name}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED: Cache Key Consistency Working!")
    print("=" * 80)
    print("\n  Key Benefits:")
    print("  - Tất cả loaders dùng chung cache key generation logic")
    print("  - BaseModelLoader đảm bảo consistency")
    print("  - Không có duplicate cache entries")
    print("  - Multiple loader instances share same cached model")
    print("  - Path normalization (relative -> absolute) consistent")
    
    return True


if __name__ == "__main__":
    try:
        success = test_cache_key_consistency()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n   FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

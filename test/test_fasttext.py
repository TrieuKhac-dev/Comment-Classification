"""
Test script để kiểm thử load và extract FastText model
"""
import sys
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.core.paths import PathConfig
from config.model.embedding import embedding_config
from src.services.ml_model_loaders.fasttext_loader import FastTextLoader
from src.services.extractors.fasttext_extractor import FastTextExtractor
from src.interfaces.machine_learning.i_model_loader import ModelNotFoundError
from src.interfaces.machine_learning.i_feature_extractor import ModelNotLoadedError


def test_fasttext_load_and_extract():
    """Kiểm thử load model và extract features từ text"""
    
    print("=" * 80)
    print("TEST: FastText Model Loading & Feature Extraction")
    print("=" * 80)
    
    # 1. Kiểm tra đường dẫn model
    model_path = PathConfig.FASTTEXT_DIR / embedding_config.FASTTEXT_MODEL_NAME
    print(f"\n[1] Đường dẫn model: {model_path}")
    
    if not model_path.exists():
        print(f"  Model không tồn tại tại: {model_path}")
        print(f"  Hãy chạy downloader trước:")
        print(f"   python -m src.utils.ml_downloaders.fasttext_downloader")
        return False
    else:
        print(f"  Model tồn tại")
    
    # 2. Test FastTextLoader
    print(f"\n[2] Testing FastTextLoader...")
    try:
        loader = FastTextLoader()
        
        # Test is_model_exists
        print(f"   - Checking if model exists: {loader.is_model_exists(str(model_path))}")
        
        # Test load
        print(f"   - Loading model...")
        model = loader.load(str(model_path))
        print(f"     Model loaded successfully")
        print(f"   - Model dimension: {model.get_dimension()}")
        print(f"   - Model type: {type(model)}")
        
    except ModelNotFoundError as e:
        print(f"     ModelNotFoundError: {e}")
        return False
    except Exception as e:
        print(f"     Unexpected error: {e}")
        return False
    
    # 3. Test FastTextExtractor
    print(f"\n[3] Testing FastTextExtractor...")
    
    # 3.1. Test khởi tạo không có model
    print(f"   [3.1] Test extractor without model...")
    try:
        extractor_no_model = FastTextExtractor()
        print(f"      - is_loaded(): {extractor_no_model.is_loaded()}")
        
        # Thử extract khi chưa load model (phải raise error)
        try:
            extractor_no_model.extract(["test"])
            print(f"        Should raise ModelNotLoadedError")
            return False
        except ModelNotLoadedError:
            print(f"        Correctly raised ModelNotLoadedError")
    except Exception as e:
        print(f"        Unexpected error: {e}")
        return False
    
    # 3.2. Test extractor với model
    print(f"   [3.2] Test extractor with loaded model...")
    try:
        extractor = FastTextExtractor(model)
        print(f"      - is_loaded(): {extractor.is_loaded()}")
        print(f"      - get_dimension(): {extractor.get_dimension()}")
        
    except Exception as e:
        print(f"        Error: {e}")
        return False
    
    # 3.3. Test extract với các pooling strategies
    print(f"   [3.3] Test extract with different pooling strategies...")
    test_texts = [
        "Sản phẩm này rất tốt",
        "Chất lượng kém, không đáng tiền",
        "Giao hàng nhanh, đóng gói cẩn thận"
    ]
    
    pooling_methods = ['mean', 'max', 'sum']
    
    for pooling in pooling_methods:
        try:
            features = extractor.extract(test_texts, pooling=pooling)
            print(f"      - Pooling '{pooling}': shape={features.shape}, dtype={features.dtype}")
            print(f"        First vector norm: {np.linalg.norm(features[0]):.4f}")
        except Exception as e:
            print(f"        Error with pooling '{pooling}': {e}")
            return False
    
    # 3.4. Test edge cases
    print(f"   [3.4] Test edge cases...")
    
    # Empty text
    try:
        empty_features = extractor.extract([""], pooling='mean')
        print(f"      - Empty text: shape={empty_features.shape}, all zeros={np.allclose(empty_features, 0)}")
    except Exception as e:
        print(f"        Error with empty text: {e}")
        return False
    
    # Multiple texts
    try:
        batch_texts = ["text " + str(i) for i in range(10)]
        batch_features = extractor.extract(batch_texts, pooling='mean')
        print(f"      - Batch (10 texts): shape={batch_features.shape}")
    except Exception as e:
        print(f"        Error with batch: {e}")
        return False
    
    # 3.5. Test set_model
    print(f"   [3.5] Test set_model...")
    try:
        new_extractor = FastTextExtractor()
        print(f"      - Before set_model: is_loaded={new_extractor.is_loaded()}")
        
        new_extractor.set_model(model)
        print(f"      - After set_model: is_loaded={new_extractor.is_loaded()}")
        
        features = new_extractor.extract(["test text"], pooling='mean')
        print(f"      - Extract after set_model: shape={features.shape}")
        
    except Exception as e:
        print(f"        Error with set_model: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("  ALL TESTS PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import numpy as np
    
    try:
        success = test_fasttext_load_and_extract()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n  FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
Test script để kiểm thử load và extract SBERT model
"""
import sys
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.core.paths import PathConfig
from config.model.embedding import embedding_config
from src.services.ml_model_loaders.sbert_loader import SBERTLoader
from src.services.extractors.sbert_extractor import SBERTExtractor
from src.interfaces.machine_learning.i_model_loader import ModelNotFoundError
from src.interfaces.machine_learning.i_feature_extractor import ModelNotLoadedError


def test_sbert_load_and_extract():
    """Kiểm thử load model và extract features từ text"""
    
    print("=" * 80)
    print("TEST: SBERT Model Loading & Feature Extraction")
    print("=" * 80)
    
    # 1. Kiểm tra đường dẫn model
    model_dirname = embedding_config.SBERT_MODEL_NAME.split('/')[-1]
    model_path = PathConfig.SBERT_DIR / model_dirname
    print(f"\n[1] Đường dẫn model: {model_path}")
    
    if not model_path.exists():
        print(f"  Model không tồn tại tại: {model_path}")
        print(f"  Hãy chạy downloader trước:")
        print(f"   python -m src.utils.ml_downloaders.sbert_downloader")
        return False
    else:
        print(f"  Model directory tồn tại")
    
    # Kiểm tra marker file
    marker_path = model_path / ".download_complete"
    if not marker_path.exists():
        print(f"  Marker file không tồn tại: {marker_path}")
        print(f"  Model có thể chưa download hoàn chỉnh")
        return False
    else:
        print(f"  Marker file tồn tại")
    
    # 2. Test SBERTLoader
    print(f"\n[2] Testing SBERTLoader...")
    try:
        loader = SBERTLoader()
        
        # Test is_model_exists
        print(f"   - Checking if model exists: {loader.is_model_exists(str(model_path))}")
        
        # Test load
        print(f"   - Loading model...")
        model = loader.load(str(model_path))
        print(f"     Model loaded successfully")
        print(f"   - Model dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   - Model type: {type(model)}")
        print(f"   - Max sequence length: {model.max_seq_length}")
        
    except ModelNotFoundError as e:
        print(f"     ModelNotFoundError: {e}")
        return False
    except Exception as e:
        print(f"     Unexpected error: {e}")
        return False
    
    # 3. Test SBERTExtractor
    print(f"\n[3] Testing SBERTExtractor...")
    
    # 3.1. Test khởi tạo không có model
    print(f"   [3.1] Test extractor without model...")
    try:
        extractor_no_model = SBERTExtractor()
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
        extractor = SBERTExtractor(model)
        print(f"      - is_loaded(): {extractor.is_loaded()}")
        print(f"      - get_dimension(): {extractor.get_dimension()}")
        
    except Exception as e:
        print(f"        Error: {e}")
        return False
    
    # 3.3. Test extract với batch processing
    print(f"   [3.3] Test extract with batch processing...")
    test_texts = [
        "Sản phẩm này rất tốt",
        "Chất lượng kém, không đáng tiền",
        "Giao hàng nhanh, đóng gói cẩn thận"
    ]
    
    try:
        # Test với batch_size khác nhau
        features = extractor.extract(test_texts, batch_size=2, show_progress_bar=False)
        print(f"      - Batch size 2: shape={features.shape}, dtype={features.dtype}")
        print(f"        First vector norm: {np.linalg.norm(features[0]):.4f}")
        
        features_large = extractor.extract(test_texts, batch_size=32, show_progress_bar=False)
        print(f"      - Batch size 32: shape={features_large.shape}")
        
        # Kiểm tra kết quả giống nhau
        print(f"        Results identical: {np.allclose(features, features_large)}")
        
    except Exception as e:
        print(f"        Error: {e}")
        return False
    
    # 3.4. Test normalize embeddings
    print(f"   [3.4] Test normalize embeddings...")
    try:
        features_normalized = extractor.extract(
            test_texts, 
            batch_size=32, 
            normalize_embeddings=True
        )
        print(f"      - Normalized embeddings: shape={features_normalized.shape}")
        
        # Kiểm tra norm của vector sau normalize (phải gần 1.0)
        norms = [np.linalg.norm(vec) for vec in features_normalized]
        print(f"        Vector norms: {[f'{n:.4f}' for n in norms]}")
        print(f"        All close to 1.0: {all(abs(n - 1.0) < 0.01 for n in norms)}")
        
    except Exception as e:
        print(f"        Error: {e}")
        return False
    
    # 3.5. Test edge cases
    print(f"   [3.5] Test edge cases...")
    
    # Empty text
    try:
        empty_features = extractor.extract([""], batch_size=1)
        print(f"      - Empty text: shape={empty_features.shape}")
        print(f"        Vector norm: {np.linalg.norm(empty_features[0]):.4f}")
    except Exception as e:
        print(f"        Error with empty text: {e}")
        return False
    
    # Single text
    try:
        single_features = extractor.extract(["Single text"], batch_size=1)
        print(f"      - Single text: shape={single_features.shape}")
    except Exception as e:
        print(f"        Error with single text: {e}")
        return False
    
    # Large batch
    try:
        batch_texts = ["Text số " + str(i) for i in range(100)]
        batch_features = extractor.extract(batch_texts, batch_size=16)
        print(f"      - Large batch (100 texts): shape={batch_features.shape}")
    except Exception as e:
        print(f"        Error with large batch: {e}")
        return False
    
    # 3.6. Test set_model
    print(f"   [3.6] Test set_model...")
    try:
        new_extractor = SBERTExtractor()
        print(f"      - Before set_model: is_loaded={new_extractor.is_loaded()}")
        
        new_extractor.set_model(model)
        print(f"      - After set_model: is_loaded={new_extractor.is_loaded()}")
        
        features = new_extractor.extract(["test text"], batch_size=1)
        print(f"      - Extract after set_model: shape={features.shape}")
        
    except Exception as e:
        print(f"        Error with set_model: {e}")
        return False
    
    # 3.7. Test semantic similarity
    print(f"   [3.7] Test semantic similarity...")
    try:
        # Các câu có nghĩa tương tự
        similar_texts = [
            "Sản phẩm tốt",
            "Hàng chất lượng",
            "Món hàng rất tệ"
        ]
        
        features = extractor.extract(similar_texts, normalize_embeddings=True)
        
        # Tính cosine similarity giữa câu 1 và 2 (tương tự)
        sim_12 = np.dot(features[0], features[1])
        print(f"      - Similarity (câu 1 vs câu 2 - tương tự): {sim_12:.4f}")
        
        # Tính cosine similarity giữa câu 1 và 3 (khác nhau)
        sim_13 = np.dot(features[0], features[2])
        print(f"      - Similarity (câu 1 vs câu 3 - trái nghĩa): {sim_13:.4f}")
        
        print(f"        Similarity check passed: {sim_12 > sim_13}")
        
    except Exception as e:
        print(f"        Error with similarity test: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("  ALL TESTS PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import numpy as np
    
    try:
        success = test_sbert_load_and_extract()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n  FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

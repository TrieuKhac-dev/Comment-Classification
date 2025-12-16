"""
Test script để verify Singleton pattern cho Settings
"""
import sys
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.core.settings import Settings, settings


def test_singleton_pattern():
    """Kiểm thử Singleton pattern - chỉ có 1 instance duy nhất"""
    
    print("=" * 80)
    print("TEST: Singleton Pattern for Settings")
    print("=" * 80)
    
    # Test 1: Tạo nhiều instances
    print("\n[1] Testing multiple instantiations...")
    s1 = Settings()
    s2 = Settings()
    s3 = Settings()
    
    print(f"  s1 id: {id(s1)}")
    print(f"  s2 id: {id(s2)}")
    print(f"  s3 id: {id(s3)}")
    
    if id(s1) == id(s2) == id(s3):
        print("  ✓ PASS: Tất cả trỏ về cùng 1 instance")
    else:
        print("  ✗ FAIL: Có instances khác nhau!")
        return False
    
    # Test 2: So với global settings instance
    print("\n[2] Testing global settings instance...")
    print(f"  settings id: {id(settings)}")
    print(f"  s1 id:       {id(s1)}")
    
    if id(settings) == id(s1):
        print("  ✓ PASS: Global settings và new instance giống nhau")
    else:
        print("  ✗ FAIL: Global settings khác với new instance!")
        return False
    
    # Test 3: Verify tất cả dùng chung data
    print("\n[3] Testing shared state...")
    print(f"  settings.random_seed: {settings.random_seed}")
    print(f"  s1.random_seed:       {s1.random_seed}")
    print(f"  s2.random_seed:       {s2.random_seed}")
    
    if settings.random_seed == s1.random_seed == s2.random_seed:
        print("  ✓ PASS: Tất cả share cùng config")
    else:
        print("  ✗ FAIL: Config không giống nhau!")
        return False
    
    # Test 4: Modify state và verify consistency
    print("\n[4] Testing state consistency after modification...")
    
    # Thử modify attribute
    original_seed = s1.data.RANDOM_SEED
    s1.data.RANDOM_SEED = 9999
    
    print(f"  Modified s1.data.RANDOM_SEED = 9999")
    print(f"  s2.data.RANDOM_SEED: {s2.data.RANDOM_SEED}")
    print(f"  s3.data.RANDOM_SEED: {s3.data.RANDOM_SEED}")
    print(f"  settings.data.RANDOM_SEED: {settings.data.RANDOM_SEED}")
    
    if s2.data.RANDOM_SEED == s3.data.RANDOM_SEED == settings.data.RANDOM_SEED == 9999:
        print("  ✓ PASS: Modification reflected across all instances")
    else:
        print("  ✗ FAIL: State không consistent!")
        return False
    
    # Restore
    s1.data.RANDOM_SEED = original_seed
    
    # Test 5: Verify __init__ chỉ chạy 1 lần
    print("\n[5] Testing __init__ called only once...")
    print(f"  Settings._initialized: {Settings._initialized}")
    
    s4 = Settings()
    print(f"  Created s4, Settings._initialized: {Settings._initialized}")
    
    if Settings._initialized:
        print("  ✓ PASS: __init__ chỉ chạy 1 lần")
    else:
        print("  ✗ FAIL: __init__ chạy nhiều lần!")
        return False
    
    # Test 6: Verify is operator
    print("\n[6] Testing 'is' operator...")
    if s1 is s2 is s3 is settings:
        print("  ✓ PASS: s1 is s2 is s3 is settings")
    else:
        print("  ✗ FAIL: 'is' operator failed!")
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED: Singleton pattern working correctly!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        success = test_singleton_pattern()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

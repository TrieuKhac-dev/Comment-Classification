from examples.examples import (
    example_basic_usage,
    example_batch_processing,
    example_simple_endpoint,
)
import requests


def main():
    print("="*80)
    print("COMMENT CLASSIFICATION API - CLIENT EXAMPLES")
    print("="*80)
    print("\nMake sure API server is running at http://localhost:8000")
    print("   Run server: python src/api_server.py")
    try:
        example_basic_usage()
        example_batch_processing()
        example_simple_endpoint()
        print("\n" + "="*80)
        print("All examples completed!")
        print("="*80)
    except requests.exceptions.ConnectionError:
        print("\nKhông thể kết nối tới API server!")
        print("   Hãy chắc chắn server đã chạy: python src/api_server.py")
    except Exception as e:
        print(f"\nLỗi: {e}")

if __name__ == "__main__":
    main()

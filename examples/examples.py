from .client import CommentClassificationClient
from .print_utils import print_prediction_results


def example_basic_usage():
    print("\nEXAMPLE 1: Basic usage")
    client = CommentClassificationClient()
    print("\n1. Health check...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    comments = {
        "1": "Sản phẩm rất tốt, tôi rất hài lòng!",
        "2": "Đồ rác, lừa đảo, đừng mua!",
        "3": "Giao hàng nhanh, đóng gói cẩn thận",
        "4": "Chất lượng tệ, không nên mua",
        "5": "Rất đáng tiền, sẽ ủng hộ shop tiếp"
    }
    print("\n2. Sending classification request...")
    result = client.predict(comments)
    print_prediction_results(result)


def example_batch_processing():
    print("\nEXAMPLE 2: Batch processing")
    client = CommentClassificationClient()
    large_batch = {
        str(i): comment for i, comment in enumerate([
            "Sản phẩm đẹp",
            "Chất lượng tốt",
            "Đồ rác, lừa đảo",
            "Giao hàng nhanh",
            "Không đáng tiền",
            "Rất hài lòng",
            "Shop lừa đảo",
            "Sẽ mua lại",
            "Không nên tin",
            "Chất lượng xuất sắc"
        ], 1)
    }
    print(f"\nProcessing {len(large_batch)} comments...")
    result = client.predict(large_batch)
    violation_rate = (result['violation_count'] / result['total_comments']) * 100
    print(f"\nStatistics:")
    print(f"   - Total: {result['total_comments']} comments")
    print(f"   - Violations: {result['violation_count']} ({violation_rate:.1f}%)")
    print(f"   - Valid: {result['total_comments'] - result['violation_count']} ({100-violation_rate:.1f}%)")


def example_simple_endpoint():
    print("\nEXAMPLE 3: Simple endpoint")
    client = CommentClassificationClient()
    comments = [
        "Sản phẩm rất tốt!",
        "Đồ rác, không nên mua!",
        "Giao hàng nhanh"
    ]
    print(f"\nClassifying {len(comments)} comments...")
    result = client.predict_simple(comments)
    print("\nResults:")
    for pred in result['predictions']:
        status = "X" if pred['is_violation'] else "OK"
        print(f"[{status}] {pred['comment']}")
        print(f"   Violation probability: {pred['violation_probability']:.2%}")

# Có thể bổ sung các example khác tương tự nếu cần

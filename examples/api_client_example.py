"""
Client example để test Comment Classification API
"""

import requests
from typing import Dict, List
import json


class CommentClassificationClient:
    """Client để tương tác với Comment Classification API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Khởi tạo client
        
        Args:
            base_url: URL của API server (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> Dict:
        """
        Kiểm tra trạng thái server
        
        Returns:
            Dict chứa thông tin health check
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict(self, comments: Dict[str, str]) -> Dict:
        """
        Phân loại bình luận (endpoint chính)
        
        Args:
            comments: Dictionary {id: comment_text}
            
        Returns:
            Dict chứa kết quả phân loại
            
        Example:
            >>> client = CommentClassificationClient()
            >>> result = client.predict({
            ...     "1": "Sản phẩm tốt",
            ...     "2": "Đồ rác"
            ... })
            >>> print(result)
        """
        response = requests.post(
            f"{self.base_url}/predict",
            json={"comments": comments},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def predict_simple(self, comments: List[str]) -> Dict:
        """
        Phân loại bình luận (endpoint đơn giản)
        
        Args:
            comments: List các comment text
            
        Returns:
            Dict chứa predictions
        """
        response = requests.post(
            f"{self.base_url}/predict/simple",
            json=comments,
            timeout=60
        )
        response.raise_for_status()
        return response.json()


def print_prediction_results(results: Dict):
    """In kết quả dự đoán một cách đẹp mắt"""
    print("\n" + "="*80)
    print(f"COMMENT CLASSIFICATION RESULTS")
    print("="*80)
    print(f"Total comments: {results['total_comments']}")
    print(f"Violating comments: {results['violation_count']}")
    print()
    
    for comment_id, prediction in results['results'].items():
        status = "VIOLATION" if prediction['is_violation'] else "VALID"
        prob = prediction['violation_probability']
        comment = prediction['comment']
        
        print(f"ID {comment_id}: {status}")
        print(f"  Comment: {comment}")
        print(f"  Violation probability: {prob:.2%}")
        print()
    
    print("="*80)


def example_basic_usage():
    """Ví dụ sử dụng cơ bản"""
    print("\nEXAMPLE 1: Basic usage")
    
    # Khởi tạo client
    client = CommentClassificationClient()
    
    # Kiểm tra health
    print("\n1. Health check...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    
    # Dữ liệu test
    comments = {
        "1": "Sản phẩm rất tốt, tôi rất hài lòng!",
        "2": "Đồ rác, lừa đảo, đừng mua!",
        "3": "Giao hàng nhanh, đóng gói cẩn thận",
        "4": "Chất lượng tệ, không nên mua",
        "5": "Rất đáng tiền, sẽ ủng hộ shop tiếp"
    }
    
    # Gửi request
    print("\n2. Sending classification request...")
    result = client.predict(comments)
    
    # In kết quả
    print_prediction_results(result)


def example_batch_processing():
    """Ví dụ xử lý batch lớn"""
    print("\nEXAMPLE 2: Batch processing")
    
    client = CommentClassificationClient()
    
    # Dữ liệu lớn
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
    
    # Thống kê
    violation_rate = (result['violation_count'] / result['total_comments']) * 100
    print(f"\nStatistics:")
    print(f"   - Total: {result['total_comments']} comments")
    print(f"   - Violations: {result['violation_count']} ({violation_rate:.1f}%)")
    print(f"   - Valid: {result['total_comments'] - result['violation_count']} ({100-violation_rate:.1f}%)")


def example_simple_endpoint():
    """Ví dụ sử dụng endpoint đơn giản"""
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


def example_filter_violations():
    """Ví dụ lọc các bình luận vi phạm"""
    print("\nEXAMPLE 4: Filter violations")
    
    client = CommentClassificationClient()
    
    comments = {
        "101": "Sản phẩm rất tốt!",
        "102": "Đồ rác, lừa đảo!",
        "103": "Giao hàng nhanh",
        "104": "Shop lừa đảo, không nên mua",
        "105": "Chất lượng xuất sắc"
    }
    
    result = client.predict(comments)
    
    # Lọc bình luận vi phạm
    violations = {
        cid: pred for cid, pred in result['results'].items()
        if pred['is_violation']
    }
    
    print(f"\nDetected {len(violations)} violating comments:")
    for comment_id, pred in violations.items():
        print(f"\nID {comment_id}:")
        print(f"  {pred['comment']}")
        print(f"  Confidence: {pred['violation_probability']:.2%}")


def example_high_confidence_filter():
    """Ví dụ lọc theo độ tin cậy cao"""
    print("\nEXAMPLE 5: Confidence-based filtering")
    
    client = CommentClassificationClient()
    
    comments = {
        str(i): comment for i, comment in enumerate([
            "Sản phẩm tốt",
            "Đồ rác, lừa đảo",
            "Bình thường",
            "Shop lừa đảo",
            "Ổn"
        ], 1)
    }
    
    result = client.predict(comments)
    
    # Lọc vi phạm với confidence > 0.8
    high_confidence_violations = {
        cid: pred for cid, pred in result['results'].items()
        if pred['is_violation'] and pred['violation_probability'] > 0.8
    }
    
    print(f"\nHigh confidence violations (>80%):")
    for comment_id, pred in high_confidence_violations.items():
        print(f"\nID {comment_id}:")
        print(f"  {pred['comment']}")
        print(f"  {pred['violation_probability']:.2%}")
    
    # Lọc cần review (0.5 < prob < 0.8)
    need_review = {
        cid: pred for cid, pred in result['results'].items()
        if 0.5 < pred['violation_probability'] < 0.8
    }
    
    print(f"\nNeed review (50%-80%):")
    for comment_id, pred in need_review.items():
        print(f"\nID {comment_id}:")
        print(f"  {pred['comment']}")
        print(f"  {pred['violation_probability']:.2%}")


def example_error_handling():
    """Ví dụ xử lý lỗi"""
    print("\nEXAMPLE 6: Error handling")
    
    client = CommentClassificationClient()
    
    try:
        # Request với empty data
        result = client.predict({})
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
    
    try:
        # Request đến server không tồn tại
        wrong_client = CommentClassificationClient("http://localhost:9999")
        result = wrong_client.predict({"1": "test"})
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: Cannot connect to server")


def example_custom_url():
    """Ví dụ sử dụng custom URL"""
    print("\nEXAMPLE 7: Custom URL")
    
    # Kết nối đến server production
    production_client = CommentClassificationClient("https://api.example.com")
    
    # Hoặc custom port
    custom_port_client = CommentClassificationClient("http://localhost:8001")
    
    print("Client configured with custom URL")


def main():
    """Chạy tất cả ví dụ"""
    print("="*80)
    print("COMMENT CLASSIFICATION API - CLIENT EXAMPLES")
    print("="*80)
    print("\nMake sure API server is running at http://localhost:8000")
    print("   Run server: python src/api_server.py")
    
    try:
        # Kiểm tra server trước
        client = CommentClassificationClient()
        health = client.health_check()
        print(f"\nServer is running (Status: {health['status']})")
        
        # Chạy các ví dụ
        example_basic_usage()
        example_batch_processing()
        example_simple_endpoint()
        example_filter_violations()
        example_high_confidence_filter()
        example_error_handling()
        example_custom_url()
        
        print("\n" + "="*80)
        print("All examples completed!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\nCannot connect to API server!")
        print("   Make sure server is running:")
        print("   python src/api_server.py")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()

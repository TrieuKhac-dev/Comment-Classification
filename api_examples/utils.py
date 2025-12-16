from typing import Dict


def print_prediction_results(results: Dict):
    """In kết quả dự đoán một cách đẹp mắt"""
    print("\n" + "="*80)
    print("COMMENT CLASSIFICATION RESULTS")
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
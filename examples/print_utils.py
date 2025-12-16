def print_prediction_results(results):
    print("\n" + "="*80)
    print(f"COMMENT CLASSIFICATION RESULTS")
    print("="*80)
    print(f"Tổng số bình luận: {results['total_comments']}")
    print(f"Số bình luận vi phạm: {results['violation_count']}")
    print()
    for comment_id, prediction in results['results'].items():
        status = "VIOLATION" if prediction['is_violation'] else "VALID"
        prob = prediction['violation_probability']
        comment = prediction['comment']
        print(f"ID {comment_id}: {status}")
        print(f"  Bình luận: {comment}")
        print(f"  Xác suất vi phạm: {prob:.2%}")
        print()
    print("="*80)

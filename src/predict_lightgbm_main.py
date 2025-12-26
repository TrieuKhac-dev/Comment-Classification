import numpy as np
from src.app_setup import container
from src.pipelines.pipeline_config.lightgbm_pipeline_config import LightGBMPipelineConfig
from src.models.classifier.base_classifier_context import BaseClassifierContextBuilder
from config.core.paths import paths
from src.utils.model_utils import get_latest_model_path

# Ngưỡng để coi là vi phạm (sử dụng P(class_1) >= THRESHOLD)
THRESHOLD = 0.75

def predict_comments(comments):
    """
    Dự đoán nhãn cho danh sách bình luận.
    Sử dụng pipeline predict với LightGBM classifier.
    """
    # Lấy các service đã đăng ký từ container
    logger_service = container.resolve("logger_service")
    repository = container.resolve("repository")
    extractor_service = container.resolve("extractor_service")
    preprocessor_service = container.resolve("preprocessor_service")

    # Xây dựng context cho pipeline predict
    context = (
        BaseClassifierContextBuilder()
        .set_logger_service(logger_service)
        .set_preprocessor_service(preprocessor_service)
        .set_extractor_service(extractor_service)
        .set_model_repository(repository)
        .set_classifier(container.resolve("lightgbm_classifier"))
        .build()
    )
    context.X_pred_texts = comments

    # Tìm model mới nhất
    model_path = get_latest_model_path(paths.LIGHTGBM_DIR, "lightgbm", ".joblib")
    if not model_path:
        raise FileNotFoundError(f"No LightGBM model found in {paths.LIGHTGBM_DIR}")
    
    print(f"Using model: {model_path}")

    # Tạo pipeline predict
    predict_pipeline = LightGBMPipelineConfig.predict_pipeline(
        model_path=model_path,
        return_proba=True
    )

    # Chạy pipeline predict
    context = predict_pipeline.run(context)

    # Đọc và in kết quả
    y_pred = context.y_pred
    labels = np.argmax(y_pred, axis=1)
    print("Kết quả dự đoán cho comment:", comments)
    print("Dự đoán:", y_pred.round(2))
    for c, l, prob in zip(comments, labels, y_pred):
        is_violation = float(prob[1]) >= THRESHOLD
        print(f"Bình luận: {c}")
        print(f"  Xác suất: {prob}")
        print(f"  Nhãn dự đoán (argmax): {l}")
        print(f"  Quyết định theo ngưỡng {THRESHOLD}: {'vi phạm' if is_violation else 'không vi phạm'}")

if __name__ == "__main__":
    comments = [
        "đả đảo chính quyền",
        "tao đập chết cha mày",
        "hàng này xịn nè",
        "đồ ngu, đồ ăn hại",
        "ngon bổ rẻ",
        "hàng xịn, ngon bổ rẻ",
        "ngon bổ rẻ, hàng xịn",
        "Sản phẩm rất tốt, tôi rất hài lòng!",
        "Đồ rác, lừa đảo, đừng mua!",
        "Giao hàng nhanh, đóng gói cẩn thận"
    ]
    predict_comments(comments)
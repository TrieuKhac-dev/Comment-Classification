import numpy as np

from src.app_setup import container
from src.pipelines.pipeline_config.logreg_pipeline_config import LogRegPipelineConfig
from src.models.classifier.base_classifier_context import BaseClassifierContextBuilder
from config.core.paths import paths
from src.utils.model_utils import get_latest_model_path


def predict_comments(comments):
    """
    Dự đoán nhãn cho danh sách bình luận bằng Logistic Regression.
    Tái sử dụng pipeline predict hiện tại (preprocess + extract + predict).
    """
    # Resolve các service đã đăng ký từ container
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
        .set_classifier(container.resolve("logreg_classifier"))
        .build()
    )
    context.X_pred_texts = comments

    # Tìm model Logistic Regression mới nhất
    model_path = get_latest_model_path(paths.LOGREG_DIR, "logreg", ".joblib")
    if not model_path:
        raise FileNotFoundError(f"No Logistic Regression model found in {paths.LOGREG_DIR}")

    print(f"Using model: {model_path}")

    # Tạo pipeline predict cho Logistic Regression
    predict_pipeline = LogRegPipelineConfig.predict_pipeline(
        model_path=model_path,
        return_proba=True,
        load_type="joblib",
    )

    # Chạy pipeline predict
    context = predict_pipeline.run(context)

    # Đọc và in kết quả
    y_pred = context.y_pred
    labels = np.argmax(y_pred, axis=1)
    print("Kết quả dự đoán cho comment:", comments)
    print("Dự đoán:", y_pred.round(2))
    for c, l, prob in zip(comments, labels, y_pred):
        print(f"Bình luận: {c}")
        print(f"  Xác suất: {prob}")
        print(f"  Nhãn dự đoán: {l} ({'vi phạm' if l == 1 else 'không vi phạm'})")


if __name__ == "__main__":
    comments = [
        "Sản phẩm rất tốt, tôi rất hài lòng!",
        "Đồ rác, lừa đảo, đừng mua!",
    ]
    predict_comments(comments)


from datetime import datetime

from src.app_setup import container
from src.pipelines.pipeline_config.logreg_pipeline_config import LogRegPipelineConfig
from src.models.classifier.base_classifier_context import BaseClassifierContextBuilder
from config.core.paths import paths
from config.training.data import data_config
from config.model.classifier import classifier_config


def main():
    # Tạo timestamp cho model và feature cache
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "logreg"
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_save_path = str(paths.LOGREG_DIR / model_filename)
    feature_cache_prefix = f"{model_name}_{timestamp}"
    processed_prefix = f"{model_name}_{timestamp}"

    # Lấy các service đã đăng ký từ container
    logger_service = container.resolve("logger_service")
    data_loader_service = container.resolve("data_loader_service")
    splitter_service = container.resolve("splitter_service")
    feature_cache_service = container.resolve("feature_cache_service")
    repository = container.resolve("repository")
    extractor_service = container.resolve("extractor_service")
    preprocessor_service = container.resolve("preprocessor_service")

    # Xây dựng context
    context = (
        BaseClassifierContextBuilder()
        .set_logger_service(logger_service)
        .set_data_loader_service(data_loader_service)
        .set_splitter_service(splitter_service)
        .set_preprocessor_service(preprocessor_service)
        .set_extractor_service(extractor_service)
        .set_feature_cache_service(feature_cache_service)
        .set_model_repository(repository)
        .set_classifier(container.resolve("logreg_classifier"))
        .set_model_save_path(model_save_path)
        .set_feature_cache_prefix(feature_cache_prefix)
        .build()
    )

    # Tạo pipeline train cho Logistic Regression (dùng các pipeline step chung)
    pipeline = LogRegPipelineConfig.train_pipeline(
        filepath="data/raw/train_1.csv",
        text_column=data_config.TEXT_COLUMN,
        label_columns=data_config.LABEL_COLUMNS,
        test_ratio=data_config.TEST_RATIO,
        val_ratio=data_config.VAL_RATIO,
        train_config=classifier_config.LOGREG_PARAMS,
        save_type="joblib",
        eval_metrics=["accuracy", "f1", "precision", "recall"],
        eval_average="weighted",
        cache_features=True,
        save_processed=True,
        processed_prefix=processed_prefix,
    )

    # Chạy pipeline
    pipeline.run(context)

    print(f"\nModel saved to: {model_save_path}")
    print(f"Feature cache prefix: {feature_cache_prefix}")
    print(f"Processed data cache prefix: {processed_prefix}")

    if hasattr(context, "metrics") and context.metrics and "evaluate" in context.metrics:
        print("\nKết quả đánh giá mô hình trên tập test:")
        for metric, value in context.metrics["evaluate"].items():
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()


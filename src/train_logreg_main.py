"""
Entry point để train Logistic Regression model.

Cho phép chọn file train (raw hoặc processed) qua terminal, tương tự evaluate_main.

Ví dụ:
    # Train với file raw mặc định
    python -m src.train_logreg_main

    # Train với file raw khác
    python -m src.train_logreg_main --data data/raw/train.csv

    # Train với file đã preprocess (ở data/processed)
    python -m src.train_logreg_main --data data/processed/processed_logreg_YYYYMMDD_train_xxx.csv
    # Train không split (dùng toàn bộ data làm train, không evaluate)
    python -m src.train_logreg_main --data data/raw/train.csv --no-split"""

import argparse
from datetime import datetime
from pathlib import Path

from src.app_setup import container
from src.pipelines.pipeline_config.logreg_pipeline_config import LogRegPipelineConfig
from src.models.classifier.base_classifier_context import BaseClassifierContextBuilder
from config.core.paths import paths
from config.training.data import data_config
from config.model.classifier import classifier_config


def main():
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression model with raw data or processed data (mutually exclusive)"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--data",
        type=str,
        help="Path to RAW training data file (Excel/CSV). Default: data/raw/train_1.csv",
    )
    group.add_argument(
        "--processed",
        type=str,
        help="Path to PROCESSED training data file (CSV UTF-8, created by SaveProcessedDataStep).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Do not split data into train/test, use all data as training set (no evaluation on test set).",
    )
    args = parser.parse_args()

    # Chọn chế độ train
    use_processed = args.processed is not None

    # Đường dẫn file train
    if use_processed:
        train_file = args.processed
    else:
        train_file = args.data or "data/raw/train_full.csv"

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training data file not found: {train_file}")

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

    if use_processed:
        logger_service.info(f"[Train LogReg] Using PROCESSED training data file: {train_file}")
    else:
        logger_service.info(f"[Train LogReg] Using RAW training data file: {train_file}")

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

    # Xác định test_ratio và val_ratio dựa trên --no-split
    test_ratio = 0.0 if args.no_split else data_config.TEST_RATIO
    val_ratio = 0.0 if args.no_split else data_config.VAL_RATIO

    # Tạo pipeline train cho Logistic Regression (dùng các pipeline step chung)
    if use_processed:
        pipeline = LogRegPipelineConfig.train_pipeline_with_processed(
            processed_file=train_file,
            text_column=data_config.TEXT_COLUMN,
            label_columns=data_config.LABEL_COLUMNS,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            train_config=classifier_config.LOGREG_PARAMS,
            save_type="joblib",
            eval_metrics=["accuracy", "f1", "precision", "recall"],
            eval_average="weighted",
            cache_features=True,
        )
    else:
        pipeline = LogRegPipelineConfig.train_pipeline(
            filepath=train_file,
            text_column=data_config.TEXT_COLUMN,
            label_columns=data_config.LABEL_COLUMNS,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
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

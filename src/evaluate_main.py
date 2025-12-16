"""
Entry point để đánh giá (evaluate) model đã train.

Hỗ trợ 2 modes:
1. Evaluate với raw data (qua preprocessing)
2. Evaluate với processed data (đã lưu sẵn - CSV UTF-8)

Usage:
    # Mode 1: Evaluate với file raw data
    python -m src.evaluate_main --data data/raw/test.xlsx
    
    # Mode 2: Evaluate với processed data file (CSV)
    python -m src.evaluate_main --processed data/processed/processed_lightgbm_20231215_test_abc123.csv
    
    # Evaluate với cache prefix tùy chỉnh
    python -m src.evaluate_main --data data/raw/test.xlsx --use-cache --cache-prefix my_eval
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from src.app_setup import container
from src.pipelines.pipeline_config.lightgbm_pipeline_config import LightGBMPipelineConfig
from src.models.classifier.base_classifier_context import BaseClassifierContextBuilder, BaseClassifierContext
from config.core.paths import paths
from config.training.data import data_config
from src.utils.model_utils import get_latest_model_path


def evaluate_with_raw_data(
    model_path: str,
    data_file_path: str,
    use_cache: bool = False,
    feature_cache_prefix: str = None
):
    """
    Đánh giá model với file data mới (test trên TOÀN BỘ samples, không split).
    
    Args:
        model_path: Đường dẫn đến model cần evaluate
        data_file_path: Đường dẫn đến file data mới (Excel/CSV)
        use_cache: Có sử dụng cache cho feature extraction không
        feature_cache_prefix: Prefix cho cache (nếu use_cache=True)
    """
    logger_service = container.resolve("logger_service")
    logger_service.info(f"=== Evaluate model with raw data (no split, test on all samples) ===")
    logger_service.info(f"Model: {model_path}")
    logger_service.info(f"Data file: {data_file_path}")
    
    # Validate data file
    if not Path(data_file_path).exists():
        raise FileNotFoundError(f"Data file not found at {data_file_path}")
    
    # Auto-generate cache prefix nếu cần
    if use_cache and not feature_cache_prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_cache_prefix = f"eval_{timestamp}"
        logger_service.info(f"Auto-generated cache prefix: {feature_cache_prefix}")
    
    # Lấy các service từ container
    data_loader_service = container.resolve("data_loader_service")
    feature_cache_service = container.resolve("feature_cache_service")
    repository = container.resolve("repository")
    extractor_service = container.resolve("extractor_service")
    preprocessor_service = container.resolve("preprocessor_service")
    
    # Xây dựng context
    builder = (
        BaseClassifierContextBuilder()
        .set_logger_service(logger_service)
        .set_data_loader_service(data_loader_service)
        .set_preprocessor_service(preprocessor_service)
        .set_extractor_service(extractor_service)
        .set_feature_cache_service(feature_cache_service)
        .set_model_repository(repository)
        .set_classifier(container.resolve("lightgbm_classifier"))
    )
    
    if feature_cache_prefix:
        builder.set_feature_cache_prefix(feature_cache_prefix)
    
    context = builder.build()
    
    # Tạo evaluation pipeline KHÔNG split (test toàn bộ data)
    pipeline = LightGBMPipelineConfig.evaluation_pipeline_no_split(
        filepath=data_file_path,
        text_column=data_config.TEXT_COLUMN,
        label_columns=data_config.LABEL_COLUMNS,
        model_path=model_path,
        provider_type="joblib",
        eval_metrics=["accuracy", "f1", "precision", "recall"],
        eval_average="weighted",
        use_cache=use_cache
    )
    
    # Chạy pipeline
    pipeline.run(context)
    
    logger_service.info("=== Evaluation completed ===")
    
    # In kết quả
    if hasattr(context, "metrics") and context.metrics and "evaluate" in context.metrics:
        print("\n" + "="*60)
        print("EVALUATION RESULTS - Full Dataset")
        print("="*60)
        for metric, value in context.metrics["evaluate"].items():
            print(f"  {metric:15s}: {value:.4f}")
        print("="*60 + "\n")
    
    return context


def evaluate_with_processed_data(
    model_path: str,
    processed_file_path: str,
    use_cache: bool = False,
    feature_cache_prefix: str = None
):
    """
    Đánh giá model với processed data file (đã qua preprocessing - CSV UTF-8).
    
    Args:
        model_path: Đường dẫn đến model cần evaluate
        processed_file_path: Đường dẫn đến processed data file (.csv)
        use_cache: Có sử dụng cache cho feature extraction không
        feature_cache_prefix: Prefix cho cache (nếu use_cache=True)
    """
    logger_service = container.resolve("logger_service")
    logger_service.info(f"=== Evaluate model with processed data ===")
    logger_service.info(f"Model: {model_path}")
    logger_service.info(f"Processed file: {processed_file_path}")
    
    # Validate processed file
    if not Path(processed_file_path).exists():
        raise FileNotFoundError(f"Processed file not found at {processed_file_path}")
    
    # Auto-generate cache prefix nếu cần
    if use_cache and not feature_cache_prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_cache_prefix = f"eval_{timestamp}"
        logger_service.info(f"Auto-generated cache prefix: {feature_cache_prefix}")
    
    # Lấy các service từ container
    feature_cache_service = container.resolve("feature_cache_service")
    repository = container.resolve("repository")
    
    # Xây dựng extractor service
    extractor_service = container.resolve("extractor_service")
    
    # Xây dựng context
    builder = (
        BaseClassifierContextBuilder()
        .set_logger_service(logger_service)
        .set_extractor_service(extractor_service)
        .set_feature_cache_service(feature_cache_service)
        .set_model_repository(repository)
        .set_classifier(container.resolve("lightgbm_classifier"))
    )
    
    if feature_cache_prefix:
        builder.set_feature_cache_prefix(feature_cache_prefix)
    
    context = builder.build()
    
    # Tạo pipeline evaluate với processed data
    pipeline = LightGBMPipelineConfig.evaluation_pipeline_with_processed(
        processed_file=processed_file_path,
        model_path=model_path,
        provider_type="joblib",
        eval_metrics=["accuracy", "f1", "precision", "recall"],
        eval_average="weighted",
        use_cache=use_cache
    )
    
    # Chạy pipeline
    pipeline.run(context)
    
    logger_service.info("=== Evaluation completed ===")
    
    # In kết quả
    if hasattr(context, "metrics") and context.metrics and "evaluate" in context.metrics:
        print("\n" + "="*60)
        print("EVALUATION RESULTS - Processed Dataset")
        print("="*60)
        for metric, value in context.metrics["evaluate"].items():
            print(f"  {metric:15s}: {value:.4f}")
        print("="*60 + "\n")
    
    return context


def main():
    """Entry point chính với argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model with raw data or processed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Mode 1: Evaluate với raw data file
        python -m src.evaluate_main --data data/raw/test.xlsx
        
        # Mode 2: Evaluate với processed data file (CSV UTF-8)
        python -m src.evaluate_main --processed data/processed/processed_lightgbm_20231215_test_abc123.csv
        
        # Evaluate với cache features
        python -m src.evaluate_main --data data/raw/test.xlsx --use-cache
        
        # Evaluate với cache prefix tùy chỉnh
        python -m src.evaluate_main --data data/raw/test.xlsx --use-cache --cache-prefix my_eval
        """
    )
    
    # Arguments - chọn 1 trong 2: --data (raw) hoặc --processed
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", type=str, 
                       help="Path to raw data file (Excel/CSV)")
    group.add_argument("--processed", type=str,
                       help="Path to processed data file (.csv UTF-8)")
    
    parser.add_argument("--model", type=str, help="Model path (default: latest model)")
    parser.add_argument("--use-cache", action="store_true", 
                       help="Enable feature caching")
    parser.add_argument("--cache-prefix", type=str, 
                       help="Feature cache prefix (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    # Auto-detect model path nếu không được chỉ định
    if not args.model:
        model_path = get_latest_model_path(paths.LIGHTGBM_DIR, "lightgbm", ".joblib")
        if not model_path:
            print(f"Error: No model found in {paths.LIGHTGBM_DIR}")
            sys.exit(1)
        print(f"Auto-detected latest model: {model_path}")
    else:
        model_path = args.model
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Chạy evaluation theo mode
    try:
        if args.data:
            # Mode 1: Raw data
            evaluate_with_raw_data(
                model_path=model_path,
                data_file_path=args.data,
                use_cache=args.use_cache,
                feature_cache_prefix=args.cache_prefix
            )
        elif args.processed:
            # Mode 2: Processed data
            evaluate_with_processed_data(
                model_path=model_path,
                processed_file_path=args.processed,
                use_cache=args.use_cache,
                feature_cache_prefix=args.cache_prefix
            )
    
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

from typing import Optional

from src.pipelines import Pipeline
from src.pipelines.pipeline_step import (
    LoadClassifierStep,
    PredictStep,
    EvaluateStep,
    LoadDataStep,
    SplitDataStep,
    PreprocessTextStep,
    ExtractFeatureStep,
    TrainStep,
    SaveClassifierStep,
    CopyDataStep,
    SaveProcessedDataStep,
    LoadProcessedDataStep,
)


class LightGBMPipelineConfig:
    @staticmethod
    def train_pipeline(
        filepath: str,
        text_column: str,
        label_columns: list[str],
        test_ratio: float = 0.2,
        val_ratio: float = 0.0,
        train_config: Optional[dict] = None,
        save_type: str = "joblib",
        eval_metrics: list[str] = None,
        eval_average: str = "weighted",
        cache_features: bool = True,
        save_processed: bool = False,
        processed_prefix: Optional[str] = None
    ) -> Pipeline:
        steps = [
            LoadDataStep(
                filepath=filepath,
                text_column=text_column,
                label_columns=label_columns,
            ),
            SplitDataStep(
                test_ratio=test_ratio,
                val_ratio=val_ratio,
            ),
            PreprocessTextStep(),
        ]
        
        # Thêm SaveProcessedDataStep nếu save_processed=True
        if save_processed and processed_prefix:
            steps.append(
                SaveProcessedDataStep(
                    prefix=processed_prefix,
                    text_column=text_column,
                    label_columns=label_columns,
                    save_train=True,
                    save_test=True,
                    save_val=(val_ratio > 0)
                )
            )
        
        # Tiếp tục với các bước còn lại
        steps.extend([
            ExtractFeatureStep(
                use_cache=cache_features,
            ),
            TrainStep(config=train_config),
            SaveClassifierStep(
                save_type=save_type,
            ),
            # Thêm bước đánh giá ngay sau khi train và lưu model
            EvaluateStep(
                metrics=eval_metrics or ["accuracy", "f1", "precision", "recall"],
                average=eval_average
            ),
        ])
        return Pipeline(steps)

    @staticmethod
    def predict_pipeline(
        model_path: str,
        return_proba: bool = True,
        load_type: str = "joblib"
    ) -> Pipeline:
        steps = [
            LoadClassifierStep(model_path=model_path, load_type=load_type),
            PreprocessTextStep(
                process_pred=True,
                process_train=False,
                process_test=False,
                process_val=False
            ),
            ExtractFeatureStep(
                extract_pred=True,
                extract_train=False,
                extract_test=False,
                extract_val=False
            ),
            PredictStep(return_proba=return_proba),
        ]
        return Pipeline(steps)
    
    @staticmethod
    def evaluation_pipeline_no_split(
        filepath: str,
        text_column: str,
        label_columns: list[str],
        model_path: str = None,
        provider_type: str = "joblib",
        eval_metrics: list[str] = None,
        eval_average: str = "weighted",
        use_cache: bool = False
    ) -> Pipeline:
        """
        Tạo pipeline để evaluate model trên TOÀN BỘ dataset (không split).
        Dùng khi muốn đánh giá model trên toàn bộ samples của file data mới.
        
        Args:
            filepath: Đường dẫn file data
            text_column: Tên cột chứa text
            label_columns: Danh sách tên cột labels
            model_path: Đường dẫn model đã train
            provider_type: Loại provider để load model (joblib/lightgbm)
            eval_metrics: Metrics để evaluate
            eval_average: Average mode cho metrics
            use_cache: Có sử dụng feature cache không
        
        Returns:
            Pipeline đã được cấu hình
        """
        steps = [
            LoadDataStep(
                filepath=filepath,
                text_column=text_column,
                label_columns=label_columns,
            ),
            # Copy texts -> X_pred_texts và labels_df -> y_pred để xử lý như prediction data
            CopyDataStep(
                from_X="texts",
                to_X="X_pred_texts",
                from_y="labels_df",
                to_y="y_pred"
            ),
            PreprocessTextStep(
                process_train=False,
                process_test=False,
                process_val=False,
                process_pred=True  # Xử lý như prediction data
            ),
            ExtractFeatureStep(
                use_cache=use_cache,
                extract_train=False,
                extract_test=False,
                extract_val=False,
                extract_pred=True  # Extract như prediction data
            ),
            LoadClassifierStep(
                model_path=model_path,
                load_type=provider_type,
            ),
            EvaluateStep(
                metrics=eval_metrics or ["accuracy", "f1", "precision", "recall"],
                average=eval_average
            ),
        ]
        return Pipeline(steps)
    
    @staticmethod
    def evaluation_pipeline_with_processed(
        processed_file: str,
        model_path: str = None,
        provider_type: str = "joblib",
        eval_metrics: list[str] = None,
        eval_average: str = "weighted",
        use_cache: bool = False
    ) -> Pipeline:
        """
        Tạo pipeline để evaluate model với processed data file (CSV UTF-8).
        Bỏ qua preprocessing step, load trực tiếp từ processed cache.
        
        Args:
            processed_file: Đường dẫn file processed cache (.csv)
                           Ví dụ: data/processed/processed_lightgbm_20231215_test_abc123.csv
            model_path: Đường dẫn model đã train
            provider_type: Loại provider để load model (joblib/lightgbm)
            eval_metrics: Metrics để evaluate
            eval_average: Average mode cho metrics
            use_cache: Có sử dụng feature cache không
        
        Returns:
            Pipeline đã được cấu hình
        """
        steps = [
            # Load processed data trực tiếp từ cache (đã qua preprocessing)
            LoadProcessedDataStep(cache_file=processed_file),
            # Extract features từ X_pred_processed
            ExtractFeatureStep(
                use_cache=use_cache,
                extract_train=False,
                extract_test=False,
                extract_val=False,
                extract_pred=True  # Extract từ X_pred_processed
            ),
            LoadClassifierStep(
                model_path=model_path,
                load_type=provider_type,
            ),
            EvaluateStep(
                metrics=eval_metrics or ["accuracy", "f1", "precision", "recall"],
                average=eval_average
            ),
        ]
        return Pipeline(steps)

    @staticmethod
    def train_pipeline_with_processed(
        processed_file: str,
        text_column: str,
        label_columns: list[str],
        train_config: Optional[dict] = None,
        save_type: str = "joblib",
        eval_metrics: list[str] = None,
        eval_average: str = "weighted",
        cache_features: bool = True,
    ) -> Pipeline:
        """
        Tạo pipeline để train LightGBM từ processed data file (CSV UTF-8).
        Bỏ qua LoadDataStep + PreprocessTextStep, dùng trực tiếp processed texts.

        Toàn bộ dữ liệu trong file được dùng làm tập train; EvaluateStep sẽ
        evaluate trên cùng tập này (dựa trên X_pred_features & y_pred).
        """
        steps = [
            # Load processed data trực tiếp từ cache (đã qua preprocessing)
            LoadProcessedDataStep(cache_file=processed_file),
            # Extract features từ X_pred_processed
            ExtractFeatureStep(
                use_cache=cache_features,
                extract_train=False,
                extract_test=False,
                extract_val=False,
                extract_pred=True,
            ),
            # Copy features/labels từ pred sang train để TrainStep sử dụng
            CopyDataStep(
                from_X="X_pred_features",
                to_X="X_train_features",
                from_y="y_pred",
                to_y="y_train",
            ),
            TrainStep(config=train_config),
            SaveClassifierStep(
                save_type=save_type,
            ),
            EvaluateStep(
                metrics=eval_metrics or ["accuracy", "f1", "precision", "recall"],
                average=eval_average,
            ),
        ]
        return Pipeline(steps)
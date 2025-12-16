"""
Pipeline step để lưu dữ liệu đã qua preprocessing.
Lưu X_train_processed, X_test_processed vào cache với ProcessedDataCacheService.
"""

from src.interfaces.pipeline import IPipelineStep
from src.models.classifier.base_classifier_context import BaseClassifierContext
from src.services.cache.processed_data_cache_service import ProcessedDataCacheService


class SaveProcessedDataStep(IPipelineStep):
    """
    Lưu dữ liệu đã qua preprocessing vào cache.
    Hỗ trợ lưu train, test, val datasets.
    """
    
    def __init__(
        self,
        prefix: str,
        text_column: str = "processed_text",
        label_columns: list = None,
        save_train: bool = True,
        save_test: bool = True,
        save_val: bool = False
    ):
        """
        Khởi tạo step.
        
        Args:
            prefix: Prefix cho cache (vd: lightgbm_20231215_143025)
            text_column: Tên cột cho processed text (mặc định: processed_text)
            label_columns: Danh sách tên cột labels (mặc định: None sẽ dùng label_0, label_1...)
            save_train: Có lưu train set không
            save_test: Có lưu test set không
            save_val: Có lưu validation set không
        """
        self.prefix = prefix
        self.text_column = text_column
        self.label_columns = label_columns
        self.save_train = save_train
        self.save_test = save_test
        self.save_val = save_val
        self.cache_service = ProcessedDataCacheService()
    
    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        """
        Lưu processed data vào cache.
        
        Args:
            context: Context chứa processed data
        
        Returns:
            Context không thay đổi (chỉ lưu cache)
        """
        saved_count = 0
        
        # Lưu train set
        if self.save_train and hasattr(context, 'X_train_processed') and context.X_train_processed is not None:
            if hasattr(context, 'texts') and context.texts is not None:
                # Kiểm tra y_train
                y_train = getattr(context, 'y_train', None)
                if context.logger_service:
                    context.logger_service.info(
                        f"SaveProcessedDataStep | Train set - "
                        f"X_train_processed: {len(context.X_train_processed)}, "
                        f"y_train type: {type(y_train)}, "
                        f"y_train shape: {y_train.shape if hasattr(y_train, 'shape') else 'N/A'}"
                    )
                
                # Tạo key từ texts gốc
                key = self.cache_service.make_cache_key(context.texts)
                self.cache_service.save(
                    processed_texts=context.X_train_processed,
                    labels_df=y_train,
                    key=key,
                    prefix=self.prefix,
                    dataset_type='train',
                    text_column=self.text_column,
                    label_columns=self.label_columns
                )
                saved_count += 1
                if context.logger_service:
                    context.logger_service.info(
                        f"SaveProcessedDataStep | Saved train set ({len(context.X_train_processed)} samples)"
                    )
        
        # Lưu test set
        if self.save_test and hasattr(context, 'X_test_processed') and context.X_test_processed is not None:
            if hasattr(context, 'texts') and context.texts is not None:
                # Kiểm tra y_test
                y_test = getattr(context, 'y_test', None)
                if context.logger_service:
                    context.logger_service.info(
                        f"SaveProcessedDataStep | Test set - "
                        f"X_test_processed: {len(context.X_test_processed)}, "
                        f"y_test type: {type(y_test)}, "
                        f"y_test shape: {y_test.shape if hasattr(y_test, 'shape') else 'N/A'}"
                    )
                
                key = self.cache_service.make_cache_key(context.texts)
                self.cache_service.save(
                    processed_texts=context.X_test_processed,
                    labels_df=y_test,
                    key=key,
                    prefix=self.prefix,
                    dataset_type='test',
                    text_column=self.text_column,
                    label_columns=self.label_columns
                )
                saved_count += 1
                if context.logger_service:
                    context.logger_service.info(
                        f"SaveProcessedDataStep | Saved test set ({len(context.X_test_processed)} samples)"
                    )
        
        # Lưu validation set
        if self.save_val and hasattr(context, 'X_val_processed') and context.X_val_processed is not None:
            if hasattr(context, 'texts') and context.texts is not None:
                key = self.cache_service.make_cache_key(context.texts)
                self.cache_service.save(
                    processed_texts=context.X_val_processed,
                    labels_df=context.y_val,
                    key=key,
                    prefix=self.prefix,
                    dataset_type='val',
                    text_column=self.text_column,
                    label_columns=self.label_columns
                )
                saved_count += 1
                if context.logger_service:
                    context.logger_service.info(
                        f"SaveProcessedDataStep | Saved val set ({len(context.X_val_processed)} samples)"
                    )
        
        if context.logger_service and saved_count == 0:
            context.logger_service.warning("SaveProcessedDataStep | No data was saved")
        
        return context

from src.interfaces.pipeline import IPipelineStep
from src.models.classifier.base_classifier_context import BaseClassifierContext

class PreprocessTextStep(IPipelineStep):
    def __init__(
        self,
        process_pred: bool = True,
        process_train: bool = True,
        process_test: bool = True,
        process_val: bool = True
    ):
        self.process_pred = process_pred
        self.process_train = process_train
        self.process_test = process_test
        self.process_val = process_val

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if context.preprocessor_service is None:
            raise ValueError("PreprocessService is not provided.")

        # Process prediction set
        if self.process_pred:
            if context.X_pred_texts is not None:
                context.X_pred_processed = context.preprocessor_service.preprocess_batch(context.X_pred_texts)
            else:
                context.X_pred_processed = None
                if context.logger_service:
                    context.logger_service.warning("No prediction data available for preprocessing.")

        # Process train set
        if self.process_train:
            if context.X_train_texts is not None:
                context.X_train_processed = context.preprocessor_service.preprocess_batch(context.X_train_texts)
            else:
                context.X_train_processed = None
                if context.logger_service:
                    context.logger_service.warning("No train data available for preprocessing.")

        # Process test set
        if self.process_test:
            if context.X_test_texts is not None:
                context.X_test_processed = context.preprocessor_service.preprocess_batch(context.X_test_texts)
            else:
                context.X_test_processed = None
                if context.logger_service:
                    context.logger_service.warning("No test data available for preprocessing.")

        # Process validation set
        if self.process_val:
            if context.X_val_texts is not None:
                context.X_val_processed = context.preprocessor_service.preprocess_batch(context.X_val_texts)
            else:
                context.X_val_processed = None
                if context.logger_service:
                    context.logger_service.warning("No validation data available for preprocessing.")

        if context.logger_service:
            train_count = len(context.X_train_processed) if context.X_train_processed is not None else 0
            test_count = len(context.X_test_processed) if context.X_test_processed is not None else 0
            val_count = len(context.X_val_processed) if context.X_val_processed is not None else 0
            context.logger_service.info(
                f"PreprocessTextStep | Preprocessed "
                f"train={train_count}, "
                f"test={test_count}, "
                f"val={val_count} samples"
            )

        return context
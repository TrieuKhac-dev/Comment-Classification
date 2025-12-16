from src.interfaces.pipeline import IPipelineStep
from src.models.classifier.base_classifier_context import BaseClassifierContext

class SplitDataStep(IPipelineStep):
    def __init__(
        self,
        test_ratio: float = 0.2,
        val_ratio: float = 0.0,
        random_seed: int = 42,
    ):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if context.splitter_service is None:
            raise ValueError("SplitterService is not provided")

        split_data = context.splitter_service.split(
            texts=context.texts,
            labels=context.labels_df,
            test_ratio=self.test_ratio,
            val_ratio=self.val_ratio,
            random_seed=self.random_seed,
        )

        context.X_train_texts = split_data["train"]["texts"]
        context.y_train = split_data["train"]["labels"]

        context.X_test_texts = split_data["test"]["texts"]
        context.y_test = split_data["test"]["labels"]

        if "val" in split_data:
            context.X_val_texts = split_data["val"]["texts"]
            context.y_val = split_data["val"]["labels"]

        if context.logger_service:
            context.logger_service.info(
                f"SplitDataStep | "
                f"train={len(context.X_train_texts)}, "
                f"test={len(context.X_test_texts)}"
                + (f", val={len(context.X_val_texts)}" if context.X_val_texts else "")
            )
               
        return context

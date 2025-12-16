from pathlib import Path
from typing import Union

from src.interfaces.pipeline import IPipelineStep
from src.models.classifier.base_classifier_context import BaseClassifierContext

class LoadDataStep(IPipelineStep):
    def __init__(
        self,
        filepath: Union[str, Path],
        text_column: str,
        label_columns: list[str],
    ):
        self.filepath = filepath
        self.text_column = text_column
        self.label_columns = label_columns

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if context.data_loader_service is None:
            raise ValueError("DataLoaderService is not provided")
        
        df = context.data_loader_service.load_csv(
            filepath=self.filepath,
            text_column=self.text_column,
            label_columns=self.label_columns,
        )

        texts, labels_df = context.data_loader_service.get_text_and_labels(
            df=df,
            text_column=self.text_column,
            label_columns=self.label_columns,
        )

        context.texts = texts
        context.labels_df = labels_df

        if context.logger_service:
            context.logger_service.info(
                f"LoadDataStep | Loaded {len(texts)} samples"
            )

        return context

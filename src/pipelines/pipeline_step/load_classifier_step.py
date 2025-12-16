from typing import Optional
from config import model
from src.interfaces.pipeline import IPipelineStep
from src.models.classifier.base_classifier_context import BaseClassifierContext

class LoadClassifierStep(IPipelineStep):
    def __init__(
        self,
        model_path: str,
        load_type: str = "joblib",
    ):
        self.model_path = model_path
        self.load_type = load_type

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if(context.classifier is None):
            raise ValueError("Classifier is not initialized")

        context.classifier.load(
            repository=context.model_repository,
            model_path=self.model_path,
            load_type=self.load_type,
        )

        if context.logger_service:
            context.logger_service.info(
                f"LoadClassifierStep | "
                f"classifier={context.classifier.__class__.__name__}, "
                f"path={self.model_path}, "
                f"type={self.load_type}"
            )

        return context

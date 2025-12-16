from src.interfaces.pipeline import IPipelineStep
from src.models.classifier.base_classifier_context import BaseClassifierContext

class SaveClassifierStep(IPipelineStep):
    def __init__(
        self,
        save_type: str = "joblib",
    ):
        self.save_type = save_type

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if context.classifier is None:
            raise ValueError("Classifier is not initialized")

        if context.model_repository is None:
            raise ValueError("ClassifierRepository is not provided")

        if context.model_save_path is None:
            raise ValueError("model_save_path is not set in context")

        context.classifier.save(
            repository=context.model_repository,
            model_path=context.model_save_path,
            save_type=self.save_type,
        )

        if context.logger_service:
            context.logger_service.info(
                "SaveModelStep | "
                f"classifier={context.classifier.__class__.__name__}, "
                f"path={context.model_save_path}, "
                f"type={self.save_type}"
            )

        return context

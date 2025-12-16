import numpy as np

from src.interfaces.pipeline import IPipelineStep
from src.models.classifier.base_classifier_context import BaseClassifierContext

class PredictStep(IPipelineStep):
    def __init__(
        self,
        return_proba: bool = False,
    ):
        self.return_proba = return_proba

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if context.classifier is None:
            raise ValueError("Classifier is not initialized")
        
        y_pred = context.classifier.predict(context.X_pred_features, self.return_proba)

        context.y_pred = y_pred

        if context.logger_service:
            context.logger_service.info(
                f"PredictStep | "
                f"samples={len(y_pred)}, "
                f"return_proba={self.return_proba}"
            )

        return context

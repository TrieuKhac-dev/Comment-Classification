from typing import List, Optional

from src.models.classifier.base_classifier_context import BaseClassifierContext
from src.interfaces.pipeline import IPipelineStep

class EvaluateStep(IPipelineStep):
    def __init__(self, metrics: Optional[List[str]] = None, average: str = "weighted"):
        self.metrics = metrics
        self.average = average

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if context.classifier is None:
            raise RuntimeError("Classifier is not trained")
        
        if context.metrics is None:
            context.metrics = {}
        
        # Trường hợp 1: Evaluate test set (có split)
        if context.X_test_features is not None and context.y_test is not None:
            if context.logger_service:
                context.logger_service.info("EvaluateStep | Evaluating on test set...")
            
            eval_metrics = context.classifier.evaluate(
                X=context.X_test_features,
                y=context.y_test,
                metrics=self.metrics,
                average=self.average
            )
            context.metrics["evaluate"] = eval_metrics
            
            if context.logger_service:
                context.logger_service.info(f"EvaluateStep | Test metrics: {eval_metrics}")
        
        # Trường hợp 2: Evaluate validation set (nếu có)
        elif context.X_val_features is not None and context.y_val is not None:
            if context.logger_service:
                context.logger_service.info("EvaluateStep | Evaluating on validation set...")
            
            eval_metrics = context.classifier.evaluate(
                X=context.X_val_features,
                y=context.y_val,
                metrics=self.metrics,
                average=self.average
            )
            context.metrics["evaluate"] = eval_metrics
            
            if context.logger_service:
                context.logger_service.info(f"EvaluateStep | Validation metrics: {eval_metrics}")
        
        # Trường hợp 3: Evaluate toàn bộ data (không split - dùng pred data)
        elif context.X_pred_features is not None and context.y_pred is not None:
            if context.logger_service:
                context.logger_service.info("EvaluateStep | Evaluating on full dataset...")
            
            eval_metrics = context.classifier.evaluate(
                X=context.X_pred_features,
                y=context.y_pred,
                metrics=self.metrics,
                average=self.average
            )
            context.metrics["evaluate"] = eval_metrics
            
            if context.logger_service:
                context.logger_service.info(f"EvaluateStep | Full dataset metrics: {eval_metrics}")
        
        else:
            raise RuntimeError("No data available for evaluation. Need either test, validation, or prediction data with labels.")

        return context
    
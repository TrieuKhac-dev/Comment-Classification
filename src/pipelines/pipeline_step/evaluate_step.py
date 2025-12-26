from typing import List, Optional

from src.models.classifier.base_classifier_context import BaseClassifierContext
from src.interfaces.pipeline import IPipelineStep

class EvaluateStep(IPipelineStep):
    def __init__(self, metrics: Optional[List[str]] = None, average: str = "weighted", threshold: Optional[float] = None):
        self.metrics = metrics
        self.average = average
        self.threshold = threshold

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if context.classifier is None:
            raise RuntimeError("Classifier is not trained")
        
        if context.metrics is None:
            context.metrics = {}
        
        # Helper: compute thresholded metrics if threshold provided
        def _compute_threshold_metrics(X, y_true, thr):
            # Compute probabilities, take class 1
            y_probs = context.classifier.predict(X, return_proba=True)
            y_prob1 = y_probs[:, 1]
            y_pred_thr = (y_prob1 >= thr).astype(int)
            # Compute metrics using classifier.evaluate-like metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            return {
                "accuracy": accuracy_score(y_true, y_pred_thr),
                "f1": f1_score(y_true, y_pred_thr, average=self.average),
                "precision": precision_score(y_true, y_pred_thr, average=self.average),
                "recall": recall_score(y_true, y_pred_thr, average=self.average),
            }

        # Trường hợp 1: Evaluate test set (có split)
        if (context.X_test_features is not None and context.y_test is not None and 
            len(context.X_test_features) > 0 and len(context.y_test) > 0):
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

            # Nếu có threshold, tính thêm metrics theo ngưỡng
            if self.threshold is not None:
                thr_key = f"evaluate_threshold_{str(self.threshold).replace('.', '_')}"
                try:
                    thr_metrics = _compute_threshold_metrics(context.X_test_features, context.y_test, self.threshold)
                    context.metrics[thr_key] = thr_metrics
                    if context.logger_service:
                        context.logger_service.info(f"EvaluateStep | Test threshold metrics ({self.threshold}): {thr_metrics}")
                except Exception:
                    if context.logger_service:
                        context.logger_service.exception("EvaluateStep | Error computing threshold metrics on test set")

        # Trường hợp 2: Evaluate validation set (nếu có)
        elif (context.X_val_features is not None and context.y_val is not None and
              len(context.X_val_features) > 0 and len(context.y_val) > 0):
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

            if self.threshold is not None:
                thr_key = f"evaluate_threshold_{str(self.threshold).replace('.', '_')}"
                try:
                    thr_metrics = _compute_threshold_metrics(context.X_val_features, context.y_val, self.threshold)
                    context.metrics[thr_key] = thr_metrics
                    if context.logger_service:
                        context.logger_service.info(f"EvaluateStep | Validation threshold metrics ({self.threshold}): {thr_metrics}")
                except Exception:
                    if context.logger_service:
                        context.logger_service.exception("EvaluateStep | Error computing threshold metrics on validation set")

        # Trường hợp 3: Evaluate toàn bộ data (không split - dùng pred data)
        elif (context.X_pred_features is not None and context.y_pred is not None and
              len(context.X_pred_features) > 0 and len(context.y_pred) > 0):
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

            if self.threshold is not None:
                thr_key = f"evaluate_threshold_{str(self.threshold).replace('.', '_')}"
                try:
                    thr_metrics = _compute_threshold_metrics(context.X_pred_features, context.y_pred, self.threshold)
                    context.metrics[thr_key] = thr_metrics
                    if context.logger_service:
                        context.logger_service.info(f"EvaluateStep | Full dataset threshold metrics ({self.threshold}): {thr_metrics}")
                except Exception:
                    if context.logger_service:
                        context.logger_service.exception("EvaluateStep | Error computing threshold metrics on full dataset")

        else:
            if context.logger_service:
                context.logger_service.warning("EvaluateStep | No data available for evaluation (test/val/pred empty or None). Skipping evaluation.")
            # Không raise error, chỉ skip

        return context
    
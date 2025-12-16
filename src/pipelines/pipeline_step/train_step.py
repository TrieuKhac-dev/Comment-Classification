from typing import Optional, Dict

from src.interfaces.pipeline import IPipelineStep
from src.interfaces.logging import ILoggerService
from src.models.classifier.base_classifier_context import BaseClassifierContext

class TrainStep(IPipelineStep):
    def __init__(
        self,
        config: Optional[Dict] = None,
        sample_weight: Optional[object] = None,
    ):
        self.config = config
        self.sample_weight = sample_weight

    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        if context.classifier is None:
            raise RuntimeError("classifier is not set in context")

        if context.X_train_features is None or context.y_train is None:
            raise RuntimeError("Training data (X_train_features / y_train) is missing")

        self._pre_train_log(
            logger=context.logger_service,
            context=context,
            config=self.config,
            sample_weight=self.sample_weight,
        )

        context.classifier.train(
            X=context.X_train_features,
            y=context.y_train,
            config=self.config,
            sample_weight=self.sample_weight,
        )
        
        if context.logger_service:
            self.trained_log(
                logger=context.logger_service,
                context=context,
            )
            
        return context
    
    def _pre_train_log(self, logger: ILoggerService, context: BaseClassifierContext, config: Optional[Dict], sample_weight: Optional[object]) -> None:
        if logger:
            logger.info("========== TrainStep START ==========")
            logger.info(f"Model class       : {context.classifier.__class__.__name__}")
            logger.info(f"Train samples     : {context.X_train_features.shape[0]}")
            logger.info(f"Feature dimension : {context.X_train_features.shape[1]}")
            logger.info(f"Labels shape      : {context.y_train.shape}")
            logger.info(f"Use sample_weight : {self.sample_weight is not None}")
            logger.info(f"Train config      : {self.config}")

            if context.X_val_features is not None and context.y_val is not None:
                logger.info(
                    f"Validation set: "
                    f"samples={context.X_val_features.shape[0]}"
                )
            else:
                logger.info("Validation set: None")

    def trained_log(self, logger: ILoggerService, context: BaseClassifierContext) -> None:
        if logger:
            model = getattr(context.classifier, "model", None)

            if model is not None:
                try:
                    logger.info(
                        f"Trained estimators: "
                        f"{getattr(model, 'n_estimators_', 'N/A')}"
                    )
                except Exception:
                    pass

            logger.info("TrainStep | Training completed successfully")
            logger.info("========== TrainStep END ==========")
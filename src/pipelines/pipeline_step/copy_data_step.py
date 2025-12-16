"""
Step để copy data từ một attribute context sang attribute khác.
Dùng khi cần chuyển đổi data format trong pipeline.
"""

from src.models.classifier.base_classifier_context import BaseClassifierContext
from src.interfaces.pipeline import IPipelineStep


class CopyDataStep(IPipelineStep):
    """
    Copy data từ một attribute sang attribute khác trong context.
    Ví dụ: Copy X_raw -> X_pred_raw để xử lý như prediction data.
    """
    
    def __init__(
        self,
        from_X: str,
        to_X: str,
        from_y: str = None,
        to_y: str = None
    ):
        """
        Khởi tạo copy step.
        
        Args:
            from_X: Tên attribute nguồn cho features (vd: "X_raw")
            to_X: Tên attribute đích cho features (vd: "X_pred_raw")
            from_y: Tên attribute nguồn cho labels (vd: "y_raw")
            to_y: Tên attribute đích cho labels (vd: "y_pred")
        """
        self.from_X = from_X
        self.to_X = to_X
        self.from_y = from_y
        self.to_y = to_y
    
    def run(self, context: BaseClassifierContext) -> BaseClassifierContext:
        """
        Copy data giữa các attributes trong context.
        
        Args:
            context: Context chứa data cần copy
        
        Returns:
            Context đã được copy data
        """
        # Copy features (X)
        if hasattr(context, self.from_X):
            source_X = getattr(context, self.from_X)
            setattr(context, self.to_X, source_X)
            
            count = len(source_X) if source_X is not None else 0
            if context.logger_service and count > 0:
                context.logger_service.info(
                    f"CopyDataStep | Copied {self.from_X} -> {self.to_X} ({count} samples)"
                )
        
        # Copy labels (y) nếu được chỉ định
        if self.from_y and self.to_y and hasattr(context, self.from_y):
            source_y = getattr(context, self.from_y)
            setattr(context, self.to_y, source_y)
            
            if context.logger_service:
                context.logger_service.info(
                    f"CopyDataStep | Copied {self.from_y} -> {self.to_y} "
                    f"(shape={source_y.shape if hasattr(source_y, 'shape') else len(source_y)})"
                )
        
        return context

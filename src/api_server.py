from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import numpy as np
from contextlib import asynccontextmanager

from src.app_setup import container
from src.pipelines.pipeline_config.lightgbm_pipeline_config import LightGBMPipelineConfig
from src.models.classifier.base_classifier_context import BaseClassifierContextBuilder
from config.core.paths import paths
from src.utils.model_utils import get_latest_model_path


# Biến global để lưu trữ pipeline và services
prediction_pipeline = None
extractor_service = None
preprocessor_service = None
logger_service = None
feature_cache_service = None
repository = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_pipeline, extractor_service, preprocessor_service
    global logger_service, feature_cache_service, repository
    
    # Startup: Load model và services
    print("Initializing Comment Classification API...")
    
    # Lấy các service đã đăng ký từ container
    logger_service = container.resolve("logger_service")
    feature_cache_service = container.resolve("feature_cache_service")
    repository = container.resolve("repository")
    extractor_service = container.resolve("extractor_service")
    preprocessor_service = container.resolve("preprocessor_service")
    
    # Tìm model mới nhất
    model_path = get_latest_model_path(paths.LIGHTGBM_DIR, "lightgbm", ".joblib")
    if not model_path:
        raise FileNotFoundError(f"No LightGBM model found in {paths.LIGHTGBM_DIR}")
    
    logger_service.info(f"Loading model: {model_path}")
    
    # Tạo prediction pipeline
    prediction_pipeline = LightGBMPipelineConfig.predict_pipeline(
        model_path=model_path,
        return_proba=True
    )
    
    logger_service.info("API ready")
    print("Server started up successfully")
    
    yield
    
    # Shutdown: Cleanup
    print("Server is shutting down...")
    logger_service.info("Server has shut down")


# Khởi tạo FastAPI app
app = FastAPI(
    title="Comment Classification API",
    description="API for classifying comments as violating or non-violating comments",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic models cho request/response
class CommentRequest(BaseModel):
    """Request model chứa danh sách bình luận cần phân loại"""
    comments: Dict[str, str] = Field(
        ...,
        description="Dictionary with keys as IDs and values as comment texts",
        example={
            "1": "Sản phẩm rất tốt, tôi rất hài lòng!",
            "2": "Đồ rác, lừa đảo, đừng mua!"
        }
    )


class PredictionResult(BaseModel):
    """Kết quả dự đoán cho một bình luận"""
    is_violation: bool = Field(
        ...,
        description="True if the comment is violating, False otherwise"
    )
    violation_probability: float = Field(
        ...,
        description="Probability that the comment is violating (0.0 - 1.0)",
        ge=0.0,
        le=1.0
    )
    comment: str = Field(
        ...,
        description="Original comment text"
    )


class CommentResponse(BaseModel):
    """Response model chứa kết quả phân loại"""
    results: Dict[str, PredictionResult] = Field(
        ...,
        description="Dictionary with keys as IDs and values as prediction results"
    )
    total_comments: int = Field(
        ...,
        description="Total number of comments classified"
    )
    violation_count: int = Field(
        ...,
        description="Number of violating comments"
    )


def predict_batch(comments: List[str]) -> tuple:
    """
    Dự đoán batch các bình luận
    
    Args:
        comments: Danh sách bình luận cần phân loại
        
    Returns:
        tuple: (labels, probabilities) 
            - labels: np.ndarray - Nhãn dự đoán (0: không vi phạm, 1: vi phạm)
            - probabilities: np.ndarray - Xác suất vi phạm cho mỗi bình luận
    """
    # Xây dựng context cho pipeline predict
    context = (
        BaseClassifierContextBuilder()
        .set_logger_service(logger_service)
        .set_preprocessor_service(preprocessor_service)
        .set_extractor_service(extractor_service)
        .set_feature_cache_service(feature_cache_service)
        .set_model_repository(repository)
        .set_classifier(container.resolve("lightgbm_classifier"))
        .build()
    )
    context.X_pred_texts = comments

    # Chạy pipeline predict
    context = prediction_pipeline.run(context)

    # Lấy kết quả
    y_pred_proba = context.y_pred  # Shape: (n_samples, 2) - [prob_class_0, prob_class_1]
    
    # Xác suất vi phạm là xác suất của class 1
    violation_probs = y_pred_proba[:, 1]
    
    # Nhãn dự đoán
    labels = np.argmax(y_pred_proba, axis=1)
    
    return labels, violation_probs


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Comment Classification API",
        "version": "1.0.0",
        "message": "API is running. Use POST /predict to classify comments."
    }


@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái API"""
    return {
        "status": "healthy",
        "model_loaded": prediction_pipeline is not None,
        "services_ready": all([
            extractor_service is not None,
            preprocessor_service is not None,
            logger_service is not None
        ])
    }


@app.post("/predict", response_model=CommentResponse)
async def predict_comments_endpoint(request: CommentRequest):
    """
    Phân loại bình luận vi phạm/không vi phạm
    
    **Request Body:**
    ```json
    {
        "comments": {
            "1": "Sản phẩm rất tốt!",
            "2": "Đồ rác, lừa đảo!"
        }
    }
    ```
    
    **Response:**
    ```json
    {
        "results": {
            "1": {
                "is_violation": false,
                "violation_probability": 0.15,
                "comment": "Sản phẩm rất tốt!"
            },
            "2": {
                "is_violation": true,
                "violation_probability": 0.92,
                "comment": "Đồ rác, lừa đảo!"
            }
        },
        "total_comments": 2,
        "violation_count": 1
    }
    ```
    """
    try:
        # Validate input
        if not request.comments:
            raise HTTPException(
                status_code=400,
                detail="Comment list cannot be empty"
            )
        
        # Lấy danh sách comments và IDs
        comment_ids = list(request.comments.keys())
        comment_texts = list(request.comments.values())
        
        # Log request
        logger_service.info(f"Received classification request for {len(comment_texts)} comments")
        
        # Dự đoán
        labels, violation_probs = predict_batch(comment_texts)
        
        # Xây dựng response
        results = {}
        violation_count = 0
        
        for comment_id, comment_text, label, prob in zip(
            comment_ids, comment_texts, labels, violation_probs
        ):
            is_violation = bool(label == 1)
            
            results[comment_id] = PredictionResult(
                is_violation=is_violation,
                violation_probability=float(prob),
                comment=comment_text
            )
            
            if is_violation:
                violation_count += 1
        
        # Log kết quả
        logger_service.info(
            f"Classification completed: {violation_count}/{len(comment_texts)} violating comments"
        )
        
        return CommentResponse(
            results=results,
            total_comments=len(comment_texts),
            violation_count=violation_count
        )
        
    except HTTPException:
        # Re-raise HTTPException (validation errors)
        raise
    except Exception as e:
        logger_service.error(f"Lỗi khi xử lý request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi server: {str(e)}"
        )


@app.post("/predict/simple")
async def predict_simple(comments: List[str]):
    """
    Endpoint đơn giản hơn - chỉ nhận list string
    
    **Request Body:**
    ```json
    ["Sản phẩm tốt", "Đồ rác"]
    ```
    
    **Response:**
    ```json
    {
        "predictions": [
            {"comment": "Sản phẩm tốt", "is_violation": false, "probability": 0.15},
            {"comment": "Đồ rác", "is_violation": true, "probability": 0.89}
        ]
    }
    ```
    """
    try:
        if not comments:
            raise HTTPException(
                status_code=400,
                detail="Comment list cannot be empty"
            )
        
        # Dự đoán
        labels, violation_probs = predict_batch(comments)
        
        # Xây dựng response
        predictions = []
        for comment, label, prob in zip(comments, labels, violation_probs):
            predictions.append({
                "comment": comment,
                "is_violation": bool(label == 1),
                "violation_probability": float(prob)
            })
        
        return {"predictions": predictions}
        
    except HTTPException:
        # Re-raise HTTPException (validation errors)
        raise
    except Exception as e:
        logger_service.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Comment Classification API Server...")
    print("Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

"""FastAPI main application for market segmentation service."""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from typing import Dict, List, Optional

from ..config.settings import settings
from ..services.prediction import PredictionService
from ..models.data_models import (
    ContactInput, 
    SegmentOutput, 
    BatchSegmentRequest, 
    BatchSegmentResponse
)
from ..utils.logger import segmentation_logger, log_model_operation


# Global prediction service instance
prediction_service: Optional[PredictionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    global prediction_service
    try:
        prediction_service = PredictionService()
        
        # Try to load existing model
        model_path = os.path.join(settings.model_path, "segmentation_model.joblib")
        if os.path.exists(model_path):
            prediction_service.load_model(model_path)
            log_model_operation("load", model_path, success=True)
            segmentation_logger.info("Model loaded successfully on startup")
        else:
            segmentation_logger.warning("No pre-trained model found. Train a model before making predictions.")
    
    except Exception as e:
        segmentation_logger.error(f"Failed to initialize prediction service: {e}")
        log_model_operation("load", success=False)
    
    yield
    
    # Shutdown
    segmentation_logger.info("Application shutting down")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Market Segmentation API using K-Means clustering",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_prediction_service() -> PredictionService:
    """Dependency to get prediction service."""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not available"
        )
    if not prediction_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train a model first."
        )
    return prediction_service


@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "running",
        "environment": settings.environment
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global prediction_service
    
    health_status = {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "model_loaded": False,
        "model_info": {}
    }
    
    if prediction_service and prediction_service.is_loaded:
        health_status["model_loaded"] = True
        health_status["model_info"] = prediction_service.get_model_info()
    
    return health_status


@app.post("/predict", response_model=SegmentOutput)
async def predict_single_contact(
    contact: ContactInput,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict market segment for a single contact.
    
    Args:
        contact: Contact information for segmentation
        
    Returns:
        SegmentOutput with prediction results
    """
    try:
        # Validate input
        is_valid, issues = service.validate_contact_input(contact)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid contact data: {'; '.join(issues)}"
            )
        
        # Make prediction
        segmentation_logger.log_prediction_request(contact.contact_id, "single")
        result = service.predict_single_contact(contact)
        segmentation_logger.log_prediction_result(
            contact.contact_id, 
            result.segment_id, 
            result.confidence_score
        )
        
        return result
        
    except ValueError as e:
        segmentation_logger.log_error(e, f"Single prediction for contact {contact.contact_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        segmentation_logger.log_error(e, f"Single prediction for contact {contact.contact_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@app.post("/predict-batch", response_model=BatchSegmentResponse)
async def predict_batch_contacts(
    batch_request: BatchSegmentRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict market segments for a batch of contacts.
    
    Args:
        batch_request: Batch of contacts for segmentation
        
    Returns:
        BatchSegmentResponse with all prediction results
    """
    try:
        # Validate batch size
        if len(batch_request.contacts) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum batch size is 1000 contacts"
            )
        
        # Validate each contact
        for i, contact in enumerate(batch_request.contacts):
            is_valid, issues = service.validate_contact_input(contact)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid contact data at index {i}: {'; '.join(issues)}"
                )
        
        # Make batch prediction
        segmentation_logger.log_prediction_request(
            f"batch_{len(batch_request.contacts)}", 
            "batch"
        )
        result = service.predict_batch_contacts(batch_request)
        segmentation_logger.log_batch_prediction(
            len(batch_request.contacts),
            result.processing_time_seconds
        )
        
        return result
        
    except ValueError as e:
        segmentation_logger.log_error(e, "Batch prediction")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        segmentation_logger.log_error(e, "Batch prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch prediction"
        )


@app.get("/segments")
async def get_segment_profiles(
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get all segment profiles and characteristics.
    
    Returns:
        Dictionary of segment profiles
    """
    try:
        profiles = service.get_segment_profiles()
        return {
            "segments": profiles,
            "total_segments": len(profiles)
        }
    except Exception as e:
        segmentation_logger.log_error(e, "Get segment profiles")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving segment profiles"
        )


@app.get("/segments/{segment_id}")
async def get_segment_profile(
    segment_id: int,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get profile and recommendations for a specific segment.
    
    Args:
        segment_id: The segment ID (0-14)
        
    Returns:
        Segment profile and marketing recommendations
    """
    try:
        if not (0 <= segment_id <= 14):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Segment ID must be between 0 and 14"
            )
        
        recommendations = service.get_segment_recommendations(segment_id)
        if 'error' in recommendations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=recommendations['error']
            )
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        segmentation_logger.log_error(e, f"Get segment profile for segment {segment_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving segment profile"
        )


@app.get("/model/info")
async def get_model_info(
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get information about the loaded model.
    
    Returns:
        Model information and statistics
    """
    try:
        model_info = service.get_model_info()
        return model_info
    except Exception as e:
        segmentation_logger.log_error(e, "Get model info")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving model information"
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    segmentation_logger.log_api_request(
        str(request.url), 
        request.method, 
        exc.status_code
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    segmentation_logger.log_error(exc, f"Unhandled exception in {request.method} {request.url}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )

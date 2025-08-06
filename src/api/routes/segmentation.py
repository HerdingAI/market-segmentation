"""API routes for market segmentation endpoints."""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import List, Optional, Dict
from datetime import datetime
import json

from ...services.prediction import PredictionService
from ...models.data_models import (
    ContactInput, 
    SegmentOutput, 
    BatchSegmentRequest, 
    BatchSegmentResponse,
    CompanySize,
    IndustryVertical,
    JobFunction
)
from ...utils.logger import segmentation_logger


router = APIRouter(prefix="/api/v1", tags=["segmentation"])


def get_prediction_service() -> PredictionService:
    """Dependency to get prediction service."""
    # This would typically be injected from the main app
    # For now, create a new instance (in production, use dependency injection)
    service = PredictionService()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train a model first."
        )
    return service


@router.post("/segment", response_model=SegmentOutput)
async def segment_contact(
    contact: ContactInput,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Segment a single contact.
    
    This endpoint takes contact information and returns the predicted market segment
    along with confidence score and recommendations.
    """
    try:
        # Validate input
        is_valid, issues = service.validate_contact_input(contact)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "Invalid contact data", "errors": issues}
            )
        
        # Make prediction
        result = service.predict_single_contact(contact)
        
        # Add confidence interpretation
        confidence_interpretation = service.get_prediction_confidence_interpretation(
            result.confidence_score
        )
        
        # Enhance response with additional information
        response_data = result.dict()
        response_data["confidence_interpretation"] = confidence_interpretation
        
        segmentation_logger.log_prediction_result(
            contact.contact_id,
            result.segment_id,
            result.confidence_score
        )
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        segmentation_logger.log_error(e, f"Segmentation error for contact {contact.contact_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during segmentation"
        )


@router.post("/segment-batch", response_model=BatchSegmentResponse)
async def segment_batch(
    batch_request: BatchSegmentRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Segment multiple contacts in a batch.
    
    Process up to 1000 contacts at once for efficient bulk segmentation.
    Returns detailed results for each contact including segment assignments
    and confidence scores.
    """
    try:
        # Process batch
        result = service.predict_batch_contacts(batch_request)
        
        segmentation_logger.log_batch_prediction(
            len(batch_request.contacts),
            result.processing_time_seconds
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        segmentation_logger.log_error(e, "Batch segmentation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during batch segmentation"
        )


@router.get("/segments/profiles")
async def get_all_segment_profiles(
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get profiles and characteristics for all market segments.
    
    Returns comprehensive information about each segment including:
    - Segment characteristics
    - Size distribution
    - Dominant attributes
    - Messaging strategies
    """
    try:
        profiles = service.get_segment_profiles()
        return {
            "segments": profiles,
            "total_segments": len(profiles),
            "retrieved_at": datetime.now().isoformat()
        }
    except Exception as e:
        segmentation_logger.log_error(e, "Error retrieving segment profiles")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving segment profiles"
        )


@router.get("/segments/{segment_id}/profile")
async def get_segment_profile(
    segment_id: int,
    include_recommendations: bool = Query(True, description="Include marketing recommendations"),
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get detailed profile for a specific segment.
    
    Args:
        segment_id: The segment ID (0-14)
        include_recommendations: Whether to include marketing recommendations
    
    Returns:
        Detailed segment profile with optional recommendations
    """
    try:
        if not (0 <= segment_id <= 14):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Segment ID must be between 0 and 14"
            )
        
        if include_recommendations:
            result = service.get_segment_recommendations(segment_id)
            if 'error' in result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result['error']
                )
        else:
            profiles = service.get_segment_profiles()
            if segment_id not in profiles:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Segment {segment_id} not found"
                )
            result = profiles[segment_id]
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        segmentation_logger.log_error(e, f"Error retrieving segment {segment_id} profile")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving segment profile"
        )


@router.get("/segments/{segment_id}/recommendations")
async def get_segment_recommendations(
    segment_id: int,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get marketing recommendations for a specific segment.
    
    Args:
        segment_id: The segment ID (0-14)
    
    Returns:
        Marketing recommendations including content, channels, and timing
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
        
        return {
            "segment_id": segment_id,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        segmentation_logger.log_error(e, f"Error retrieving recommendations for segment {segment_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving segment recommendations"
        )


@router.get("/model/status")
async def get_model_status(
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get current model status and information.
    
    Returns:
        Model status, metrics, and configuration
    """
    try:
        model_info = service.get_model_info()
        return {
            "model_status": model_info,
            "checked_at": datetime.now().isoformat()
        }
    except Exception as e:
        segmentation_logger.log_error(e, "Error retrieving model status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving model status"
        )


@router.get("/enums")
async def get_enum_values():
    """
    Get all valid enum values for contact inputs.
    
    Returns:
        Available values for company_size, industry_vertical, and job_function
    """
    return {
        "company_size": [size.value for size in CompanySize],
        "industry_vertical": [industry.value for industry in IndustryVertical],
        "job_function": [function.value for function in JobFunction],
        "seniority_level_range": {"min": 1, "max": 5},
        "engagement_score_range": {"min": 0.0, "max": 1.0}
    }


@router.post("/validate")
async def validate_contact(contact: ContactInput):
    """
    Validate contact input without making a prediction.
    
    Useful for frontend validation before submitting prediction requests.
    """
    try:
        # Create a temporary service instance for validation
        temp_service = PredictionService()
        is_valid, issues = temp_service.validate_contact_input(contact)
        
        if is_valid:
            return {
                "valid": True,
                "message": "Contact data is valid",
                "contact_id": contact.contact_id
            }
        else:
            return {
                "valid": False,
                "message": "Contact data has validation errors",
                "errors": issues,
                "contact_id": contact.contact_id
            }
            
    except Exception as e:
        segmentation_logger.log_error(e, f"Validation error for contact {contact.contact_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during validation"
        )

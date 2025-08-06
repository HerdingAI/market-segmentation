"""Pydantic models for data validation and serialization."""

from datetime import datetime
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class CompanySize(str, Enum):
    SMB = "smb"
    MID_MARKET = "mid_market"
    ENTERPRISE = "enterprise"


class IndustryVertical(str, Enum):
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    TECHNOLOGY = "technology"
    FINANCIAL_SERVICES = "financial_services"
    RETAIL = "retail"
    OTHER = "other"


class JobFunction(str, Enum):
    C_SUITE = "c_suite"
    OPERATIONS = "operations"
    HR = "hr"
    FINANCE = "finance"
    IT = "it"
    SALES = "sales"
    OTHER = "other"


class ContactInput(BaseModel):
    """Input schema for contact data."""

    contact_id: str = Field(..., description="Unique contact identifier")
    company_size: CompanySize
    industry_vertical: IndustryVertical
    job_function: JobFunction
    seniority_level: int = Field(..., ge=1, le=5, description="Seniority level 1-5")
    geographic_region: str
    engagement_score: float = Field(..., ge=0.0, le=1.0, description="Engagement score 0-1")
    account_revenue: Optional[float] = Field(None, ge=0, description="Estimated annual revenue")

    @validator('engagement_score')
    def validate_engagement_score(cls, v):
        """Ensure engagement score is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Engagement score must be between 0 and 1')
        return v


class SegmentOutput(BaseModel):
    """Output schema for segmentation results."""

    contact_id: str
    segment_id: int = Field(..., ge=0, description="Segment ID")
    segment_name: str = Field(..., description="Human-readable segment name")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    predicted_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SegmentProfile(BaseModel):
    """Segment characteristics and metadata."""

    segment_id: int = Field(..., ge=0)
    segment_name: str
    description: str
    size_distribution: int = Field(..., ge=0, description="Number of contacts in segment")
    avg_engagement_score: float = Field(..., ge=0.0, le=1.0)
    dominant_company_size: CompanySize
    dominant_industry: IndustryVertical
    dominant_job_function: JobFunction
    messaging_strategy: str = Field(..., description="Recommended messaging approach")


class BatchSegmentRequest(BaseModel):
    """Batch segmentation request."""

    contacts: List[ContactInput] = Field(..., description="List of contacts to segment (max 1000)")

    @validator('contacts')
    def validate_contacts_length(cls, v):
        if len(v) < 1:
            raise ValueError('At least one contact is required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 contacts allowed per batch')
        return v


class BatchSegmentResponse(BaseModel):
    """Batch segmentation response."""

    results: List[SegmentOutput]
    total_processed: int
    processing_time_seconds: float
    processed_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

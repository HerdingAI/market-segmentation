"""Prediction service for market segmentation."""

import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Handle imports for both package and direct execution
try:
    from ..models.segmentation import MarketSegmentationModel
    from ..models.data_models import ContactInput, SegmentOutput, BatchSegmentRequest, BatchSegmentResponse
    from ..services.feature_engineering import FeatureEngineeringService
except ImportError:
    # Fallback for direct execution or notebook usage
    from models.segmentation import MarketSegmentationModel
    from models.data_models import ContactInput, SegmentOutput, BatchSegmentRequest, BatchSegmentResponse
    from services.feature_engineering import FeatureEngineeringService


class PredictionService:
    """Service for making segmentation predictions."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = MarketSegmentationModel()
        self.feature_service = FeatureEngineeringService()
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Optional[str] = None):
        """Load a trained model."""
        try:
            if model_path:
                self.model.load_model(model_path)
            else:
                self.model.load_model()  # Use default path
            self.is_loaded = True
        except FileNotFoundError:
            raise ValueError(f"Model file not found. Please train a model first.")
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

    def predict_single_contact(self, contact: ContactInput) -> SegmentOutput:
        """
        Predict segment for a single contact.
        
        Args:
            contact: Contact input data
            
        Returns:
            SegmentOutput with prediction results
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        # Make prediction
        segment_id = self.model.predict_single(contact)
        
        # Calculate confidence score
        confidence_score = self.model.calculate_confidence_score(contact)
        
        # Get segment profile
        segment_profile = self.model.get_segment_profile(segment_id)
        segment_name = segment_profile.segment_name if segment_profile else f"Segment {segment_id}"
        
        return SegmentOutput(
            contact_id=contact.contact_id,
            segment_id=segment_id,
            segment_name=segment_name,
            confidence_score=confidence_score,
            predicted_at=datetime.now()
        )

    def predict_batch_contacts(self, batch_request: BatchSegmentRequest) -> BatchSegmentResponse:
        """
        Predict segments for a batch of contacts.
        
        Args:
            batch_request: Batch request with list of contacts
            
        Returns:
            BatchSegmentResponse with all predictions
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        start_time = time.time()
        results = []
        
        # Make predictions for all contacts
        segment_ids = self.model.predict(batch_request.contacts)
        
        # Generate detailed results for each contact
        for i, contact in enumerate(batch_request.contacts):
            segment_id = segment_ids[i]
            confidence_score = self.model.calculate_confidence_score(contact)
            
            # Get segment profile
            segment_profile = self.model.get_segment_profile(segment_id)
            segment_name = segment_profile.segment_name if segment_profile else f"Segment {segment_id}"
            
            result = SegmentOutput(
                contact_id=contact.contact_id,
                segment_id=segment_id,
                segment_name=segment_name,
                confidence_score=confidence_score,
                predicted_at=datetime.now()
            )
            results.append(result)
        
        processing_time = time.time() - start_time
        
        return BatchSegmentResponse(
            results=results,
            total_processed=len(results),
            processing_time_seconds=processing_time,
            processed_at=datetime.now()
        )

    def get_segment_profiles(self) -> Dict[int, Dict]:
        """Get all segment profiles."""
        if not self.is_loaded:
            raise ValueError("Model must be loaded to get segment profiles")
        
        profiles = {}
        for segment_id, profile in self.model.segment_profiles.items():
            profiles[segment_id] = {
                'segment_id': profile.segment_id,
                'segment_name': profile.segment_name,
                'description': profile.description,
                'size_distribution': profile.size_distribution,
                'avg_engagement_score': profile.avg_engagement_score,
                'dominant_company_size': profile.dominant_company_size.value,
                'dominant_industry': profile.dominant_industry.value,
                'dominant_job_function': profile.dominant_job_function.value,
                'messaging_strategy': profile.messaging_strategy
            }
        
        return profiles

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {'status': 'not_loaded', 'message': 'No model loaded'}
        
        cluster_centers = self.model.get_cluster_centers()
        
        return {
            'status': 'loaded',
            'n_clusters': self.model.n_clusters,
            'n_features': len(self.model.feature_names) if self.model.feature_names else 0,
            'feature_names': self.model.feature_names,
            'n_segments_with_profiles': len(self.model.segment_profiles),
            'has_cluster_centers': cluster_centers is not None,
            'cluster_centers_shape': cluster_centers.shape if cluster_centers is not None else None
        }

    def validate_contact_input(self, contact: ContactInput) -> Tuple[bool, List[str]]:
        """
        Validate contact input data.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if not contact.contact_id:
            issues.append("Contact ID is required")
        
        # Check engagement score range
        if not (0.0 <= contact.engagement_score <= 1.0):
            issues.append("Engagement score must be between 0.0 and 1.0")
        
        # Check seniority level range
        if not (1 <= contact.seniority_level <= 5):
            issues.append("Seniority level must be between 1 and 5")
        
        # Check account revenue if provided
        if contact.account_revenue is not None and contact.account_revenue < 0:
            issues.append("Account revenue cannot be negative")
        
        # Check geographic region
        if not contact.geographic_region or len(contact.geographic_region.strip()) == 0:
            issues.append("Geographic region is required")
        
        return len(issues) == 0, issues

    def get_prediction_confidence_interpretation(self, confidence_score: float) -> str:
        """
        Interpret confidence score into human-readable format.
        
        Args:
            confidence_score: Confidence score between 0 and 1
            
        Returns:
            Human-readable confidence interpretation
        """
        if confidence_score >= 0.8:
            return "High confidence - Contact clearly fits this segment"
        elif confidence_score >= 0.6:
            return "Medium-high confidence - Good segment fit"
        elif confidence_score >= 0.4:
            return "Medium confidence - Reasonable segment fit"
        elif confidence_score >= 0.2:
            return "Low-medium confidence - Uncertain segment fit"
        else:
            return "Low confidence - Contact may not fit well in any segment"

    def get_segment_recommendations(self, segment_id: int) -> Dict:
        """
        Get marketing recommendations for a specific segment.
        
        Args:
            segment_id: The segment ID
            
        Returns:
            Dictionary with marketing recommendations
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded to get recommendations")
        
        segment_profile = self.model.get_segment_profile(segment_id)
        if not segment_profile:
            return {'error': f'Segment {segment_id} not found'}
        
        # Generate recommendations based on segment characteristics
        recommendations = {
            'segment_id': segment_id,
            'segment_name': segment_profile.segment_name,
            'messaging_strategy': segment_profile.messaging_strategy,
            'content_recommendations': self._generate_content_recommendations(segment_profile),
            'channel_recommendations': self._generate_channel_recommendations(segment_profile),
            'timing_recommendations': self._generate_timing_recommendations(segment_profile)
        }
        
        return recommendations

    def _generate_content_recommendations(self, profile) -> List[str]:
        """Generate content recommendations based on segment profile."""
        recommendations = []
        
        # Based on company size
        if profile.dominant_company_size.value == 'enterprise':
            recommendations.append("Enterprise-focused case studies and ROI calculators")
            recommendations.append("White papers on scalability and security")
        elif profile.dominant_company_size.value == 'smb':
            recommendations.append("Small business success stories and quick wins")
            recommendations.append("Cost-effective solution guides")
        else:
            recommendations.append("Mid-market growth stories and expansion strategies")
        
        # Based on job function
        if profile.dominant_job_function.value == 'c_suite':
            recommendations.append("Strategic vision and competitive advantage content")
            recommendations.append("Executive summaries and board-ready materials")
        elif profile.dominant_job_function.value == 'it':
            recommendations.append("Technical documentation and integration guides")
            recommendations.append("Security and compliance information")
        elif profile.dominant_job_function.value == 'sales':
            recommendations.append("Revenue impact and sales enablement tools")
            recommendations.append("Performance metrics and KPI dashboards")
        
        # Based on engagement level
        if profile.avg_engagement_score > 0.7:
            recommendations.append("Advanced feature demonstrations and roadmap previews")
        elif profile.avg_engagement_score < 0.4:
            recommendations.append("Basic value proposition and problem-solution fit content")
        
        return recommendations

    def _generate_channel_recommendations(self, profile) -> List[str]:
        """Generate channel recommendations based on segment profile."""
        channels = []
        
        # Based on job function
        if profile.dominant_job_function.value == 'c_suite':
            channels.extend(["LinkedIn executive outreach", "Industry conference speaking"])
        elif profile.dominant_job_function.value == 'it':
            channels.extend(["Technical webinars", "Developer communities"])
        elif profile.dominant_job_function.value == 'sales':
            channels.extend(["Sales-focused events", "Revenue operations communities"])
        
        # Based on engagement level
        if profile.avg_engagement_score > 0.6:
            channels.append("Direct sales outreach")
        else:
            channels.extend(["Email nurture campaigns", "Retargeting ads"])
        
        # Based on company size
        if profile.dominant_company_size.value == 'enterprise':
            channels.append("Account-based marketing")
        else:
            channels.append("Self-service digital channels")
        
        return channels

    def _generate_timing_recommendations(self, profile) -> List[str]:
        """Generate timing recommendations based on segment profile."""
        timing = []
        
        # Based on engagement level
        if profile.avg_engagement_score > 0.7:
            timing.append("Immediate follow-up within 24 hours")
        elif profile.avg_engagement_score > 0.4:
            timing.append("Follow-up within 3-5 business days")
        else:
            timing.append("Longer nurture cycle with weekly touchpoints")
        
        # Based on job function
        if profile.dominant_job_function.value == 'c_suite':
            timing.append("Schedule for Tuesday-Thursday, 10 AM - 2 PM")
        else:
            timing.append("Flexible scheduling, avoid Monday mornings and Friday afternoons")
        
        return timing

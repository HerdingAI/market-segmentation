"""Tests for the market segmentation system."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import os
import tempfile

# Import the modules to test
from src.models.data_models import (
    ContactInput, 
    SegmentOutput, 
    CompanySize, 
    IndustryVertical, 
    JobFunction
)
from src.models.segmentation import MarketSegmentationModel
from src.services.feature_engineering import FeatureEngineeringService
from src.services.prediction import PredictionService


class TestDataModels:
    """Test data models and validation."""
    
    def test_contact_input_valid(self):
        """Test valid contact input creation."""
        contact = ContactInput(
            contact_id="test_001",
            company_size=CompanySize.SMB,
            industry_vertical=IndustryVertical.TECHNOLOGY,
            job_function=JobFunction.IT,
            seniority_level=3,
            geographic_region="North America",
            engagement_score=0.75,
            account_revenue=100000.0
        )
        
        assert contact.contact_id == "test_001"
        assert contact.company_size == CompanySize.SMB
        assert contact.engagement_score == 0.75
    
    def test_contact_input_validation_errors(self):
        """Test contact input validation errors."""
        # Test invalid engagement score
        with pytest.raises(ValueError):
            ContactInput(
                contact_id="test_001",
                company_size=CompanySize.SMB,
                industry_vertical=IndustryVertical.TECHNOLOGY,
                job_function=JobFunction.IT,
                seniority_level=3,
                geographic_region="North America",
                engagement_score=1.5,  # Invalid: > 1.0
                account_revenue=100000.0
            )
        
        # Test invalid seniority level
        with pytest.raises(ValueError):
            ContactInput(
                contact_id="test_001",
                company_size=CompanySize.SMB,
                industry_vertical=IndustryVertical.TECHNOLOGY,
                job_function=JobFunction.IT,
                seniority_level=6,  # Invalid: > 5
                geographic_region="North America",
                engagement_score=0.75,
                account_revenue=100000.0
            )
    
    def test_segment_output_creation(self):
        """Test segment output creation."""
        output = SegmentOutput(
            contact_id="test_001",
            segment_id=5,
            segment_name="Tech SMB",
            confidence_score=0.85
        )
        
        assert output.contact_id == "test_001"
        assert output.segment_id == 5
        assert output.confidence_score == 0.85


class TestFeatureEngineering:
    """Test feature engineering service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = FeatureEngineeringService()
        self.sample_contacts = [
            ContactInput(
                contact_id="test_001",
                company_size=CompanySize.SMB,
                industry_vertical=IndustryVertical.TECHNOLOGY,
                job_function=JobFunction.IT,
                seniority_level=3,
                geographic_region="North America",
                engagement_score=0.75,
                account_revenue=50000.0
            ),
            ContactInput(
                contact_id="test_002",
                company_size=CompanySize.ENTERPRISE,
                industry_vertical=IndustryVertical.HEALTHCARE,
                job_function=JobFunction.C_SUITE,
                seniority_level=5,
                geographic_region="Europe",
                engagement_score=0.9,
                account_revenue=1000000.0
            )
        ]
    
    def test_contacts_to_dataframe(self):
        """Test conversion of contacts to DataFrame."""
        df = self.service._contacts_to_dataframe(self.sample_contacts)
        
        assert len(df) == 2
        assert 'contact_id' in df.columns
        assert 'company_size' in df.columns
        assert df.loc[0, 'contact_id'] == "test_001"
        assert df.loc[1, 'engagement_score'] == 0.9
    
    def test_feature_engineering(self):
        """Test feature engineering pipeline."""
        df = self.service._contacts_to_dataframe(self.sample_contacts)
        df_engineered = self.service._engineer_features(df)
        
        # Check new features are created
        assert 'has_revenue_data' in df_engineered.columns
        assert 'high_engagement' in df_engineered.columns
        assert 'senior_role' in df_engineered.columns
        assert 'tech_industry' in df_engineered.columns
        
        # Check feature values
        assert df_engineered.loc[0, 'has_revenue_data'] == 1  # Has revenue
        assert df_engineered.loc[1, 'high_engagement'] == 1   # High engagement (>0.7)
        assert df_engineered.loc[1, 'senior_role'] == 1       # Senior role (level 5)
    
    def test_prepare_features_fit_transform(self):
        """Test feature preparation with fitting."""
        df_transformed = self.service.prepare_features(self.sample_contacts, fit=True)
        
        assert self.service.is_fitted
        assert len(self.service.feature_columns) > 0
        assert df_transformed.shape[0] == 2
        assert df_transformed.shape[1] == len(self.service.feature_columns)
    
    def test_feature_validation(self):
        """Test feature validation."""
        df_transformed = self.service.prepare_features(self.sample_contacts, fit=True)
        is_valid, issues = self.service.validate_features(df_transformed)
        
        assert is_valid
        assert len(issues) == 0


class TestSegmentationModel:
    """Test market segmentation model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = MarketSegmentationModel(n_clusters=3)  # Smaller for testing
        self.sample_contacts = [
            ContactInput(
                contact_id=f"test_{i:03d}",
                company_size=CompanySize.SMB if i % 2 == 0 else CompanySize.ENTERPRISE,
                industry_vertical=IndustryVertical.TECHNOLOGY if i % 3 == 0 else IndustryVertical.HEALTHCARE,
                job_function=JobFunction.IT if i % 2 == 0 else JobFunction.C_SUITE,
                seniority_level=(i % 5) + 1,
                geographic_region="North America" if i % 2 == 0 else "Europe",
                engagement_score=min(0.1 + (i * 0.1), 1.0),
                account_revenue=50000.0 + (i * 10000)
            )
            for i in range(20)  # Generate 20 test contacts
        ]
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.n_clusters == 3
        assert self.model.random_state == 42
        assert self.model.model is None
        assert len(self.model.encoders) == 4  # 4 categorical features
    
    def test_prepare_features(self):
        """Test feature preparation."""
        X = self.model._prepare_features(self.sample_contacts)
        
        assert X.shape[0] == 20  # 20 contacts
        assert X.shape[1] == 7   # 7 features
        assert isinstance(X, np.ndarray)
    
    def test_model_training(self):
        """Test model training."""
        metrics = self.model.train(self.sample_contacts)
        
        assert self.model.model is not None
        assert 'n_clusters' in metrics
        assert 'silhouette_score' in metrics
        assert 'inertia' in metrics
        assert metrics['n_clusters'] == 3
        assert len(self.model.segment_profiles) <= 3
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Train model first
        self.model.train(self.sample_contacts)
        
        # Test single prediction
        test_contact = self.sample_contacts[0]
        segment_id = self.model.predict_single(test_contact)
        
        assert isinstance(segment_id, int)
        assert 0 <= segment_id < 3
        
        # Test batch prediction
        predictions = self.model.predict(self.sample_contacts[:5])
        assert len(predictions) == 5
        assert all(0 <= pred < 3 for pred in predictions)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Train model first
        self.model.train(self.sample_contacts)
        
        test_contact = self.sample_contacts[0]
        confidence = self.model.calculate_confidence_score(test_contact)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Train model
        self.model.train(self.sample_contacts)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            saved_path = self.model.save_model(tmp_file.name)
            assert os.path.exists(saved_path)
            
            # Create new model and load
            new_model = MarketSegmentationModel()
            new_model.load_model(saved_path)
            
            # Test loaded model works
            test_contact = self.sample_contacts[0]
            prediction1 = self.model.predict_single(test_contact)
            prediction2 = new_model.predict_single(test_contact)
            
            assert prediction1 == prediction2
            
            # Clean up - handle Windows file permission issues
            try:
                os.unlink(saved_path)
            except PermissionError:
                # Windows sometimes keeps file handles, ignore cleanup error
                pass


class TestPredictionService:
    """Test prediction service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = PredictionService()
        self.sample_contact = ContactInput(
            contact_id="test_001",
            company_size=CompanySize.SMB,
            industry_vertical=IndustryVertical.TECHNOLOGY,
            job_function=JobFunction.IT,
            seniority_level=3,
            geographic_region="North America",
            engagement_score=0.75,
            account_revenue=50000.0
        )
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service.model is not None
        assert not self.service.is_loaded
    
    def test_contact_validation(self):
        """Test contact input validation."""
        # Valid contact
        is_valid, issues = self.service.validate_contact_input(self.sample_contact)
        assert is_valid
        assert len(issues) == 0

        # Invalid contact - bad engagement score should raise validation error
        with pytest.raises(ValueError):
            invalid_contact = ContactInput(
                contact_id="test_002",
                company_size=CompanySize.SMB,
                industry_vertical=IndustryVertical.TECHNOLOGY,
                job_function=JobFunction.IT,
                seniority_level=3,
                geographic_region="North America",
                engagement_score=1.5,  # Invalid
                account_revenue=50000.0
            )
    
    def test_confidence_interpretation(self):
        """Test confidence score interpretation."""
        interpretations = [
            (0.9, "High confidence"),
            (0.7, "Medium-high confidence"),
            (0.5, "Medium confidence"),
            (0.3, "Low-medium confidence"),
            (0.1, "Low confidence")
        ]
        
        for score, expected_start in interpretations:
            interpretation = self.service.get_prediction_confidence_interpretation(score)
            assert interpretation.startswith(expected_start)
    
    @patch('src.services.prediction.PredictionService.load_model')
    def test_model_info_not_loaded(self, mock_load):
        """Test model info when model is not loaded."""
        service = PredictionService()
        info = service.get_model_info()
        
        assert info['status'] == 'not_loaded'
        assert 'message' in info


def test_integration_workflow():
    """Test the complete workflow integration."""
    # Create sample data
    contacts = [
        ContactInput(
            contact_id=f"integration_test_{i:03d}",
            company_size=CompanySize.SMB if i % 2 == 0 else CompanySize.ENTERPRISE,
            industry_vertical=IndustryVertical.TECHNOLOGY if i % 3 == 0 else IndustryVertical.HEALTHCARE,
            job_function=JobFunction.IT if i % 2 == 0 else JobFunction.C_SUITE,
            seniority_level=(i % 5) + 1,
            geographic_region="North America" if i % 2 == 0 else "Europe",
            engagement_score=min(0.1 + (i * 0.05), 1.0),
            account_revenue=50000.0 + (i * 10000)
        )
        for i in range(30)
    ]
    
    # Train model
    model = MarketSegmentationModel(n_clusters=5)
    training_metrics = model.train(contacts)
    
    assert 'silhouette_score' in training_metrics
    assert model.model is not None
    
    # Test prediction service
    service = PredictionService()
    service.model = model  # Use trained model
    service.is_loaded = True
    
    # Test single prediction
    result = service.predict_single_contact(contacts[0])
    assert isinstance(result, SegmentOutput)
    assert 0 <= result.segment_id < 5
    assert 0 <= result.confidence_score <= 1
    
    # Test batch prediction
    from src.models.data_models import BatchSegmentRequest
    batch_request = BatchSegmentRequest(contacts=contacts[:10])
    batch_result = service.predict_batch_contacts(batch_request)
    
    assert len(batch_result.results) == 10
    assert batch_result.total_processed == 10
    assert batch_result.processing_time_seconds > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

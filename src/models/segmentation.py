"""K-Means clustering model for market segmentation."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import joblib
import os
from typing import Dict, List, Tuple, Optional

# Handle imports for both package and direct execution
try:
    from ..config.settings import settings
    from ..models.data_models import ContactInput, SegmentProfile, CompanySize, IndustryVertical, JobFunction
except ImportError:
    # Fallback for direct execution or notebook usage
    from config.settings import settings
    from models.data_models import ContactInput, SegmentProfile, CompanySize, IndustryVertical, JobFunction


class MarketSegmentationModel:
    """K-Means clustering model for market segmentation."""

    def __init__(self, n_clusters: Optional[int] = None):
        self.n_clusters = n_clusters or settings.n_clusters
        self.random_state = settings.random_state
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        self.segment_profiles = {}
        self._setup_encoders()

    def _setup_encoders(self):
        """Initialize label encoders for categorical variables."""
        self.encoders = {
            'company_size': LabelEncoder(),
            'industry_vertical': LabelEncoder(),
            'job_function': LabelEncoder(),
            'geographic_region': LabelEncoder()
        }

    def _prepare_features(self, contacts: List[ContactInput]) -> np.ndarray:
        """Convert contact data to feature matrix."""
        # Convert to DataFrame for easier manipulation
        data = []
        for contact in contacts:
            row = {
                'company_size': contact.company_size.value,
                'industry_vertical': contact.industry_vertical.value,
                'job_function': contact.job_function.value,
                'seniority_level': contact.seniority_level,
                'geographic_region': contact.geographic_region,
                'engagement_score': contact.engagement_score,
                'account_revenue': contact.account_revenue or 0.0
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Encode categorical variables
        categorical_cols = ['company_size', 'industry_vertical', 'job_function', 'geographic_region']
        for col in categorical_cols:
            if hasattr(self.encoders[col], 'classes_'):
                # Transform using fitted encoder, handle unknown values
                try:
                    df[col] = self.encoders[col].transform(df[col])
                except ValueError as e:
                    # Handle unknown categories by assigning them to the first class
                    unknown_values = set(df[col]) - set(self.encoders[col].classes_)
                    if unknown_values:
                        print(f"Warning: Unknown categories in {col}: {unknown_values}. Mapping to first class.")
                        df[col] = df[col].apply(lambda x: x if x in self.encoders[col].classes_ else self.encoders[col].classes_[0])
                        df[col] = self.encoders[col].transform(df[col])
            else:
                # Fit and transform for training
                df[col] = self.encoders[col].fit_transform(df[col])
        
        return df.values

    def train(self, contacts: List[ContactInput]) -> Dict:
        """Train the segmentation model."""
        # Prepare features
        X = self._prepare_features(contacts)
        self.feature_names = ['company_size', 'industry_vertical', 'job_function', 
                             'seniority_level', 'geographic_region', 'engagement_score', 
                             'account_revenue']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-Means model
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=self.random_state
        )
        
        cluster_labels = self.model.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        inertia = self.model.inertia_
        
        # Generate segment profiles
        self._generate_segment_profiles(contacts, cluster_labels)
        
        return {
            'n_clusters': self.n_clusters,
            'silhouette_score': silhouette_avg,
            'inertia': inertia,
            'n_samples': len(contacts)
        }

    def predict(self, contacts: List[ContactInput]) -> List[int]:
        """Predict segments for new contacts."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X = self._prepare_features(contacts)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).tolist()

    def predict_single(self, contact: ContactInput) -> int:
        """Predict segment for a single contact."""
        return self.predict([contact])[0]

    def get_segment_profile(self, segment_id: int) -> Optional[SegmentProfile]:
        """Get profile information for a specific segment."""
        return self.segment_profiles.get(segment_id)

    def _generate_segment_profiles(self, contacts: List[ContactInput], cluster_labels: np.ndarray):
        """Generate profiles for each segment."""
        df_contacts = pd.DataFrame([
            {
                'contact_id': c.contact_id,
                'company_size': c.company_size.value,
                'industry_vertical': c.industry_vertical.value,
                'job_function': c.job_function.value,
                'seniority_level': c.seniority_level,
                'geographic_region': c.geographic_region,
                'engagement_score': c.engagement_score,
                'account_revenue': c.account_revenue or 0.0,
                'segment': cluster_labels[i]
            }
            for i, c in enumerate(contacts)
        ])
        
        for segment_id in range(self.n_clusters):
            segment_data = df_contacts[df_contacts['segment'] == segment_id]
            
            if len(segment_data) == 0:
                continue
            
            # Calculate segment characteristics
            profile = SegmentProfile(
                segment_id=segment_id,
                segment_name=f"Segment {segment_id}",
                description=self._generate_segment_description(segment_data),
                size_distribution=len(segment_data),
                avg_engagement_score=segment_data['engagement_score'].mean(),
                dominant_company_size=CompanySize(segment_data['company_size'].mode().iloc[0]),
                dominant_industry=IndustryVertical(segment_data['industry_vertical'].mode().iloc[0]),
                dominant_job_function=JobFunction(segment_data['job_function'].mode().iloc[0]),
                messaging_strategy=self._generate_messaging_strategy(segment_data)
            )
            
            self.segment_profiles[segment_id] = profile

    def _generate_segment_description(self, segment_data: pd.DataFrame) -> str:
        """Generate a human-readable description for a segment."""
        dominant_size = segment_data['company_size'].mode().iloc[0]
        dominant_industry = segment_data['industry_vertical'].mode().iloc[0]
        dominant_function = segment_data['job_function'].mode().iloc[0]
        avg_engagement = segment_data['engagement_score'].mean()
        
        return (f"{dominant_size.replace('_', ' ').title()} companies in "
                f"{dominant_industry.replace('_', ' ').title()} industry, "
                f"primarily {dominant_function.replace('_', ' ').title()} roles "
                f"with {avg_engagement:.1%} average engagement")

    def _generate_messaging_strategy(self, segment_data: pd.DataFrame) -> str:
        """Generate messaging strategy recommendations."""
        avg_engagement = segment_data['engagement_score'].mean()
        avg_seniority = segment_data['seniority_level'].mean()
        
        if avg_engagement > 0.7:
            engagement_strategy = "Highly engaged - focus on advanced features and ROI"
        elif avg_engagement > 0.4:
            engagement_strategy = "Moderately engaged - educational content and demos"
        else:
            engagement_strategy = "Low engagement - awareness building and value proposition"
        
        if avg_seniority > 3.5:
            seniority_strategy = "Senior audience - strategic and executive messaging"
        else:
            seniority_strategy = "Operational audience - practical and tactical messaging"
        
        return f"{engagement_strategy}. {seniority_strategy}."

    def save_model(self, filepath: Optional[str] = None):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            os.makedirs(settings.model_path, exist_ok=True)
            filepath = os.path.join(settings.model_path, "segmentation_model.joblib")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'segment_profiles': self.segment_profiles,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        return filepath

    def load_model(self, filepath: Optional[str] = None):
        """Load a trained model from disk."""
        if filepath is None:
            filepath = os.path.join(settings.model_path, "segmentation_model.joblib")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.encoders = model_data['encoders']
        self.feature_names = model_data['feature_names']
        self.segment_profiles = model_data['segment_profiles']
        self.n_clusters = model_data['n_clusters']
        self.random_state = model_data['random_state']

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers from trained model."""
        if self.model is None:
            return None
        return self.model.cluster_centers_

    def calculate_confidence_score(self, contact: ContactInput) -> float:
        """Calculate confidence score for a prediction."""
        if self.model is None:
            raise ValueError("Model must be trained before calculating confidence")
        
        X = self._prepare_features([contact])
        X_scaled = self.scaler.transform(X)
        
        # Calculate distances to all cluster centers
        distances = self.model.transform(X_scaled)[0]
        
        # Confidence is inverse of distance to closest cluster
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Normalize to 0-1 range (closer = higher confidence)
        if max_distance > min_distance:
            confidence = 1 - (min_distance / max_distance)
        else:
            confidence = 1.0
        
        return confidence

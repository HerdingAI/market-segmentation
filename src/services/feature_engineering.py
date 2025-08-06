"""Feature engineering service for market segmentation."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Handle imports for both package and direct execution
try:
    from ..models.data_models import ContactInput
except ImportError:
    # Fallback for direct execution or notebook usage
    from models.data_models import ContactInput


class FeatureEngineeringService:
    """Service for transforming raw contact data into ML-ready features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_columns = []
        self.is_fitted = False

    def prepare_features(self, contacts: List[ContactInput], fit: bool = False) -> pd.DataFrame:
        """
        Transform contact data into feature matrix.
        
        Args:
            contacts: List of contact inputs
            fit: Whether to fit encoders (True for training, False for prediction)
        
        Returns:
            DataFrame with engineered features
        """
        # Convert contacts to DataFrame
        df = self._contacts_to_dataframe(contacts)
        
        # Engineer features
        df_features = self._engineer_features(df)
        
        if fit:
            # Fit transformers during training
            self._fit_transformers(df_features)
        
        # Transform features
        df_transformed = self._transform_features(df_features)
        
        return df_transformed

    def _contacts_to_dataframe(self, contacts: List[ContactInput]) -> pd.DataFrame:
        """Convert list of contacts to DataFrame."""
        data = []
        for contact in contacts:
            row = {
                'contact_id': contact.contact_id,
                'company_size': contact.company_size.value,
                'industry_vertical': contact.industry_vertical.value,
                'job_function': contact.job_function.value,
                'seniority_level': contact.seniority_level,
                'geographic_region': contact.geographic_region,
                'engagement_score': contact.engagement_score,
                'account_revenue': contact.account_revenue or 0.0
            }
            data.append(row)
        
        return pd.DataFrame(data)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from raw data."""
        df_engineered = df.copy()
        
        # Revenue-based features
        df_engineered['has_revenue_data'] = (df['account_revenue'] > 0).astype(int)
        df_engineered['log_revenue'] = np.log1p(df['account_revenue'])  # log(1 + revenue)
        
        # Engagement-based features
        df_engineered['high_engagement'] = (df['engagement_score'] > 0.7).astype(int)
        df_engineered['low_engagement'] = (df['engagement_score'] < 0.3).astype(int)
        df_engineered['engagement_squared'] = df['engagement_score'] ** 2
        
        # Seniority-based features
        df_engineered['senior_role'] = (df['seniority_level'] >= 4).astype(int)
        df_engineered['junior_role'] = (df['seniority_level'] <= 2).astype(int)
        
        # Company size mapping to numeric
        size_mapping = {'smb': 1, 'mid_market': 2, 'enterprise': 3}
        df_engineered['company_size_numeric'] = df['company_size'].map(size_mapping)
        
        # Industry groupings
        tech_industries = ['technology', 'financial_services']
        traditional_industries = ['manufacturing', 'healthcare', 'retail']
        
        df_engineered['tech_industry'] = df['industry_vertical'].isin(tech_industries).astype(int)
        df_engineered['traditional_industry'] = df['industry_vertical'].isin(traditional_industries).astype(int)
        
        # Job function groupings
        leadership_roles = ['c_suite']
        operational_roles = ['operations', 'hr', 'finance', 'it']
        revenue_roles = ['sales']
        
        df_engineered['leadership_role'] = df['job_function'].isin(leadership_roles).astype(int)
        df_engineered['operational_role'] = df['job_function'].isin(operational_roles).astype(int)
        df_engineered['revenue_role'] = df['job_function'].isin(revenue_roles).astype(int)
        
        # Interaction features
        df_engineered['size_seniority_interaction'] = (
            df_engineered['company_size_numeric'] * df['seniority_level']
        )
        df_engineered['engagement_seniority_interaction'] = (
            df['engagement_score'] * df['seniority_level']
        )
        df_engineered['revenue_engagement_interaction'] = (
            df_engineered['log_revenue'] * df['engagement_score']
        )
        
        return df_engineered

    def _fit_transformers(self, df: pd.DataFrame):
        """Fit all transformers on the training data."""
        # Define categorical and numerical columns
        categorical_cols = ['company_size', 'industry_vertical', 'job_function', 'geographic_region']
        numerical_cols = [
            'seniority_level', 'engagement_score', 'account_revenue', 'log_revenue',
            'company_size_numeric', 'engagement_squared', 'size_seniority_interaction',
            'engagement_seniority_interaction', 'revenue_engagement_interaction'
        ]
        
        # Fit label encoders for categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(df[col])
        
        # Fit scaler for numerical variables
        self.scaler.fit(df[numerical_cols])
        
        # Store feature columns for consistent ordering
        binary_cols = [
            'has_revenue_data', 'high_engagement', 'low_engagement', 'senior_role',
            'junior_role', 'tech_industry', 'traditional_industry', 'leadership_role',
            'operational_role', 'revenue_role'
        ]
        
        self.feature_columns = categorical_cols + numerical_cols + binary_cols
        self.is_fitted = True

    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted transformers."""
        if not self.is_fitted:
            raise ValueError("Transformers must be fitted before transformation")
        
        df_transformed = df.copy()
        
        # Transform categorical variables
        categorical_cols = ['company_size', 'industry_vertical', 'job_function', 'geographic_region']
        for col in categorical_cols:
            df_transformed[col] = self.label_encoders[col].transform(df[col])
        
        # Transform numerical variables
        numerical_cols = [
            'seniority_level', 'engagement_score', 'account_revenue', 'log_revenue',
            'company_size_numeric', 'engagement_squared', 'size_seniority_interaction',
            'engagement_seniority_interaction', 'revenue_engagement_interaction'
        ]
        df_transformed[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Select only the feature columns in consistent order
        df_final = df_transformed[self.feature_columns]
        
        return df_final

    def get_feature_names(self) -> List[str]:
        """Get the names of all engineered features."""
        return self.feature_columns.copy()

    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """Get mapping of feature names to their descriptions."""
        return {
            'company_size': 'Company size category (SMB, Mid-market, Enterprise)',
            'industry_vertical': 'Industry vertical',
            'job_function': 'Job function/role',
            'geographic_region': 'Geographic region',
            'seniority_level': 'Seniority level (1-5)',
            'engagement_score': 'Engagement score (0-1)',
            'account_revenue': 'Account revenue',
            'log_revenue': 'Log-transformed revenue',
            'company_size_numeric': 'Numeric company size',
            'engagement_squared': 'Squared engagement score',
            'size_seniority_interaction': 'Company size × Seniority interaction',
            'engagement_seniority_interaction': 'Engagement × Seniority interaction',
            'revenue_engagement_interaction': 'Revenue × Engagement interaction',
            'has_revenue_data': 'Binary: Has revenue data',
            'high_engagement': 'Binary: High engagement (>0.7)',
            'low_engagement': 'Binary: Low engagement (<0.3)',
            'senior_role': 'Binary: Senior role (level ≥4)',
            'junior_role': 'Binary: Junior role (level ≤2)',
            'tech_industry': 'Binary: Technology industry',
            'traditional_industry': 'Binary: Traditional industry',
            'leadership_role': 'Binary: Leadership role',
            'operational_role': 'Binary: Operational role',
            'revenue_role': 'Binary: Revenue-generating role'
        }

    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that the feature DataFrame has expected structure.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required columns
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for NaN values
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"NaN values found in columns: {nan_cols}")
        
        # Check data types
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Non-numeric data type in column: {col}")
        
        # Check value ranges for specific columns
        if 'engagement_score' in df.columns:
            invalid_engagement = ((df['engagement_score'] < -3) | (df['engagement_score'] > 3)).any()
            if invalid_engagement:
                issues.append("Engagement scores outside expected range after scaling")
        
        return len(issues) == 0, issues

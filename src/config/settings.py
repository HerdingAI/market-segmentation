"""Application configuration management with environment-based overrides."""

import os
from typing import Optional


class Settings:
    """Application settings with validation and type safety."""

    def __init__(self):
        # Application
        self.app_name = "Market Segmentation"
        self.version = "1.0.0"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "False").lower() == "true"

        # Database
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///market_segmentation.db")

        # Model Configuration
        self.n_clusters = int(os.getenv("N_CLUSTERS", "15"))
        self.random_state = int(os.getenv("RANDOM_STATE", "42"))
        self.model_path = os.getenv("MODEL_PATH", "models/")

        # API Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "4"))

        # Monitoring
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.monitoring_enabled = os.getenv("MONITORING_ENABLED", "True").lower() == "true"

        # External Services
        self.crm_api_url = os.getenv("CRM_API_URL")
        self.crm_api_key = os.getenv("CRM_API_KEY")


# Global settings instance
settings = Settings()

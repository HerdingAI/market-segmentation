"""Logging utility for the market segmentation application."""

import logging
import sys
from datetime import datetime
from typing import Optional
from ..config.settings import settings


def setup_logger(
    name: str = "market_segmentation",
    level: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set logging level
    log_level = level or settings.log_level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create formatter
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "market_segmentation") -> logging.Logger:
    """Get or create a logger with the specified name."""
    return logging.getLogger(name)


class SegmentationLogger:
    """Enhanced logger for market segmentation operations."""
    
    def __init__(self, name: str = "market_segmentation"):
        self.logger = setup_logger(name)
        self.name = name
    
    def log_prediction_request(self, contact_id: str, prediction_type: str = "single"):
        """Log a prediction request."""
        self.logger.info(
            f"Prediction request - Type: {prediction_type}, Contact ID: {contact_id}"
        )
    
    def log_prediction_result(self, contact_id: str, segment_id: int, confidence: float):
        """Log a prediction result."""
        self.logger.info(
            f"Prediction result - Contact: {contact_id}, "
            f"Segment: {segment_id}, Confidence: {confidence:.3f}"
        )
    
    def log_batch_prediction(self, batch_size: int, processing_time: float):
        """Log batch prediction results."""
        self.logger.info(
            f"Batch prediction completed - Size: {batch_size}, "
            f"Processing time: {processing_time:.3f}s"
        )
    
    def log_model_operation(self, operation: str, model_path: Optional[str] = None, success: bool = True):
        """Log model operations like loading, saving, training."""
        status = "SUCCESS" if success else "FAILED"
        path_info = f", Path: {model_path}" if model_path else ""
        self.logger.info(f"Model {operation} - Status: {status}{path_info}")
    
    def log_training_metrics(self, metrics: dict):
        """Log training metrics."""
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"Training metrics - {metrics_str}")
    
    def log_feature_engineering(self, n_contacts: int, n_features: int):
        """Log feature engineering operation."""
        self.logger.info(
            f"Feature engineering completed - "
            f"Contacts: {n_contacts}, Features: {n_features}"
        )
    
    def log_validation_error(self, contact_id: str, errors: list):
        """Log validation errors."""
        error_str = "; ".join(errors)
        self.logger.warning(
            f"Validation error - Contact: {contact_id}, Errors: {error_str}"
        )
    
    def log_api_request(self, endpoint: str, method: str, status_code: Optional[int] = None):
        """Log API requests."""
        status_info = f", Status: {status_code}" if status_code else ""
        self.logger.info(f"API request - {method} {endpoint}{status_info}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context."""
        context_info = f" - Context: {context}" if context else ""
        self.logger.error(f"Error occurred{context_info} - {str(error)}", exc_info=True)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics."""
        unit_info = f" {unit}" if unit else ""
        self.logger.info(f"Performance metric - {metric_name}: {value}{unit_info}")
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


# Global logger instance
segmentation_logger = SegmentationLogger()


# Convenience functions
def log_prediction_request(contact_id: str, prediction_type: str = "single"):
    """Log a prediction request."""
    segmentation_logger.log_prediction_request(contact_id, prediction_type)


def log_prediction_result(contact_id: str, segment_id: int, confidence: float):
    """Log a prediction result."""
    segmentation_logger.log_prediction_result(contact_id, segment_id, confidence)


def log_batch_prediction(batch_size: int, processing_time: float):
    """Log batch prediction results."""
    segmentation_logger.log_batch_prediction(batch_size, processing_time)


def log_model_operation(operation: str, model_path: Optional[str] = None, success: bool = True):
    """Log model operations."""
    segmentation_logger.log_model_operation(operation, model_path, success)


def log_training_metrics(metrics: dict):
    """Log training metrics."""
    segmentation_logger.log_training_metrics(metrics)


def log_error(error: Exception, context: str = ""):
    """Log errors with context."""
    segmentation_logger.log_error(error, context)

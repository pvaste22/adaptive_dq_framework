"""
Module: Common
Phase: All
Author: Pranjal V
Created: 06/09/2025
Purpose: Custom exception classes for the framework
"""

class DataQualityFrameworkError(Exception):
    """Base exception for all framework errors."""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code

class DataLoadingError(DataQualityFrameworkError):
    """Raised when data loading fails."""
    pass

class DataValidationError(DataQualityFrameworkError):
    """Raised when data validation fails."""
    pass

class UnitConversionError(DataQualityFrameworkError):
    """Raised when unit conversion fails."""
    pass

class WindowGenerationError(DataQualityFrameworkError):
    """Raised when window generation fails."""
    pass

class QualityDimensionError(DataQualityFrameworkError):
    """Raised when quality dimension calculation fails."""
    pass

class DriftDetectionError(DataQualityFrameworkError):
    """Raised when drift detection fails."""
    pass

class ModelTrainingError(DataQualityFrameworkError):
    """Raised when model training fails."""
    pass

class ArtifactError(DataQualityFrameworkError):
    """Raised when artifact operations fail."""
    pass

class ConfigurationError(DataQualityFrameworkError):
    """Raised when configuration is invalid."""
    pass
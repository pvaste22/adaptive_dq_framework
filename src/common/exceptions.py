
class DataQualityFrameworkError(Exception):
    """Base exception for all framework errors."""
    pass

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
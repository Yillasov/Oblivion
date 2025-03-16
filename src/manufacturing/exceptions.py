"""
Custom exceptions for the manufacturing module.
"""

from src.core.utils.exceptions import NeuromorphicError

class ManufacturingError(NeuromorphicError):
    """
    Base class for all manufacturing-related exceptions.
    
    This exception serves as the parent class for all exceptions
    specific to the manufacturing module.
    """
    pass

class ManufacturingValidationError(ManufacturingError):
    """
    Exception raised when a manufacturing validation fails.
    
    This exception is raised when an airframe design fails validation
    against manufacturing constraints or when manufacturing results
    are incomplete or invalid.
    """
    pass

class PipelineCreationError(ManufacturingError):
    """
    Exception raised when a manufacturing pipeline cannot be created.
    
    This exception is raised when the system fails to create a
    manufacturing pipeline for a specific airframe type.
    """
    pass

class MaterialUnavailableError(ManufacturingError):
    """
    Exception raised when a required material is unavailable.
    
    This exception is raised when a design requires materials
    that are not available in the manufacturing system.
    """
    pass

class ManufacturingCapabilityError(ManufacturingError):
    """
    Exception raised when a required manufacturing capability is unavailable.
    
    This exception is raised when a design requires manufacturing
    capabilities that are not available in the current system.
    """
    pass
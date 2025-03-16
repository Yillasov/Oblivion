"""
Hardware-related exceptions for the neuromorphic SDK.
"""

class NeuromorphicHardwareError(Exception):
    """Base exception for all neuromorphic hardware errors."""
    pass


class HardwareInitializationError(NeuromorphicHardwareError):
    """Exception raised when hardware initialization fails."""
    pass


class HardwareAllocationError(NeuromorphicHardwareError):
    """Exception raised when resource allocation on hardware fails."""
    pass


class HardwareSimulationError(NeuromorphicHardwareError):
    """Exception raised when simulation execution fails."""
    pass


class HardwareCommunicationError(NeuromorphicHardwareError):
    """Exception raised when communication with hardware fails."""
    pass


class UnsupportedFeatureError(NeuromorphicHardwareError):
    """Exception raised when attempting to use an unsupported hardware feature."""
    pass
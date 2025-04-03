#!/usr/bin/env python3
"""
Custom exceptions for the neuromorphic SDK.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class NeuromorphicError(Exception):
    """Base class for all neuromorphic SDK exceptions."""
    pass


class HardwareError(NeuromorphicError):
    """Base class for hardware-related errors."""
    pass


class ConfigurationError(NeuromorphicError):
    """Error raised when there's an issue with configuration."""
    pass


class SimulationError(NeuromorphicError):
    """Error raised when there's an issue with simulation."""
    pass


class ResourceError(NeuromorphicError):
    """Error raised when there's an issue with resources."""
    pass


class NetworkError(NeuromorphicError):
    """Error raised when there's an issue with neural networks."""
    pass


class ValidationError(NeuromorphicError):
    """Error raised when validation fails."""
    pass


class TimeoutError(NeuromorphicError):
    """Error raised when an operation times out."""
    pass


class NotImplementedError(NeuromorphicError):
    """Error raised when a feature is not implemented."""
    pass


class MemoryError(ResourceError):
    """Error raised when there's an issue with memory."""
    pass


class ProcessingError(ResourceError):
    """Error raised when there's an issue with processing."""
    pass


class HardwareConnectionError(HardwareError):
    """Error raised when connection to hardware fails."""
    pass


class HardwareConfigurationError(HardwareError):
    """Error raised when hardware configuration fails."""
    pass


class HardwareOperationError(HardwareError):
    """Error raised when a hardware operation fails."""
    pass
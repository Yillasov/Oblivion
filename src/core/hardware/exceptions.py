#!/usr/bin/env python3
"""
Hardware-related exceptions for the neuromorphic SDK.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class NeuromorphicHardwareError(Exception):
    """Base exception for all neuromorphic hardware errors."""
    pass


class HardwareInitializationError(NeuromorphicHardwareError):
    """Exception raised when hardware initialization fails."""
    pass


# Add the new exception class
class HardwareSwitchingError(NeuromorphicHardwareError):
    """Exception raised when hardware switching fails."""
    pass


class HardwareCommunicationError(NeuromorphicHardwareError):
    """Exception raised when communication with hardware fails."""
    pass


class HardwareAllocationError(NeuromorphicHardwareError):
    """Exception raised when resource allocation on hardware fails."""
    pass


class HardwareSimulationError(NeuromorphicHardwareError):
    """Exception raised when simulation execution fails."""
    pass


class UnsupportedFeatureError(NeuromorphicHardwareError):
    """Exception raised when attempting to use an unsupported hardware feature."""
    pass


class ResourceSharingError(Exception):
    """Exception raised for errors in hardware resource sharing."""
    pass

#!/usr/bin/env python3
"""
Hardware-specific error codes for the neuromorphic SDK.

Provides standardized error codes for hardware-related issues to improve
error handling, debugging, and reporting.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple

from src.core.utils.logging_framework import get_logger

logger = get_logger("hardware_errors")


class HardwareErrorCode(Enum):
    """Hardware-specific error codes."""
    
    # General hardware errors (1000-1099)
    UNKNOWN_ERROR = 1000
    INITIALIZATION_FAILED = 1001
    COMMUNICATION_FAILED = 1002
    RESOURCE_ALLOCATION_FAILED = 1003
    SIMULATION_FAILED = 1004
    UNSUPPORTED_FEATURE = 1005
    HARDWARE_SWITCHING_FAILED = 1006
    
    # Loihi-specific errors (1100-1199)
    LOIHI_CHIP_ERROR = 1100
    LOIHI_MEMORY_OVERFLOW = 1101
    LOIHI_SYNAPSE_LIMIT_EXCEEDED = 1102
    
    # SpiNNaker-specific errors (1200-1299)
    SPINNAKER_CORE_FAILURE = 1200
    SPINNAKER_ROUTING_ERROR = 1201
    SPINNAKER_PACKET_LOSS = 1202
    
    # TrueNorth-specific errors (1300-1399)
    TRUENORTH_CONFIGURATION_ERROR = 1300
    TRUENORTH_TIMING_ERROR = 1301
    
    # Simulation-specific errors (1400-1499)
    SIMULATION_TIMEOUT = 1400
    SIMULATION_NUMERICAL_ERROR = 1401


class HardwareErrorInfo:
    """Container for hardware error information."""
    
    def __init__(self, 
                 code: HardwareErrorCode,
                 message: str,
                 hardware_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize hardware error information.
        
        Args:
            code: Error code
            message: Error message
            hardware_type: Type of hardware that generated the error
            details: Additional error details
        """
        self.code = code
        self.message = message
        self.hardware_type = hardware_type
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation of error."""
        return f"[{self.code.name}:{self.code.value}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code.value,
            "code_name": self.code.name,
            "message": self.message,
            "hardware_type": self.hardware_type,
            "details": self.details
        }


def get_error_info(exception: Exception) -> Tuple[HardwareErrorCode, Dict[str, Any]]:
    """
    Extract error code and details from exception.
    
    Args:
        exception: Exception to analyze
        
    Returns:
        Tuple[HardwareErrorCode, Dict[str, Any]]: Error code and details
    """
    error_type = exception.__class__.__name__
    error_msg = str(exception)
    
    # Map exception types to error codes
    if error_type == "HardwareInitializationError":
        return HardwareErrorCode.INITIALIZATION_FAILED, {}
    elif error_type == "HardwareCommunicationError":
        return HardwareErrorCode.COMMUNICATION_FAILED, {}
    elif error_type == "HardwareAllocationError":
        return HardwareErrorCode.RESOURCE_ALLOCATION_FAILED, {}
    elif error_type == "HardwareSimulationError":
        return HardwareErrorCode.SIMULATION_FAILED, {}
    elif error_type == "UnsupportedFeatureError":
        return HardwareErrorCode.UNSUPPORTED_FEATURE, {}
    elif error_type == "HardwareSwitchingError":
        return HardwareErrorCode.HARDWARE_SWITCHING_FAILED, {}
    
    # Extract hardware-specific information from error message
    details = {}
    if "loihi" in error_msg.lower():
        if "memory" in error_msg.lower():
            return HardwareErrorCode.LOIHI_MEMORY_OVERFLOW, details
        elif "synapse" in error_msg.lower():
            return HardwareErrorCode.LOIHI_SYNAPSE_LIMIT_EXCEEDED, details
        return HardwareErrorCode.LOIHI_CHIP_ERROR, details
    elif "spinnaker" in error_msg.lower():
        if "routing" in error_msg.lower():
            return HardwareErrorCode.SPINNAKER_ROUTING_ERROR, details
        elif "packet" in error_msg.lower():
            return HardwareErrorCode.SPINNAKER_PACKET_LOSS, details
        return HardwareErrorCode.SPINNAKER_CORE_FAILURE, details
    elif "truenorth" in error_msg.lower():
        if "timing" in error_msg.lower():
            return HardwareErrorCode.TRUENORTH_TIMING_ERROR, details
        return HardwareErrorCode.TRUENORTH_CONFIGURATION_ERROR, details
    elif "simulation" in error_msg.lower():
        if "timeout" in error_msg.lower():
            return HardwareErrorCode.SIMULATION_TIMEOUT, details
        elif "numerical" in error_msg.lower():
            return HardwareErrorCode.SIMULATION_NUMERICAL_ERROR, details
    
    # Default error code
    return HardwareErrorCode.UNKNOWN_ERROR, details


def create_error_info(exception: Exception, hardware_type: Optional[str] = None) -> HardwareErrorInfo:
    """
    Create hardware error information from exception.
    
    Args:
        exception: Exception to analyze
        hardware_type: Type of hardware that generated the error
        
    Returns:
        HardwareErrorInfo: Hardware error information
    """
    code, details = get_error_info(exception)
    return HardwareErrorInfo(
        code=code,
        message=str(exception),
        hardware_type=hardware_type,
        details=details
    )


def log_hardware_error(error_info: HardwareErrorInfo) -> None:
    """
    Log hardware error with standardized format.
    
    Args:
        error_info: Hardware error information
    """
    logger.error(f"Hardware Error {error_info.code.value}: {error_info.message}")
    if error_info.hardware_type:
        logger.error(f"Hardware Type: {error_info.hardware_type}")
    if error_info.details:
        logger.error(f"Error Details: {error_info.details}")
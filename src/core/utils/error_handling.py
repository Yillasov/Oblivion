#!/usr/bin/env python3
"""
Centralized error handling for the Oblivion project.

This module provides a standardized approach to error handling across the codebase,
including custom exceptions, error logging, and recovery mechanisms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import traceback
from enum import Enum, auto
from typing import Dict, Any, Optional, Callable, List, Type, Union

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class OblivionError(Exception):
    """Base exception class for all Oblivion-specific errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            severity: Error severity level
            details: Additional error details
        """
        self.message = message
        self.severity = severity
        self.details = details or {}
        super().__init__(message)
    
    def log(self) -> None:
        """Log the error with appropriate severity."""
        log_message = f"{self.__class__.__name__}: {self.message}"
        
        if self.details:
            log_message += f" - Details: {self.details}"
            
        if self.severity == ErrorSeverity.DEBUG:
            logger.debug(log_message)
        elif self.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)


# Hardware-related exceptions
class HardwareError(OblivionError):
    """Base class for hardware-related errors."""
    pass


class HardwareInitializationError(HardwareError):
    """Error during hardware initialization."""
    pass


class HardwareCommunicationError(HardwareError):
    """Error during hardware communication."""
    pass


class HardwareAllocationError(HardwareError):
    """Error during hardware resource allocation."""
    pass


class HardwareSimulationError(HardwareError):
    """Error during hardware simulation."""
    pass


class UnsupportedFeatureError(HardwareError):
    """Error when attempting to use an unsupported hardware feature."""
    pass


# Manufacturing-related exceptions
class ManufacturingError(OblivionError):
    """Base class for manufacturing-related errors."""
    pass


class MaterialError(ManufacturingError):
    """Error related to manufacturing materials."""
    pass


class EquipmentError(ManufacturingError):
    """Error related to manufacturing equipment."""
    pass


class QualityControlError(ManufacturingError):
    """Error during quality control checks."""
    pass


# Testing-related exceptions
class TestingError(OblivionError):
    """Base class for testing-related errors."""
    pass


class TestSetupError(TestingError):
    """Error during test setup."""
    pass


class TestExecutionError(TestingError):
    """Error during test execution."""
    pass


class TestValidationError(TestingError):
    """Error during test validation."""
    pass


# Error handler class
class ErrorHandler:
    """
    Centralized error handler for consistent error management.
    
    This class provides methods for handling errors, logging them,
    and executing recovery strategies when applicable.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_log: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[Type[OblivionError], Callable] = {}
        
    def register_recovery_strategy(self, error_type: Type[OblivionError], 
                                  strategy: Callable[[OblivionError], bool]) -> None:
        """
        Register a recovery strategy for a specific error type.
        
        Args:
            error_type: The type of error to handle
            strategy: Function that attempts to recover from the error
        """
        self.recovery_strategies[error_type] = strategy
        
    def handle_error(self, error: Union[OblivionError, Exception], 
                    context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: The error to handle
            context: Additional context information
            
        Returns:
            bool: True if error was recovered, False otherwise
        """
        context = context or {}
        
        # Convert standard exceptions to OblivionError
        if not isinstance(error, OblivionError):
            error = OblivionError(
                message=str(error),
                severity=ErrorSeverity.ERROR,
                details={"exception_type": error.__class__.__name__}
            )
        
        # Log the error
        error.log()
        
        # Record error in log
        error_entry = {
            "type": error.__class__.__name__,
            "message": error.message,
            "severity": error.severity.name,
            "details": error.details,
            "context": context,
            "traceback": traceback.format_exc()
        }
        self.error_log.append(error_entry)
        
        # Attempt recovery if strategy exists
        for error_type, strategy in self.recovery_strategies.items():
            if isinstance(error, error_type):
                try:
                    recovered = strategy(error)
                    error_entry["recovered"] = recovered
                    return recovered
                except Exception as recovery_error:
                    logger.error(f"Error during recovery: {recovery_error}")
                    error_entry["recovery_error"] = str(recovery_error)
        
        error_entry["recovered"] = False
        return False
    
    def clear_error_log(self) -> None:
        """Clear the error log."""
        self.error_log = []
    
    def get_error_log(self) -> List[Dict[str, Any]]:
        """Get the error log."""
        return self.error_log


# Create a global error handler instance
global_error_handler = ErrorHandler()


# Context manager for error handling
class ErrorContext:
    """
    Context manager for handling errors.
    """
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None, 
                context: Optional[Dict[str, Any]] = None):
        """
        Initialize the error context.
        
        Args:
            error_handler: Error handler to use (defaults to global handler)
            context: Optional context information
        """
        self.error_handler = error_handler or global_error_handler
        self.context = context or {}
    
    def __enter__(self):
        """Enter the context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context, handling any errors.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
            
        Returns:
            bool: True to suppress the exception, False to propagate it
        """
        if exc_val:
            return self.error_handler.handle_error(exc_val, self.context)
        
        return False


# Decorator for error handling
def handle_errors(error_handler: Optional[ErrorHandler] = None, 
                 context: Optional[Dict[str, Any]] = None,
                 suppress_exceptions: bool = False):
    """
    Decorator for handling errors in functions.
    
    Args:
        error_handler: Error handler to use (defaults to global handler)
        context: Optional context information
        suppress_exceptions: Whether to suppress exceptions
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = error_handler or global_error_handler
            ctx = context or {}
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                recovered = handler.handle_error(e, ctx)
                
                if not recovered and not suppress_exceptions:
                    raise
                
                # Return a default value if suppressing exceptions
                return None
                
        return wrapper
    return decorator
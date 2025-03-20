"""
Context manager for hardware operations with enhanced error handling.
"""

from typing import Dict, Any, Optional, Callable, Type, List, Union
import contextlib
import time
import functools
import traceback

from src.core.utils.logging_framework import get_logger
from src.core.hardware.exceptions import NeuromorphicHardwareError
from src.core.hardware.error_codes import HardwareErrorCode, HardwareErrorInfo, create_error_info, log_hardware_error
from src.core.hardware.recovery_strategies import attempt_recovery

logger = get_logger("hardware_error_context")


class HardwareErrorContext:
    """
    Context manager for hardware operations with enhanced error handling.
    
    Provides:
    - Automatic error classification
    - Detailed error reporting
    - Automatic recovery attempts
    - Operation retry with backoff
    """
    
    def __init__(self, 
                 hardware_type: str, 
                 operation_name: str,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 error_callback: Optional[Callable] = None):
        """
        Initialize hardware error context.
        
        Args:
            hardware_type: Type of hardware being used
            operation_name: Name of the operation being performed
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            error_callback: Optional callback for errors
        """
        self.hardware_type = hardware_type
        self.operation_name = operation_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_callback = error_callback
        self.attempts = 0
        self.last_error = None
        self.error_info = None
        
    def __enter__(self):
        """Enter the context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager with error handling.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
            
        Returns:
            bool: True if exception was handled, False otherwise
        """
        if exc_type is None:
            return False
            
        # Only handle NeuromorphicHardwareError
        if not issubclass(exc_type, NeuromorphicHardwareError):
            return False
            
        self.attempts += 1
        self.last_error = exc_val
        
        # Create and log error info
        self.error_info = create_error_info(exc_val, self.hardware_type)
        log_hardware_error(self.error_info)
        
        # Call error callback if provided
        if self.error_callback:
            try:
                self.error_callback(self.error_info)
            except Exception as e:
                logger.warning(f"Error in callback: {str(e)}")
        
        # Attempt recovery
        recovery_successful = attempt_recovery(
            self.hardware_type, 
            self.error_info.code, 
            self.error_info.details
        )
        
        # If recovery was successful and we haven't exceeded max retries
        if recovery_successful and self.attempts <= self.max_retries:
            # Calculate backoff delay
            delay = self.retry_delay * (2 ** (self.attempts - 1))
            logger.info(f"Recovery successful, retrying operation '{self.operation_name}' "
                       f"(attempt {self.attempts}/{self.max_retries}) after {delay:.2f}s delay")
            
            # Wait before retry
            time.sleep(delay)
            
            # Indicate that we've handled the exception
            return True
            
        # If we've reached max retries or recovery failed
        if self.attempts > self.max_retries:
            logger.error(f"Maximum retry attempts ({self.max_retries}) reached for operation '{self.operation_name}'")
        elif not recovery_successful:
            logger.error(f"Recovery failed for operation '{self.operation_name}'")
            
        # Add retry information to the exception
        error_message = f"{str(exc_val)} [Error Code: {self.error_info.code.name}:{self.error_info.code.value}] "
        error_message += f"[Retries: {self.attempts}/{self.max_retries}]"
        
        # Create a new exception with enhanced information
        new_exception = exc_type(error_message)
        
        # Replace the original exception
        raise new_exception from exc_val


def with_hardware_error_context(hardware_type: str, 
                               operation_name: Optional[str] = None,
                               max_retries: int = 3,
                               retry_delay: float = 1.0,
                               error_callback: Optional[Callable] = None) -> Callable:
    """
    Decorator for hardware operations with enhanced error handling.
    
    Args:
        hardware_type: Type of hardware being used
        operation_name: Name of the operation (defaults to function name)
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (seconds)
        error_callback: Optional callback for errors
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use function name if operation_name not provided
            op_name = operation_name or func.__name__
            
            # Create context manager
            with HardwareErrorContext(
                hardware_type=hardware_type,
                operation_name=op_name,
                max_retries=max_retries,
                retry_delay=retry_delay,
                error_callback=error_callback
            ):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


@contextlib.contextmanager
def hardware_operation(hardware_type: str, 
                      operation_name: str,
                      max_retries: int = 3,
                      retry_delay: float = 1.0):
    """
    Context manager for hardware operations with enhanced error handling.
    
    Args:
        hardware_type: Type of hardware being used
        operation_name: Name of the operation
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (seconds)
        
    Yields:
        None
    """
    with HardwareErrorContext(
        hardware_type=hardware_type,
        operation_name=operation_name,
        max_retries=max_retries,
        retry_delay=retry_delay
    ):
        yield
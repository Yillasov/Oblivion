"""
Hardware error handler integration with the core error handling system.
"""

from typing import Dict, Any, Optional, Callable
import functools

from src.core.utils.logging_framework import get_logger
from src.core.hardware.exceptions import NeuromorphicHardwareError
from src.core.hardware.error_codes import create_error_info, log_hardware_error, HardwareErrorInfo
from src.core.hardware.recovery_strategies import attempt_recovery

logger = get_logger("hardware_error_handler")


def handle_hardware_error(func: Callable) -> Callable:
    """
    Decorator for handling hardware errors with error codes.
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NeuromorphicHardwareError as e:
            # Get hardware type from instance if available
            hardware_type = None
            if args and hasattr(args[0], 'hardware_type'):
                hardware_type = args[0].hardware_type
            
            # Create and log error info
            error_info = create_error_info(e, hardware_type)
            log_hardware_error(error_info)
            
            # Attempt recovery if hardware type is available
            if hardware_type:
                if attempt_recovery(hardware_type, error_info.code, error_info.details):
                    logger.info(f"Recovery successful, retrying operation")
                    # Retry the operation
                    return func(*args, **kwargs)
            
            # Create a new exception with the error info included in the message
            error_message = f"{str(e)} [Error Code: {error_info.code.name}:{error_info.code.value}]"
            new_exception = e.__class__(error_message)
            
            # Preserve the original traceback
            raise new_exception from e
    
    return wrapper


def with_hardware_error_handling(hardware_type: Optional[str] = None) -> Callable:
    """
    Context manager for hardware error handling with error codes.
    
    Args:
        hardware_type: Type of hardware
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except NeuromorphicHardwareError as e:
                # Create and log error info
                error_info = create_error_info(e, hardware_type)
                log_hardware_error(error_info)
                
                # Attempt recovery if hardware type is available
                if hardware_type:
                    if attempt_recovery(hardware_type, error_info.code, error_info.details):
                        logger.info(f"Recovery successful, retrying operation")
                        # Retry the operation
                        return func(*args, **kwargs)
                
                # Create a new exception with the error info included in the message
                error_message = f"{str(e)} [Error Code: {error_info.code.name}:{error_info.code.value}]"
                new_exception = e.__class__(error_message)
                
                # Preserve the original traceback
                raise new_exception from e
        
        return wrapper
    
    return decorator


def get_error_info_from_exception(exception: NeuromorphicHardwareError) -> Optional[HardwareErrorInfo]:
    """
    Extract error info from an exception message if it contains error code information.
    
    Args:
        exception: The exception to analyze
        
    Returns:
        Optional[HardwareErrorInfo]: Extracted error info or None if not found
    """
    message = str(exception)
    # Check if the message contains error code information
    if "[Error Code:" in message:
        try:
            # We don't need to extract hardware type from the exception
            # since it's not available as an attribute
            hardware_type = None
            
            # Create error info from the exception
            return create_error_info(exception, hardware_type)
        except Exception as e:
            logger.warning(f"Failed to extract error info from exception: {str(e)}")
    
    return None
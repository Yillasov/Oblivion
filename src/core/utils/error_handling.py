#!/usr/bin/env python3
"""
Error handling utilities for the Oblivion project.
Provides decorators and context managers for consistent error handling.
"""

import os
import sys
import traceback
import functools
from typing import Dict, Any, Callable, Optional, Type, Union
from contextlib import contextmanager

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger

logger = get_logger("error_handling")


class ErrorContext:
    """Context manager for error handling with additional context."""
    
    def __init__(self, context: Dict[str, Any] = None, 
                 reraise: bool = True, 
                 error_types: Optional[Union[Type[Exception], tuple]] = None):
        """
        Initialize error context.
        
        Args:
            context: Additional context information for error logs
            reraise: Whether to re-raise the exception
            error_types: Specific error types to catch (defaults to all exceptions)
        """
        self.context = context or {}
        self.reraise = reraise
        self.error_types = error_types or Exception
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
            
        Returns:
            bool: True if exception was handled, False otherwise
        """
        if exc_type is not None and issubclass(exc_type, self.error_types):
            # Log the error with context
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            logger.error(f"Error in context [{context_str}]: {exc_val}")
            logger.debug(f"Traceback: {''.join(traceback.format_tb(exc_tb))}")
            
            # Return True to suppress the exception if reraise is False
            return not self.reraise
        
        return False


def handle_errors(context: Dict[str, Any] = None, 
                 reraise: bool = True, 
                 error_types: Optional[Union[Type[Exception], tuple]] = None):
    """
    Decorator for handling errors with additional context.
    
    Args:
        context: Additional context information for error logs
        reraise: Whether to re-raise the exception
        error_types: Specific error types to catch (defaults to all exceptions)
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Add function name to context
            ctx = context.copy() if context else {}
            ctx["function"] = func.__name__
            
            with ErrorContext(ctx, reraise, error_types):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def log_and_reraise(error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log an error with additional context and re-raise it.
    
    Args:
        error: The exception to log
        context: Additional context information for error logs
        
    Raises:
        Exception: The original exception
    """
    context_str = ", ".join(f"{k}={v}" for k, v in (context or {}).items())
    logger.error(f"Error [{context_str}]: {error}")
    logger.debug(f"Traceback: {''.join(traceback.format_tb(error.__traceback__))}")
    raise error
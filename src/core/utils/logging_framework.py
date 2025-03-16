"""
Error Handling and Logging Framework

This module provides a unified approach to error handling and logging
for the neuromorphic SDK.
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import json


class NeuromorphicLogger:
    """
    Centralized logging system for the neuromorphic SDK.
    """
    
    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    # Default format
    DEFAULT_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    def __init__(self, name: str = "neuromorphic_sdk", log_dir: str = "/Users/yessine/Oblivion/logs"):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory to store log files
        """
        self.name = name
        self.log_dir = log_dir
        self.loggers: Dict[str, logging.Logger] = {}
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up root logger
        self.setup_logger(name)
    
    def setup_logger(self, logger_name: str, level: int = logging.INFO) -> logging.Logger:
        """
        Set up a logger with the specified name and level.
        
        Args:
            logger_name: Name of the logger
            level: Logging level
            
        Returns:
            logging.Logger: Configured logger
        """
        if logger_name in self.loggers:
            return self.loggers[logger_name]
        
        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(self.DEFAULT_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        log_file = os.path.join(self.log_dir, f"{logger_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(self.DEFAULT_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Store logger
        self.loggers[logger_name] = logger
        
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger by name, creating it if it doesn't exist.
        
        Args:
            name: Logger name
            
        Returns:
            logging.Logger: The requested logger
        """
        if name not in self.loggers:
            return self.setup_logger(name)
        
        return self.loggers[name]
    
    def set_level(self, name: str, level: int):
        """
        Set the logging level for a specific logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        logger = self.get_logger(name)
        logger.setLevel(level)
        
        for handler in logger.handlers:
            handler.setLevel(level)
    
    def set_global_level(self, level: int):
        """
        Set the logging level for all loggers.
        
        Args:
            level: Logging level
        """
        for name, logger in self.loggers.items():
            self.set_level(name, level)


class ErrorHandler:
    """
    Error handling system for the neuromorphic SDK.
    """
    
    def __init__(self, logger: NeuromorphicLogger):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.error_callbacks: Dict[str, List[Callable]] = {}
        self.error_log: List[Dict[str, Any]] = []
    
    def register_error_callback(self, error_type: str, callback: Callable):
        """
        Register a callback for a specific error type.
        
        Args:
            error_type: Type of error to register for
            callback: Function to call when error occurs
        """
        if error_type not in self.error_callbacks:
            self.error_callbacks[error_type] = []
        
        self.error_callbacks[error_type].append(callback)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error, logging it and executing any registered callbacks.
        
        Args:
            error: The exception to handle
            context: Optional context information
            
        Returns:
            Dict[str, Any]: Error information
        """
        error_type = error.__class__.__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Create error info
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'stack_trace': stack_trace,
            'context': context or {}
        }
        
        # Log the error
        log_message = f"{error_type}: {error_message}"
        if context:
            log_message += f" | Context: {json.dumps(context)}"
        
        self.logger.get_logger(self.logger.name).error(log_message)
        self.logger.get_logger(self.logger.name).debug(stack_trace)
        
        # Add to error log
        self.error_log.append(error_info)
        
        # Execute callbacks
        if error_type in self.error_callbacks:
            for callback in self.error_callbacks[error_type]:
                try:
                    callback(error, context)
                except Exception as callback_error:
                    self.logger.get_logger(self.logger.name).error(
                        f"Error in callback for {error_type}: {str(callback_error)}"
                    )
        
        return error_info
    
    def get_error_log(self) -> List[Dict[str, Any]]:
        """
        Get the error log.
        
        Returns:
            List[Dict[str, Any]]: List of error information dictionaries
        """
        return self.error_log
    
    def clear_error_log(self):
        """Clear the error log."""
        self.error_log = []


class ErrorContext:
    """
    Context manager for handling errors.
    """
    
    def __init__(self, error_handler: ErrorHandler, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the error context.
        
        Args:
            error_handler: Error handler to use
            context: Optional context information
        """
        self.error_handler = error_handler
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
            self.error_handler.handle_error(exc_val, self.context)
            return True  # Suppress the exception
        
        return False  # No exception to handle


# Create global instances
neuromorphic_logger = NeuromorphicLogger()
error_handler = ErrorHandler(neuromorphic_logger)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: The requested logger
    """
    return neuromorphic_logger.get_logger(name)


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Handle an error.
    
    Args:
        error: The exception to handle
        context: Optional context information
        
    Returns:
        Dict[str, Any]: Error information
    """
    return error_handler.handle_error(error, context)


def error_context(context: Optional[Dict[str, Any]] = None) -> ErrorContext:
    """
    Create an error context.
    
    Args:
        context: Optional context information
        
    Returns:
        ErrorContext: Error context manager
    """
    return ErrorContext(error_handler, context)
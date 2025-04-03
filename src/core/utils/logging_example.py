#!/usr/bin/env python3
"""
Example usage of the error handling and logging framework.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .logging_framework import get_logger, handle_error, error_context
from .exceptions import HardwareError, ConfigurationError, SimulationError

# Get a logger
logger = get_logger("example")

def example_function():
    """Example function demonstrating logging and error handling."""
    # Basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    
    # Error handling with context
    try:
        # Simulate an error
        raise HardwareError("Example hardware error")
    except Exception as e:
        # Handle the error with context
        handle_error(e, {"function": "example_function", "operation": "hardware_test"})
    
    # Using error context manager
    with error_context({"function": "example_function", "operation": "config_test"}):
        # Simulate another error
        raise ConfigurationError("Example configuration error")
    
    # This code will still execute because the error was handled by the context manager
    logger.info("Continuing after handled error")
    
    # Nested error contexts
    with error_context({"level": "outer"}):
        logger.info("In outer context")
        
        try:
            with error_context({"level": "inner"}):
                logger.info("In inner context")
                raise SimulationError("Example simulation error")
        except Exception:
            # This won't execute because the inner context handles the error
            logger.error("This won't be logged")
        
        logger.info("Back to outer context")


if __name__ == "__main__":
    # Set log level
    from .logging_framework import neuromorphic_logger
    neuromorphic_logger.set_global_level(neuromorphic_logger.DEBUG)
    
    # Run example
    try:
        example_function()
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
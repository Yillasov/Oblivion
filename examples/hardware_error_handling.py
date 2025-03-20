"""
Example demonstrating enhanced hardware error handling.
"""

import sys
import time
import logging
from typing import Dict, Any

from src.core.hardware.enhanced_driver import EnhancedHardwareDriver
from src.core.hardware.exceptions import NeuromorphicHardwareError
from src.core.utils.logging_framework import get_logger

# Configure logging using standard library instead
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger("hardware_example")


def error_callback(error_info):
    """Custom error callback."""
    logger.warning(f"Custom error handler called: {error_info}")


def run_hardware_example():
    """Run hardware error handling example."""
    logger.info("Starting hardware error handling example")
    
    # Create hardware driver with normal configuration
    normal_config = {"device_id": "test-device-1"}
    normal_driver = EnhancedHardwareDriver("loihi", normal_config)
    
    try:
        # Initialize hardware
        normal_driver.initialize()
        
        # Execute operations
        result = normal_driver.execute_operation("test_operation", {"param1": "value1"})
        logger.info(f"Operation result: {result}")
        
        # Try operation with simulated error
        try:
            normal_driver.execute_operation("error_operation", {"simulate_error": True})
        except NeuromorphicHardwareError as e:
            logger.info(f"Caught expected error: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    # Create hardware driver with error configuration
    error_config = {"device_id": "test-device-2", "simulate_error": True}
    error_driver = EnhancedHardwareDriver("loihi", error_config)
    
    try:
        # This should trigger error handling and retries
        error_driver.initialize()
    except NeuromorphicHardwareError as e:
        logger.info(f"Caught expected initialization error: {e}")
    
    logger.info("Hardware error handling example completed")


if __name__ == "__main__":
    run_hardware_example()
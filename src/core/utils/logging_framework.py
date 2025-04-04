#!/usr/bin/env python3
"""
Logging framework for the Oblivion project.
Provides consistent logging across all modules.
"""

import os
import sys
import logging
from typing import Optional

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

# Create logs directory if it doesn't exist
logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../logs'))
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)


def get_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Logger name
        log_level: Logging level (defaults to INFO)
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # Set log level
    level = log_level if log_level is not None else DEFAULT_LOG_LEVEL
    logger.setLevel(level)
    
    # Create handlers if they don't exist
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(logs_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    
    return logger


def set_global_log_level(log_level: int) -> None:
    """
    Set the log level for all loggers.
    
    Args:
        log_level: Logging level
    """
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)


# Initialize root logger
root_logger = get_logger("root")
root_logger.info("Logging framework initialized")
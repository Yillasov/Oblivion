"""
Hardware Configuration Integration

Provides simplified interfaces for using the unified configuration system
from other parts of the codebase.
"""

from typing import Dict, Any, Optional, Union, List
import os

from src.core.hardware.unified_config_manager import (
    UnifiedConfigManager, ConfigCategory, HardwareType
)
from src.core.utils.logging_framework import get_logger

logger = get_logger("config_integration")


def get_hardware_config(hardware_type: str, config_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get hardware configuration.
    
    Args:
        hardware_type: Hardware type
        config_name: Configuration name (if None, returns active configuration)
        
    Returns:
        Optional[Dict[str, Any]]: Configuration data or None if not found
    """
    config_manager = UnifiedConfigManager.get_instance()
    
    if config_name:
        return config_manager.load_config(hardware_type, config_name, ConfigCategory.HARDWARE)
    else:
        return config_manager.get_active_config(hardware_type)


def set_hardware_config(hardware_type: str, config_name: str) -> bool:
    """
    Set active hardware configuration.
    
    Args:
        hardware_type: Hardware type
        config_name: Configuration name
        
    Returns:
        bool: Success status
    """
    config_manager = UnifiedConfigManager.get_instance()
    return config_manager.set_active_config(hardware_type, config_name, ConfigCategory.HARDWARE)


def create_hardware_config(hardware_type: str, 
                          config_name: str, 
                          template_name: Optional[str] = "default",
                          overrides: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Create a new hardware configuration.
    
    Args:
        hardware_type: Hardware type
        config_name: Configuration name
        template_name: Template name (default: "default")
        overrides: Optional parameter overrides
        
    Returns:
        Optional[Dict[str, Any]]: New configuration or None if failed
    """
    config_manager = UnifiedConfigManager.get_instance()
    
    if template_name:
        return config_manager.create_config_from_template(
            hardware_type, template_name, config_name, ConfigCategory.HARDWARE, overrides
        )
    else:
        # Create from scratch
        if not overrides:
            logger.error("Must provide either a template or overrides")
            return None
            
        success = config_manager.save_config(hardware_type, config_name, overrides, ConfigCategory.HARDWARE)
        
        if success:
            return overrides
        return None


def get_available_hardware_types() -> List[str]:
    """
    Get list of available hardware types.
    
    Returns:
        List[str]: Available hardware types
    """
    return HardwareType.list()


def get_available_configs(hardware_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get available configurations.
    
    Args:
        hardware_type: Optional hardware type filter
        
    Returns:
        Dict[str, List[str]]: Hardware types and their configurations
    """
    config_manager = UnifiedConfigManager.get_instance()
    return config_manager.list_configs(hardware_type, ConfigCategory.HARDWARE)


def convert_config_between_hardware(source_hardware: str, 
                                   source_config_name: str,
                                   target_hardware: str,
                                   target_config_name: str) -> bool:
    """
    Convert configuration between hardware types.
    
    Args:
        source_hardware: Source hardware type
        source_config_name: Source configuration name
        target_hardware: Target hardware type
        target_config_name: Target configuration name
        
    Returns:
        bool: Success status
    """
    config_manager = UnifiedConfigManager.get_instance()
    
    # Load source configuration
    source_config = config_manager.load_config(source_hardware, source_config_name, ConfigCategory.HARDWARE)
    
    if not source_config:
        logger.error(f"Source configuration '{source_config_name}' not found")
        return False
    
    # Convert configuration
    target_config = config_manager.convert_config(
        source_hardware, target_hardware, source_config
    )
    
    # Save target configuration
    return config_manager.save_config(
        target_hardware, target_config_name, target_config, ConfigCategory.HARDWARE
    )


def initialize_hardware_with_config(hardware_interface, 
                                   hardware_type: str, 
                                   config_name: Optional[str] = None) -> bool:
    """
    Initialize hardware with configuration.
    
    Args:
        hardware_interface: Hardware interface object
        hardware_type: Hardware type
        config_name: Configuration name (if None, uses active configuration)
        
    Returns:
        bool: Success status
    """
    # Get configuration
    config = get_hardware_config(hardware_type, config_name)
    
    if not config:
        logger.error(f"Configuration not found for {hardware_type}")
        return False
    
    # Initialize hardware
    try:
        return hardware_interface.initialize(config)
    except Exception as e:
        logger.error(f"Failed to initialize hardware: {str(e)}")
        return False
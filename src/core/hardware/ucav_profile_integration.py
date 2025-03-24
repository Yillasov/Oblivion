"""
UCAV Hardware Profile Integration

Integrates UCAV hardware profiles with the unified configuration system.
"""

from typing import Dict, Any, Optional, List
from src.core.hardware.unified_config_manager import UnifiedConfigManager, ConfigCategory
from src.core.hardware.ucav_profiles import UCAVHardwareProfiles
from src.core.utils.logging_framework import get_logger

logger = get_logger("ucav_profile_integration")

def register_ucav_profiles():
    """Register all UCAV hardware profiles with the configuration manager."""
    config_manager = UnifiedConfigManager.get_instance()
    
    # Get all profiles
    profile_list = UCAVHardwareProfiles.get_profile_list()
    
    # Register each profile as a template
    for hardware_type, profiles in profile_list.items():
        for profile_name in profiles:
            profile = UCAVHardwareProfiles.get_profile(hardware_type, profile_name)
            if profile:
                # Use save_template instead of register_template
                # Add ucav_ prefix to distinguish from general templates
                template_name = f"ucav_{profile_name}"
                config_manager.save_template(hardware_type, template_name, profile)
                logger.info(f"Registered UCAV profile template: {hardware_type}/{template_name}")
    
    logger.info("UCAV hardware profiles registered successfully")

def get_ucav_profile(hardware_type: str, profile_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a UCAV hardware profile.
    
    Args:
        hardware_type: Hardware type
        profile_name: Profile name (without ucav_ prefix)
        
    Returns:
        Optional[Dict[str, Any]]: Profile configuration or None if not found
    """
    return UCAVHardwareProfiles.get_profile(hardware_type, profile_name)

def create_ucav_config(hardware_type: str, 
                      profile_name: str, 
                      config_name: str,
                      overrides: Optional[Dict[str, Any]] = None) -> bool:
    """
    Create a configuration from a UCAV profile.
    
    Args:
        hardware_type: Hardware type
        profile_name: Profile name
        config_name: Name for the new configuration
        overrides: Optional parameter overrides
        
    Returns:
        bool: Success status
    """
    config_manager = UnifiedConfigManager.get_instance()
    
    # Get profile
    profile = UCAVHardwareProfiles.get_profile(hardware_type, profile_name)
    if not profile:
        logger.error(f"UCAV profile not found: {hardware_type}/{profile_name}")
        return False
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if key in profile:
                profile[key] = value
            elif "ucav_specific" in profile and key in profile["ucav_specific"]:
                profile["ucav_specific"][key] = value
    
    # Save as configuration
    success = config_manager.save_config(hardware_type, config_name, profile, ConfigCategory.HARDWARE)
    
    if success:
        logger.info(f"Created UCAV configuration '{config_name}' from profile '{profile_name}'")
    else:
        logger.error(f"Failed to create UCAV configuration from profile")
    
    return success

def list_ucav_profiles() -> Dict[str, List[str]]:
    """
    List available UCAV profiles.
    
    Returns:
        Dict[str, List[str]]: Hardware types and their UCAV profiles
    """
    return UCAVHardwareProfiles.get_profile_list()
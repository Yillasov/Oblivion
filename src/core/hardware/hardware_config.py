"""
Enhanced Hardware Configuration Management System

Provides simplified interfaces for storing and retrieving hardware settings.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

from src.core.utils.logging_framework import get_logger
from src.core.hardware.config_validation import ConfigValidator

logger = get_logger("hardware_config")


class HardwareConfigStore:
    """Simplified hardware configuration storage and retrieval."""
    
    def __init__(self, config_dir: str = "/Users/yessine/Oblivion/configs/hardware"):
        """
        Initialize hardware configuration store.
        
        Args:
            config_dir: Directory to store hardware configurations
        """
        self.config_dir = config_dir
        self.active_configs = {}
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Create hardware type directories
        for hw_type in ["loihi", "truenorth", "spinnaker", "simulated"]:
            os.makedirs(os.path.join(config_dir, hw_type), exist_ok=True)
    
    # Add these methods to the HardwareConfigStore class
    def validate_config(self, hardware_type: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate hardware configuration.
        
        Args:
            hardware_type: Hardware type
            config: Configuration to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        return ConfigValidator.validate(hardware_type, config)
    
    def save_config(self, hardware_type: str, name: str, config: Dict[str, Any]) -> bool:
        """
        Save hardware configuration.
        
        Args:
            hardware_type: Hardware type (loihi, truenorth, etc.)
            name: Configuration name
            config: Configuration data
            
        Returns:
            bool: Success status
        """
        # Validate configuration before saving
        is_valid, errors = self.validate_config(hardware_type, config)
        if not is_valid:
            logger.error(f"Invalid configuration for {hardware_type}: {', '.join(errors)}")
            return False
        
        try:
            # Add metadata
            config_with_meta = config.copy()
            config_with_meta["_metadata"] = {
                "hardware_type": hardware_type,
                "name": name,
                "created": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # Save to file
            path = os.path.join(self.config_dir, hardware_type, f"{name}.json")
            with open(path, "w") as f:
                json.dump(config_with_meta, f, indent=2)
                
            logger.info(f"Saved {hardware_type} configuration '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def load_config(self, hardware_type: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Load hardware configuration.
        
        Args:
            hardware_type: Hardware type (loihi, truenorth, etc.)
            name: Configuration name
            
        Returns:
            Optional[Dict[str, Any]]: Configuration data or None if not found
        """
        path = os.path.join(self.config_dir, hardware_type, f"{name}.json")
        
        if not os.path.exists(path):
            logger.warning(f"Configuration not found: {hardware_type}/{name}")
            return None
            
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return None
    
    def list_configs(self, hardware_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available configurations.
        
        Args:
            hardware_type: Optional hardware type filter
            
        Returns:
            Dict[str, List[str]]: Hardware types and their configurations
        """
        result = {}
        
        if hardware_type:
            # List configs for specific hardware
            hw_dir = os.path.join(self.config_dir, hardware_type)
            if os.path.exists(hw_dir):
                configs = [f[:-5] for f in os.listdir(hw_dir) if f.endswith(".json")]
                result[hardware_type] = configs
        else:
            # List all configs
            for hw_type in os.listdir(self.config_dir):
                hw_dir = os.path.join(self.config_dir, hw_type)
                if os.path.isdir(hw_dir):
                    configs = [f[:-5] for f in os.listdir(hw_dir) if f.endswith(".json")]
                    result[hw_type] = configs
        
        return result
    
    def set_active_config(self, hardware_type: str, name: str) -> bool:
        """
        Set active configuration for hardware type.
        
        Args:
            hardware_type: Hardware type
            name: Configuration name
            
        Returns:
            bool: Success status
        """
        config = self.load_config(hardware_type, name)
        if not config:
            return False
            
        self.active_configs[hardware_type] = {
            "name": name,
            "config": config
        }
        
        logger.info(f"Set active configuration for {hardware_type}: {name}")
        return True
    
    def get_active_config(self, hardware_type: str) -> Optional[Dict[str, Any]]:
        """
        Get active configuration for hardware type.
        
        Args:
            hardware_type: Hardware type
            
        Returns:
            Optional[Dict[str, Any]]: Active configuration or None
        """
        if hardware_type not in self.active_configs:
            return None
            
        return self.active_configs[hardware_type]["config"]
    
    def create_default_configs(self):
        """Create default configurations for all hardware types."""
        # Loihi default config
        loihi_config = {
            "board_id": 0,
            "chip_id": 0,
            "neurons_per_core": 1024,
            "cores_per_chip": 128,
            "monitoring": {
                "enabled": True,
                "interval_ms": 100
            }
        }
        self.save_config("loihi", "default", loihi_config)
        
        # SpiNNaker default config
        spinnaker_config = {
            "board_address": "192.168.1.1",
            "neurons_per_core": 255,
            "cores_per_chip": 16,
            "monitoring": {
                "enabled": True,
                "interval_ms": 200
            }
        }
        self.save_config("spinnaker", "default", spinnaker_config)
        
        # TrueNorth default config
        truenorth_config = {
            "board_id": 0,
            "neurons_per_core": 256,
            "cores_per_chip": 4096,
            "monitoring": {
                "enabled": True,
                "interval_ms": 500
            }
        }
        self.save_config("truenorth", "default", truenorth_config)
        
        # Simulated hardware config
        simulated_config = {
            "hardware_type": "simulated",
            "neurons_per_core": 1000,
            "cores_per_chip": 16,
            "chips_available": 4,
            "monitoring": {
                "enabled": True,
                "interval_ms": 50
            }
        }
        self.save_config("simulated", "default", simulated_config)


# Global instance
config_store = HardwareConfigStore()
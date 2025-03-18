"""
Hardware Configuration Migration System

Provides utilities for migrating configurations between different hardware types.
"""

import os
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from src.core.utils.logging_framework import get_logger
from src.core.hardware.hardware_config import config_store
from src.core.hardware.config_validation import ConfigValidator

logger = get_logger("config_migration")


class ConfigMigration:
    """Handles migration of configurations between hardware types."""
    
    @staticmethod
    def migrate_config(source_type: str, source_name: str, 
                      target_type: str, target_name: str) -> bool:
        """
        Migrate configuration from one hardware type to another.
        
        Args:
            source_type: Source hardware type
            source_name: Source configuration name
            target_type: Target hardware type
            target_name: Target configuration name
            
        Returns:
            bool: Success status
        """
        # Load source configuration
        source_config = config_store.load_config(source_type, source_name)
        if not source_config:
            logger.error(f"Source configuration not found: {source_type}/{source_name}")
            return False
        
        # Transform configuration
        target_config = ConfigMigration.transform_config(source_config, source_type, target_type)
        if not target_config:
            logger.error(f"Failed to transform configuration from {source_type} to {target_type}")
            return False
        
        # Save target configuration
        return config_store.save_config(target_type, target_name, target_config)
    
    @staticmethod
    def transform_config(config: Dict[str, Any], source_type: str, 
                        target_type: str) -> Optional[Dict[str, Any]]:
        """
        Transform configuration between hardware types.
        
        Args:
            config: Source configuration
            source_type: Source hardware type
            target_type: Target hardware type
            
        Returns:
            Optional[Dict[str, Any]]: Transformed configuration or None if not possible
        """
        # Get transformation function
        transform_func = ConfigMigration._get_transform_function(source_type, target_type)
        if not transform_func:
            logger.error(f"No transformation available from {source_type} to {target_type}")
            return None
        
        # Apply transformation
        try:
            target_config = transform_func(config)
            
            # Set hardware type
            target_config["hardware_type"] = target_type
            
            # Add migration metadata
            if "_metadata" not in target_config:
                target_config["_metadata"] = {}
                
            target_config["_metadata"]["migrated_from"] = {
                "hardware_type": source_type,
                "name": config.get("_metadata", {}).get("name", "unknown"),
                "migration_date": datetime.now().isoformat()
            }
            
            return target_config
        except Exception as e:
            logger.error(f"Error transforming configuration: {str(e)}")
            return None
    
    @staticmethod
    def _get_transform_function(source_type: str, target_type: str):
        """Get appropriate transformation function."""
        # Define transformation map
        transform_map = {
            ("loihi", "spinnaker"): ConfigMigration._loihi_to_spinnaker,
            ("loihi", "truenorth"): ConfigMigration._loihi_to_truenorth,
            ("loihi", "simulated"): ConfigMigration._loihi_to_simulated,
            ("spinnaker", "loihi"): ConfigMigration._spinnaker_to_loihi,
            ("spinnaker", "truenorth"): ConfigMigration._spinnaker_to_truenorth,
            ("spinnaker", "simulated"): ConfigMigration._spinnaker_to_simulated,
            ("truenorth", "loihi"): ConfigMigration._truenorth_to_loihi,
            ("truenorth", "spinnaker"): ConfigMigration._truenorth_to_spinnaker,
            ("truenorth", "simulated"): ConfigMigration._truenorth_to_simulated,
            ("simulated", "loihi"): ConfigMigration._simulated_to_loihi,
            ("simulated", "spinnaker"): ConfigMigration._simulated_to_spinnaker,
            ("simulated", "truenorth"): ConfigMigration._simulated_to_truenorth,
        }
        
        return transform_map.get((source_type.lower(), target_type.lower()))
    
    # Transformation functions
    
    @staticmethod
    def _loihi_to_spinnaker(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Loihi configuration to SpiNNaker."""
        result = {
            "board_address": "192.168.1.1",  # Default SpiNNaker address
            "neurons_per_core": min(255, config.get("neurons_per_core", 1024)),  # SpiNNaker limit
            "cores_per_chip": 16,  # SpiNNaker standard
            "monitoring": config.get("monitoring", {"enabled": True, "interval_ms": 200})
        }
        
        # Copy common fields
        for field in ["debug_mode", "logging_level", "simulation"]:
            if field in config:
                result[field] = config[field]
        
        return result
    
    @staticmethod
    def _loihi_to_truenorth(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Loihi configuration to TrueNorth."""
        result = {
            "board_id": config.get("board_id", 0),
            "neurons_per_core": 256,  # TrueNorth fixed value
            "cores_per_chip": 4096,  # TrueNorth fixed value
            "monitoring": config.get("monitoring", {"enabled": True, "interval_ms": 500})
        }
        
        # Copy common fields
        for field in ["debug_mode", "logging_level", "simulation"]:
            if field in config:
                result[field] = config[field]
        
        return result
    
    @staticmethod
    def _loihi_to_simulated(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Loihi configuration to Simulated."""
        result = {
            "neurons_per_core": config.get("neurons_per_core", 1024),
            "cores_per_chip": config.get("cores_per_chip", 128),
            "chips_available": 4,  # Default for simulation
            "monitoring": config.get("monitoring", {"enabled": True, "interval_ms": 50}),
            "simulation_speed": "balanced"
        }
        
        # Copy common fields
        for field in ["debug_mode", "logging_level", "simulation"]:
            if field in config:
                result[field] = config[field]
        
        return result
    
    @staticmethod
    def _spinnaker_to_loihi(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SpiNNaker configuration to Loihi."""
        result = {
            "board_id": 0,
            "chip_id": 0,
            "neurons_per_core": 1024,  # Loihi standard
            "cores_per_chip": 128,  # Loihi standard
            "monitoring": config.get("monitoring", {"enabled": True, "interval_ms": 100})
        }
        
        # Copy common fields
        for field in ["debug_mode", "logging_level", "simulation"]:
            if field in config:
                result[field] = config[field]
        
        return result
    
    # Add other transformation functions with similar patterns
    @staticmethod
    def _spinnaker_to_truenorth(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SpiNNaker configuration to TrueNorth."""
        # Implementation similar to other transformations
        result = {
            "board_id": 0,
            "neurons_per_core": 256,  # TrueNorth fixed value
            "cores_per_chip": 4096,  # TrueNorth fixed value
            "monitoring": config.get("monitoring", {"enabled": True, "interval_ms": 500})
        }
        
        # Copy common fields
        for field in ["debug_mode", "logging_level", "simulation"]:
            if field in config:
                result[field] = config[field]
        
        return result
    
    @staticmethod
    def _spinnaker_to_simulated(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SpiNNaker configuration to Simulated."""
        # Implementation similar to other transformations
        return ConfigMigration._generic_to_simulated(config)
    
    @staticmethod
    def _truenorth_to_loihi(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform TrueNorth configuration to Loihi."""
        # Implementation similar to other transformations
        return ConfigMigration._generic_to_loihi(config)
    
    @staticmethod
    def _truenorth_to_spinnaker(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform TrueNorth configuration to SpiNNaker."""
        # Implementation similar to other transformations
        return ConfigMigration._generic_to_spinnaker(config)
    
    @staticmethod
    def _truenorth_to_simulated(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform TrueNorth configuration to Simulated."""
        # Implementation similar to other transformations
        return ConfigMigration._generic_to_simulated(config)
    
    @staticmethod
    def _simulated_to_loihi(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Simulated configuration to Loihi."""
        # Implementation similar to other transformations
        return ConfigMigration._generic_to_loihi(config)
    
    @staticmethod
    def _simulated_to_spinnaker(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Simulated configuration to SpiNNaker."""
        # Implementation similar to other transformations
        return ConfigMigration._generic_to_spinnaker(config)
    
    @staticmethod
    def _simulated_to_truenorth(config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Simulated configuration to TrueNorth."""
        # Implementation similar to other transformations
        return ConfigMigration._generic_to_truenorth(config)
    
    # Generic transformation helpers
    @staticmethod
    def _generic_to_loihi(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic transformation to Loihi."""
        result = {
            "board_id": 0,
            "chip_id": 0,
            "neurons_per_core": 1024,
            "cores_per_chip": 128,
            "monitoring": config.get("monitoring", {"enabled": True, "interval_ms": 100})
        }
        
        # Copy common fields
        for field in ["debug_mode", "logging_level", "simulation"]:
            if field in config:
                result[field] = config[field]
        
        return result
    
    @staticmethod
    def _generic_to_spinnaker(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic transformation to SpiNNaker."""
        result = {
            "board_address": "192.168.1.1",
            "neurons_per_core": 255,
            "cores_per_chip": 16,
            "monitoring": config.get("monitoring", {"enabled": True, "interval_ms": 200})
        }
        
        # Copy common fields
        for field in ["debug_mode", "logging_level", "simulation"]:
            if field in config:
                result[field] = config[field]
        
        return result
    
    @staticmethod
    def _generic_to_truenorth(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic transformation to TrueNorth."""
        result = {
            "board_id": 0,
            "neurons_per_core": 256,
            "cores_per_chip": 4096,
            "monitoring": config.get("monitoring", {"enabled": True, "interval_ms": 500})
        }
        
        # Copy common fields
        for field in ["debug_mode", "logging_level", "simulation"]:
            if field in config:
                result[field] = config[field]
        
        return result
    
    @staticmethod
    def _generic_to_simulated(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic transformation to Simulated."""
        result = {
            "neurons_per_core": 1000,
            "cores_per_chip": 16,
            "chips_available": 4,
            "monitoring": config.get("monitoring", {"enabled": True, "interval_ms": 50}),
            "simulation_speed": "balanced"
        }
        
        # Copy common fields
        for field in ["debug_mode", "logging_level", "simulation"]:
            if field in config:
                result[field] = config[field]
        
        return result
#!/usr/bin/env python3
"""
Sensor Configuration Manager
Handles sensor configuration loading, validation, and updates.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from src.simulation.sensors.sensor_framework import SensorConfig, SensorType
from src.core.sdk.sensor_connector import get_sensor_connector

logger = logging.getLogger(__name__)

class SensorConfigManager:
    def __init__(self, config_path: str = "configs/sensors"):
        self.config_path = config_path
        self.configs: Dict[str, Dict[str, Any]] = {}
        self._load_configs()
    
    def _load_configs(self) -> None:
        """Load sensor configurations from files."""
        os.makedirs(self.config_path, exist_ok=True)
        
        for filename in os.listdir(self.config_path):
            if filename.endswith('.json'):
                with open(os.path.join(self.config_path, filename), 'r') as f:
                    self.configs[filename[:-5]] = json.load(f)
    
    def get_config(self, sensor_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific sensor."""
        return self.configs.get(sensor_name)
    
    def update_config(self, sensor_name: str, updates: Dict[str, Any]) -> bool:
        """Update sensor configuration."""
        if sensor_name not in self.configs:
            return False
        
        # Update configuration
        self.configs[sensor_name].update(updates)
        
        # Save to file
        config_file = os.path.join(self.config_path, f"{sensor_name}.json")
        with open(config_file, 'w') as f:
            json.dump(self.configs[sensor_name], f, indent=2)
        
        # Update running sensor if exists
        connector = get_sensor_connector()
        return connector.update_sensor_config(sensor_name, updates)
    
    def create_sensor_config(self, name: str, sensor_type: SensorType, 
                           params: Dict[str, Any]) -> SensorConfig:
        """Create a new sensor configuration."""
        config = SensorConfig(
            type=sensor_type,
            name=name,
            **params
        )
        
        # Save configuration
        self.configs[name] = asdict(config)
        config_file = os.path.join(self.config_path, f"{name}.json")
        with open(config_file, 'w') as f:
            json.dump(self.configs[name], f, indent=2)
        
        return config

# Global instance
_config_manager = None

def get_config_manager() -> SensorConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = SensorConfigManager()
    return _config_manager
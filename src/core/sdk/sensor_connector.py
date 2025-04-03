#!/usr/bin/env python3
"""
Sensor Connector Module

Provides an interface between the Oblivion SDK and the sensor simulation environment.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import os
import json

from src.simulation.environment.sensor_sim_env import SensorSimEnvironment, SensorSimConfig
from src.simulation.sensors.sensor_framework import SensorType, SensorConfig, Sensor
from src.core.signal.neuromorphic_signal import NeuromorphicSignalProcessor

logger = logging.getLogger(__name__)


class SensorSDKConnector:
    """
    Connects the Oblivion SDK to the sensor simulation environment.
    Provides methods to access sensor data and control sensors.
    """
    
    def __init__(self, sim_env: Optional[SensorSimEnvironment] = None):
        """Initialize the sensor connector."""
        self.sim_env = sim_env or SensorSimEnvironment()
        self.sensor_cache = {}
        self.processor_cache = {}
        self.last_update_time = 0.0
        logger.info("Sensor SDK connector initialized")
    
    def get_available_sensors(self) -> List[str]:
        """Get list of available sensor names."""
        if not self.sim_env or not self.sim_env.sensor_manager:
            return []
        return list(self.sim_env.sensor_manager.sensors.keys())
    
    def get_sensor_data(self, sensor_name: str) -> Dict[str, Any]:
        """Get data from a specific sensor."""
        if not self.sim_env or not self.sim_env.sensor_manager:
            return {}
            
        sensor = self.sim_env.sensor_manager.get_sensor(sensor_name)
        if not sensor:
            logger.warning(f"Sensor {sensor_name} not found")
            return {}
            
        return sensor.data
    
    def get_all_sensor_data(self) -> Dict[str, Dict[str, Any]]:
        """Get data from all sensors."""
        if not self.sim_env or not self.sim_env.sensor_manager:
            return {}
            
        result = {}
        for name, sensor in self.sim_env.sensor_manager.sensors.items():
            result[name] = sensor.data
            
        return result
    
    def get_processed_data(self, sensor_name: str) -> np.ndarray:
        """Get processed data from a specific sensor."""
        if not self.sim_env or sensor_name not in self.sim_env.signal_processors:
            return np.array([])
            
        # Get the latest processed data if available
        if sensor_name in self.processor_cache:
            return self.processor_cache[sensor_name]
            
        # Otherwise return empty array
        return np.array([])
    
    def get_all_processed_data(self) -> Dict[str, np.ndarray]:
        """Get processed data from all sensors."""
        if not self.sim_env:
            return {}
            
        return self.processor_cache
    
    def get_fusion_data(self) -> Dict[str, Any]:
        """Get the latest sensor fusion data."""
        if not self.sim_env or not self.sim_env.recorded_data["fusion_data"]:
            return {}
            
        # Return the latest fusion data
        return self.sim_env.recorded_data["fusion_data"][-1]["data"]
    
    def add_sensor(self, config: SensorConfig) -> bool:
        """Add a new sensor to the simulation."""
        if not self.sim_env:
            return False
            
        try:
            # Create sensor based on type
            sensor = Sensor(config)
            self.sim_env.add_sensor(sensor)
            return True
        except Exception as e:
            logger.error(f"Failed to add sensor: {e}")
            return False
    
    def remove_sensor(self, sensor_name: str) -> bool:
        """Remove a sensor from the simulation."""
        if not self.sim_env or not self.sim_env.sensor_manager:
            return False
            
        try:
            self.sim_env.sensor_manager.remove_sensor(sensor_name)
            if sensor_name in self.sim_env.signal_processors:
                del self.sim_env.signal_processors[sensor_name]
            return True
        except Exception as e:
            logger.error(f"Failed to remove sensor: {e}")
            return False
    
    def update_sensor_config(self, sensor_name: str, config_updates: Dict[str, Any]) -> bool:
        """Update configuration of an existing sensor."""
        if not self.sim_env or not self.sim_env.sensor_manager:
            return False
            
        sensor = self.sim_env.sensor_manager.get_sensor(sensor_name)
        if not sensor:
            logger.warning(f"Sensor {sensor_name} not found")
            return False
            
        try:
            # Update sensor configuration
            for key, value in config_updates.items():
                if hasattr(sensor.config, key):
                    setattr(sensor.config, key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to update sensor config: {e}")
            return False
    
    def update_sensors(self, time_now: float, platform_state: Dict[str, Any], 
                  environment: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Update all sensors with new platform and environment data.
        
        This method can be called from the main SDK loop to update sensor readings.
        """
        if not self.sim_env or not self.sim_env.sensor_manager:
            logger.warning("Simulation environment or sensor manager not initialized")
            return {}
            
        try:
            # Update all sensors
            raw_data = self.sim_env.sensor_manager.update_all(
                time_now, platform_state, environment
            )
            
            # Process sensor data
            for sensor_name, sensor_data in raw_data.items():
                if sensor_name in self.sim_env.signal_processors:
                    processor = self.sim_env.signal_processors[sensor_name]
                    try:
                        data_array = np.array(sensor_data.get("data", []))
                        if data_array.size > 0:  # Check if data is not empty
                            self.processor_cache[sensor_name] = processor.process(data_array)
                    except Exception as e:
                        logger.error(f"Error processing data for sensor {sensor_name}: {e}")
            
            # Update fusion system
            processed_data = {}
            for sensor_name, processed in self.processor_cache.items():
                processed_data[sensor_name] = {
                    "processed": processed,
                    "original": raw_data.get(sensor_name, {})
                }
                
            fusion_data = self.sim_env.fusion_system.process(processed_data, time_now)
            
            # Cache the update time
            self.last_update_time = time_now
            
            return raw_data
        except Exception as e:
            logger.error(f"Error updating sensors: {e}")
            return {}


# Create a singleton instance for easy access
_default_connector = None

def get_sensor_connector() -> SensorSDKConnector:
    """Get the default sensor connector instance."""
    global _default_connector
    if _default_connector is None:
        _default_connector = SensorSDKConnector()
    return _default_connector
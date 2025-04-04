#!/usr/bin/env python3
"""
Flexible Sensor Integration Framework for Biomimetic Systems.
Provides a unified interface for connecting various sensor types to biomimetic controllers.
"""

import os
import sys
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable, Protocol, Union
from dataclasses import dataclass

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.simulation.sensors.sensor_framework import SensorType, SensorConfig, Sensor
from src.core.fusion.sensor_fusion import SensorFusion, FusionConfig

logger = get_logger("sensor_integration")


class SensorDataType(Enum):
    """Types of sensor data for biomimetic systems."""
    PROPRIOCEPTIVE = "proprioceptive"  # Body position/movement
    EXTEROCEPTIVE = "exteroceptive"    # External environment
    INTEROCEPTIVE = "interoceptive"    # Internal state
    TACTILE = "tactile"                # Touch/pressure
    VISUAL = "visual"                  # Vision
    AUDITORY = "auditory"              # Sound
    CHEMICAL = "chemical"              # Chemical sensing
    THERMAL = "thermal"                # Temperature
    ELECTROMAGNETIC = "electromagnetic" # EM field sensing
    FLOW = "flow"                      # Air/fluid flow


@dataclass
class SensorMapping:
    """Maps a sensor to a specific biomimetic component."""
    sensor_id: str
    sensor_type: SensorType
    data_type: SensorDataType
    target_component: str
    data_field: str
    scaling_factor: float = 1.0
    offset: float = 0.0
    threshold: float = 0.0
    filter_function: Optional[Callable] = None
    is_enabled: bool = True


class BiomimeticSensorInterface:
    """Interface for connecting sensors to biomimetic systems."""
    
    def __init__(self, fusion_config: Optional[FusionConfig] = None):
        """
        Initialize the biomimetic sensor interface.
        
        Args:
            fusion_config: Optional configuration for sensor fusion
        """
        self.sensors: Dict[str, Sensor] = {}
        self.mappings: List[SensorMapping] = []
        self.fusion_system = SensorFusion(fusion_config or FusionConfig())
        self.last_readings: Dict[str, Dict[str, Any]] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        
        logger.info("Biomimetic sensor interface initialized")
    
    def add_sensor(self, sensor: Sensor) -> bool:
        """
        Add a sensor to the interface.
        
        Args:
            sensor: The sensor to add
            
        Returns:
            Success status
        """
        if sensor.config.name in self.sensors:
            logger.warning(f"Sensor {sensor.config.name} already exists")
            return False
        
        self.sensors[sensor.config.name] = sensor
        self.last_readings[sensor.config.name] = {}
        logger.info(f"Added sensor: {sensor.config.name} ({sensor.config.type.name})")
        return True
    
    def add_mapping(self, mapping: SensorMapping) -> bool:
        """
        Add a mapping between a sensor and a biomimetic component.
        
        Args:
            mapping: The sensor mapping to add
            
        Returns:
            Success status
        """
        if mapping.sensor_id not in self.sensors:
            logger.warning(f"Sensor {mapping.sensor_id} not found")
            return False
        
        self.mappings.append(mapping)
        logger.info(f"Added mapping: {mapping.sensor_id} -> {mapping.target_component}")
        return True
    
    def register_callback(self, component_id: str, callback: Callable) -> bool:
        """
        Register a callback for a biomimetic component.
        
        Args:
            component_id: ID of the component
            callback: Callback function to register
            
        Returns:
            Success status
        """
        if component_id not in self.callbacks:
            self.callbacks[component_id] = []
        
        self.callbacks[component_id].append(callback)
        logger.info(f"Registered callback for component: {component_id}")
        return True
    
    def update(self, platform_state: Dict[str, Any], environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update all sensors and process mappings.
        
        Args:
            platform_state: Current platform state
            environment: Current environment state
            
        Returns:
            Processed sensor data
        """
        # Update all sensors
        raw_data = {}
        for sensor_id, sensor in self.sensors.items():
            try:
                # Fix: The Sensor.update() method might have a different signature
                # Let's try to inspect the method and call it correctly
                import inspect
                update_params = inspect.signature(sensor.update).parameters
                
                if len(update_params) == 3:  # If it expects self, platform_state, environment
                    sensor_data = sensor.update(platform_state, environment)
                elif len(update_params) == 2:  # If it expects self and a combined state
                    # Combine platform_state and environment into a single state dict
                    combined_state = {**platform_state, "environment": environment}
                    sensor_data = sensor.update(combined_state)
                else:
                    logger.warning(f"Unexpected signature for sensor.update in {sensor_id}")
                    sensor_data = {}
                    
                self.last_readings[sensor_id] = sensor_data
                raw_data[sensor_id] = sensor_data
            except Exception as e:
                logger.error(f"Error updating sensor {sensor_id}: {e}")
                raw_data[sensor_id] = {}
        
        # Apply sensor fusion
        fused_data = self.fusion_system.process(raw_data, platform_state.get("time", 0.0))
        
        # Process mappings and trigger callbacks
        processed_data = self._process_mappings(raw_data, fused_data)
        
        return {
            "raw": raw_data,
            "fused": fused_data,
            "processed": processed_data
        }
    
    def _process_mappings(self, raw_data: Dict[str, Any], 
                         fused_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Process sensor mappings and trigger callbacks.
        
        Args:
            raw_data: Raw sensor data
            fused_data: Fused sensor data
            
        Returns:
            Processed data by component
        """
        processed_data: Dict[str, Dict[str, Any]] = {}
        
        # Process each mapping
        for mapping in self.mappings:
            if not mapping.is_enabled or mapping.sensor_id not in raw_data:
                continue
            
            # Get sensor data
            sensor_data = raw_data[mapping.sensor_id]
            
            # Extract relevant field
            if mapping.data_field in sensor_data:
                value = sensor_data[mapping.data_field]
                
                # Apply filter function if provided
                if mapping.filter_function:
                    value = mapping.filter_function(value)
                
                # Apply scaling and offset
                value = value * mapping.scaling_factor + mapping.offset
                
                # Apply threshold
                if abs(value) < mapping.threshold:
                    value = 0.0
                
                # Store processed value
                if mapping.target_component not in processed_data:
                    processed_data[mapping.target_component] = {}
                
                processed_data[mapping.target_component][mapping.data_type.value] = value
                
                # Trigger callback if registered
                if mapping.target_component in self.callbacks:
                    for callback in self.callbacks[mapping.target_component]:
                        callback({mapping.data_type.value: value})
        
        return processed_data
    
    def get_sensor_data(self, sensor_id: str) -> Dict[str, Any]:
        """
        Get the latest data from a specific sensor.
        
        Args:
            sensor_id: ID of the sensor
            
        Returns:
            Latest sensor data
        """
        return self.last_readings.get(sensor_id, {})
    
    def get_component_data(self, component_id: str) -> Dict[str, Any]:
        """
        Get all sensor data mapped to a specific component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            Mapped sensor data
        """
        component_data = {}
        
        for mapping in self.mappings:
            if mapping.target_component == component_id and mapping.is_enabled:
                sensor_data = self.get_sensor_data(mapping.sensor_id)
                if mapping.data_field in sensor_data:
                    value = sensor_data[mapping.data_field]
                    
                    # Apply processing
                    if mapping.filter_function:
                        value = mapping.filter_function(value)
                    
                    value = value * mapping.scaling_factor + mapping.offset
                    
                    if abs(value) >= mapping.threshold:
                        component_data[mapping.data_type.value] = value
        
        return component_data


class MuscleProprioceptiveSensor:
    """Proprioceptive sensor for artificial muscles."""
    
    def __init__(self, muscle_id: str, sensor_interface: BiomimeticSensorInterface):
        """
        Initialize muscle proprioceptive sensor.
        
        Args:
            muscle_id: ID of the muscle to monitor
            sensor_interface: Biomimetic sensor interface
        """
        self.muscle_id = muscle_id
        self.sensor_interface = sensor_interface
        
        # Create sensor config
        sensor_config = SensorConfig(
            type=SensorType.BIO_MIMETIC,
            name=f"{muscle_id}_proprioceptive",
            update_rate=100.0,  # 100 Hz
            accuracy=0.98,
            noise_factor=0.01
        )
        
        # Create sensor
        self.sensor = Sensor(sensor_config)
        
        # Add sensor to interface
        self.sensor_interface.add_sensor(self.sensor)
        
        # Create mappings
        self._create_mappings()
        
        logger.info(f"Created proprioceptive sensor for muscle: {muscle_id}")
    
    def _create_mappings(self) -> None:
        """Create sensor mappings for the muscle."""
        # Length mapping
        length_mapping = SensorMapping(
            sensor_id=self.sensor.config.name,
            sensor_type=SensorType.BIO_MIMETIC,
            data_type=SensorDataType.PROPRIOCEPTIVE,
            target_component=self.muscle_id,
            data_field="length",
            scaling_factor=1.0,
            threshold=0.001
        )
        
        # Force mapping
        force_mapping = SensorMapping(
            sensor_id=self.sensor.config.name,
            sensor_type=SensorType.BIO_MIMETIC,
            data_type=SensorDataType.PROPRIOCEPTIVE,
            target_component=self.muscle_id,
            data_field="force",
            scaling_factor=0.01,
            threshold=0.1
        )
        
        # Add mappings
        self.sensor_interface.add_mapping(length_mapping)
        self.sensor_interface.add_mapping(force_mapping)
    
    def update(self, muscle_state: Dict[str, float]) -> None:
        """
        Update sensor with current muscle state.
        
        Args:
            muscle_state: Current muscle state
        """
        try:
            # Update sensor data
            # Create a platform_state and environment dict to match the Sensor.update() signature
            platform_state = {"muscle_state": muscle_state, "time": 0.0}
            environment = {}  # Empty environment as we're only concerned with muscle state
            
            import inspect
            update_params = inspect.signature(self.sensor.update).parameters
            
            if len(update_params) == 3:  # If it expects self, platform_state, environment
                self.sensor.update(platform_state, environment)
            elif len(update_params) == 2:  # If it expects self and a combined state
                # Combine platform_state and environment into a single state dict
                combined_state = {**platform_state, "environment": environment}
                self.sensor.update(combined_state)
            else:
                logger.warning(f"Unexpected signature for sensor.update in {self.sensor.config.name}")
        except Exception as e:
            logger.error(f"Error updating muscle proprioceptive sensor: {e}")


class CPGSensorIntegration:
    """Integrates sensors with CPG controllers."""
    
    def __init__(self, cpg_id: str, sensor_interface: BiomimeticSensorInterface):
        """
        Initialize CPG sensor integration.
        
        Args:
            cpg_id: ID of the CPG controller
            sensor_interface: Biomimetic sensor interface
        """
        self.cpg_id = cpg_id
        self.sensor_interface = sensor_interface
        
        # Register callback
        self.sensor_interface.register_callback(cpg_id, self._sensor_callback)
        
        logger.info(f"Created sensor integration for CPG: {cpg_id}")
    
    def _sensor_callback(self, sensor_data: Dict[str, Any]) -> None:
        """
        Process sensor data for CPG controller.
        
        Args:
            sensor_data: Sensor data
        """
        # This would be implemented to modify CPG parameters based on sensor data
        # For example, adjusting oscillation frequency or phase based on sensory feedback
        pass
    
    def add_environmental_sensing(self, sensor_id: str) -> bool:
        """
        Add environmental sensing to CPG controller.
        
        Args:
            sensor_id: ID of the environmental sensor
            
        Returns:
            Success status
        """
        if sensor_id not in self.sensor_interface.sensors:
            logger.warning(f"Sensor {sensor_id} not found")
            return False
        
        # Create mapping for environmental sensing
        mapping = SensorMapping(
            sensor_id=sensor_id,
            sensor_type=self.sensor_interface.sensors[sensor_id].config.type,
            data_type=SensorDataType.EXTEROCEPTIVE,
            target_component=self.cpg_id,
            data_field="data",
            scaling_factor=0.5,
            threshold=0.1
        )
        
        return self.sensor_interface.add_mapping(mapping)


def create_wing_sensor_system(muscle_controller_id: str) -> BiomimeticSensorInterface:
    """
    Create a complete wing sensor system.
    
    Args:
        muscle_controller_id: ID of the muscle controller
        
    Returns:
        Configured sensor interface
    """
    # Create sensor interface
    fusion_config = FusionConfig(
        grid_size=(10, 10),
        temporal_window=5,
        confidence_threshold=0.2
    )
    sensor_interface = BiomimeticSensorInterface(fusion_config)
    
    # Create air flow sensor
    air_flow_config = SensorConfig(
        type=SensorType.BIO_MIMETIC,
        name="wing_air_flow",
        update_rate=50.0,
        fov_horizontal=180.0,
        fov_vertical=180.0,
        accuracy=0.9,
        noise_factor=0.05
    )
    air_flow_sensor = Sensor(air_flow_config)
    sensor_interface.add_sensor(air_flow_sensor)
    
    # Create wing strain sensor
    strain_config = SensorConfig(
        type=SensorType.BIO_MIMETIC,
        name="wing_strain",
        update_rate=100.0,
        accuracy=0.95,
        noise_factor=0.02
    )
    strain_sensor = Sensor(strain_config)
    sensor_interface.add_sensor(strain_sensor)
    
    # Create mappings for air flow sensor
    air_flow_mapping = SensorMapping(
        sensor_id="wing_air_flow",
        sensor_type=SensorType.BIO_MIMETIC,
        data_type=SensorDataType.FLOW,
        target_component=muscle_controller_id,
        data_field="flow_velocity",
        scaling_factor=0.2,
        threshold=0.05
    )
    sensor_interface.add_mapping(air_flow_mapping)
    
    # Create mappings for strain sensor
    strain_mapping = SensorMapping(
        sensor_id="wing_strain",
        sensor_type=SensorType.BIO_MIMETIC,
        data_type=SensorDataType.PROPRIOCEPTIVE,
        target_component=muscle_controller_id,
        data_field="strain",
        scaling_factor=1.0,
        threshold=0.01
    )
    sensor_interface.add_mapping(strain_mapping)
    
    return sensor_interface


if __name__ == "__main__":
    # Example usage
    sensor_interface = create_wing_sensor_system("wing_muscle_controller")
    
    # Simulate platform and environment
    platform_state = {
        "position": [0.0, 0.0, 100.0],
        "velocity": [50.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0],
        "time": 0.0
    }
    
    environment = {
        "wind": [5.0, 2.0, 0.0],
        "temperature": 20.0,
        "pressure": 101.3,
        "targets": []
    }
    
    try:
        # Update sensors
        sensor_data = sensor_interface.update(platform_state, environment)
        
        print("Sensor integration framework test complete")
        print(f"Processed {len(sensor_data['raw'])} sensors")
        
        # Print some sample data
        print("\nSample sensor readings:")
        for sensor_id, data in sensor_data['raw'].items():
            print(f"  {sensor_id}: {data}")
            
    except Exception as e:
        print(f"Error during sensor update: {e}")
        import traceback
        traceback.print_exc()
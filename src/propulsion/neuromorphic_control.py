"""
Neuromorphic control interfaces for propulsion systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.propulsion.base import NeuromorphicPropulsion
from src.core.integration.neuromorphic_system import NeuromorphicSystem


class PropulsionControlInterface(ABC):
    """Simple interface for neuromorphic propulsion control."""
    
    @abstractmethod
    def process_inputs(self, sensor_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process sensor inputs and return control signals."""
        pass
    
    @abstractmethod
    def adapt(self, performance_metrics: Dict[str, float]) -> None:
        """Adapt control parameters based on performance feedback."""
        pass


class NeuromorphicPropulsionController:
    """Controller for neuromorphic propulsion systems."""
    
    def __init__(self, neuromorphic_system: NeuromorphicSystem):
        """Initialize with a neuromorphic system."""
        self.system = neuromorphic_system
        self.propulsion_units: Dict[str, NeuromorphicPropulsion] = {}
        self.control_mappings: Dict[str, Dict[str, Any]] = {}
        
    def register_propulsion_unit(self, unit_id: str, unit: NeuromorphicPropulsion, 
                               input_mapping: Dict[str, str], output_mapping: Dict[str, str]) -> bool:
        """Register a propulsion unit with the controller."""
        if unit_id in self.propulsion_units:
            return False
            
        self.propulsion_units[unit_id] = unit
        self.control_mappings[unit_id] = {
            "inputs": input_mapping,
            "outputs": output_mapping
        }
        return True
        
    def process_control_cycle(self, sensor_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Process a single control cycle and return control outputs."""
        results = {}
        
        for unit_id, unit in self.propulsion_units.items():
            if not unit.initialized:
                continue
                
            # Map sensor data to unit-specific inputs
            unit_inputs = self._map_inputs(unit_id, sensor_data)
            
            # Process data through neuromorphic system
            processed_data = unit.process_data(unit_inputs)
            
            # Apply control outputs
            control_outputs = self._map_outputs(unit_id, processed_data)
            unit.set_power_state(control_outputs)
            
            results[unit_id] = control_outputs
            
        return results
    
    def _map_inputs(self, unit_id: str, sensor_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Map global sensor data to unit-specific inputs."""
        if unit_id not in self.control_mappings:
            return {}
            
        input_mapping = self.control_mappings[unit_id]["inputs"]
        unit_inputs = {}
        
        for unit_input, sensor_key in input_mapping.items():
            if sensor_key in sensor_data:
                unit_inputs[unit_input] = sensor_data[sensor_key]
                
        return unit_inputs
    
    def _map_outputs(self, unit_id: str, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map processed data to control outputs."""
        if unit_id not in self.control_mappings:
            return {}
            
        output_mapping = self.control_mappings[unit_id]["outputs"]
        control_outputs = {}
        
        for control_key, data_key in output_mapping.items():
            if data_key in processed_data:
                control_outputs[control_key] = processed_data[data_key]
                
        return control_outputs
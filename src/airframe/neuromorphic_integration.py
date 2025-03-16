from typing import Dict, Any, List, Callable, Optional
import numpy as np
from abc import ABC, abstractmethod

class NeuromorphicInterface(ABC):
    """Base interface for neuromorphic control system integration."""
    
    @abstractmethod
    def get_sensor_inputs(self) -> Dict[str, np.ndarray]:
        """Get sensor inputs for neuromorphic processing."""
        pass
    
    @abstractmethod
    def apply_control_outputs(self, outputs: Dict[str, np.ndarray]) -> None:
        """Apply control outputs from neuromorphic system to airframe."""
        pass
    
    @abstractmethod
    def get_state_representation(self) -> np.ndarray:
        """Get state representation for neuromorphic learning."""
        pass

class AirframeNeuromorphicAdapter:
    """Adapter between airframe and neuromorphic control systems."""
    
    def __init__(self, airframe_type: str):
        self.airframe_type = airframe_type
        self.sensor_mappings: Dict[str, Callable] = {}
        self.control_mappings: Dict[str, Callable] = {}
        self._initialize_mappings()
    
    def _initialize_mappings(self):
        """Initialize mappings based on airframe type."""
        if self.airframe_type == "morphing_wing":
            self._init_morphing_wing_mappings()
        elif self.airframe_type == "hypersonic":
            self._init_hypersonic_mappings()
        # Add other airframe types as needed
    
    def _init_morphing_wing_mappings(self):
        """Initialize mappings for morphing wing drones."""
        self.sensor_mappings = {
            "wing_shape": lambda state: np.array(state.get("wing_configuration", [0.0, 0.0])),
            "airspeed": lambda state: np.array([state.get("airspeed", 0.0)]),
            "angle_of_attack": lambda state: np.array([state.get("aoa", 0.0)])
        }
        
        self.control_mappings = {
            "wing_morphing": lambda outputs, state: {"wing_morph_command": outputs[0]},
            "control_surfaces": lambda outputs, state: {
                "elevator": outputs[0],
                "aileron": outputs[1],
                "rudder": outputs[2]
            }
        }
    
    def _init_hypersonic_mappings(self):
        """Initialize mappings for hypersonic drones."""
        self.sensor_mappings = {
            "temperature": lambda state: np.array([state.get("surface_temp", 0.0)]),
            "pressure": lambda state: np.array([state.get("dynamic_pressure", 0.0)]),
            "mach": lambda state: np.array([state.get("mach_number", 0.0)])
        }
        
        self.control_mappings = {
            "thermal_management": lambda outputs, state: {"cooling_rate": outputs[0]},
            "flight_surfaces": lambda outputs, state: {
                "elevon_left": outputs[0],
                "elevon_right": outputs[1]
            }
        }
    
    def get_sensor_data(self, airframe_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract sensor data from airframe state for neuromorphic processing."""
        sensor_data = {}
        for sensor_name, mapping_func in self.sensor_mappings.items():
            sensor_data[sensor_name] = mapping_func(airframe_state)
        return sensor_data
    
    def apply_control_commands(self, 
                              neuromorphic_outputs: Dict[str, np.ndarray], 
                              airframe_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply neuromorphic control outputs to airframe."""
        control_commands = {}
        for control_name, mapping_func in self.control_mappings.items():
            if control_name in neuromorphic_outputs:
                control_commands.update(
                    mapping_func(neuromorphic_outputs[control_name], airframe_state)
                )
        return control_commands

class NeuromorphicIntegrationFactory:
    """Factory for creating neuromorphic integration components."""
    
    @staticmethod
    def create_adapter(airframe_type: str) -> AirframeNeuromorphicAdapter:
        """Create a neuromorphic adapter for the specified airframe type."""
        return AirframeNeuromorphicAdapter(airframe_type)
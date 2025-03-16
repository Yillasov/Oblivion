from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class AerodynamicModel(ABC):
    """Base interface for all aerodynamic models."""
    
    @abstractmethod
    def calculate_forces(self, 
                        flight_conditions: Dict[str, float],
                        airframe_properties: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate aerodynamic forces based on flight conditions.
        
        Args:
            flight_conditions: Dictionary containing flight parameters
                (airspeed, angle of attack, sideslip, altitude, etc.)
            airframe_properties: Dictionary containing airframe parameters
            
        Returns:
            Dictionary of force vectors (lift, drag, side force)
        """
        pass
    
    @abstractmethod
    def calculate_moments(self,
                         flight_conditions: Dict[str, float],
                         airframe_properties: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate aerodynamic moments based on flight conditions.
        
        Returns:
            Dictionary of moment vectors (roll, pitch, yaw)
        """
        pass

class CFDModel(AerodynamicModel):
    """Computational Fluid Dynamics based aerodynamic model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mesh_resolution = config.get("mesh_resolution", "medium")
        self.solver_type = config.get("solver_type", "euler")
        
    def calculate_forces(self, flight_conditions, airframe_properties):
        # Simplified implementation
        return {
            "lift": np.array([0.0, 0.0, 1.0]),
            "drag": np.array([1.0, 0.0, 0.0]),
            "side": np.array([0.0, 1.0, 0.0])
        }
        
    def calculate_moments(self, flight_conditions, airframe_properties):
        # Simplified implementation
        return {
            "roll": np.array([1.0, 0.0, 0.0]),
            "pitch": np.array([0.0, 1.0, 0.0]),
            "yaw": np.array([0.0, 0.0, 1.0])
        }

class LookupTableModel(AerodynamicModel):
    """Lookup table based aerodynamic model."""
    
    def __init__(self, table_data: Dict[str, Any]):
        self.table_data = table_data
        
    def calculate_forces(self, flight_conditions, airframe_properties):
        # Implementation would interpolate from tables
        return {"lift": np.zeros(3), "drag": np.zeros(3), "side": np.zeros(3)}
        
    def calculate_moments(self, flight_conditions, airframe_properties):
        # Implementation would interpolate from tables
        return {"roll": np.zeros(3), "pitch": np.zeros(3), "yaw": np.zeros(3)}

class AerodynamicModelFactory:
    """Factory for creating aerodynamic model instances."""
    
    @staticmethod
    def create(model_type: str, config: Dict[str, Any]) -> AerodynamicModel:
        """Create an aerodynamic model of the specified type."""
        if model_type == "cfd":
            return CFDModel(config)
        elif model_type == "lookup_table":
            return LookupTableModel(config)
        else:
            raise ValueError(f"Unsupported aerodynamic model type: {model_type}")
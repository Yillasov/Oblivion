"""
Directed energy weapon systems for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, field

from src.payload.base import NeuromorphicPayload, PayloadSpecs
from src.payload.types import WeaponType, PayloadCategory


@dataclass
class DirectedEnergySpecs(PayloadSpecs):
    """Specifications for directed energy weapons."""
    energy_type: str  # laser, microwave, particle
    power_output: float  # Maximum power output in kW
    beam_divergence: float  # Beam divergence in mrad
    wavelength: Optional[float] = None  # Wavelength in nm (for lasers)
    cooling_capacity: float = 0.0  # Cooling capacity in kW


class DirectedEnergyWeapon(NeuromorphicPayload):
    """Base class for directed energy weapon systems."""
    
    def __init__(self, specs: DirectedEnergySpecs, hardware_interface=None):
        super().__init__(hardware_interface)
        self.specs = specs
        self.status = {
            "active": False,
            "power_level": 0.0,
            "temperature": 20.0,
            "target_locked": False
        }
    
    def get_specifications(self) -> PayloadSpecs:
        return self.specs
    
    def calculate_impact(self) -> Dict[str, float]:
        return {
            "weight_impact": self.specs.weight,
            "drag_coefficient": 0.01,
            "power_consumption": self.specs.power_requirements
        }
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        if not self.initialized:
            return False
        
        # Use neuromorphic processing for targeting
        targeting_result = self.process_data({
            "target": target_data,
            "computation": "targeting"
        })
        
        if targeting_result.get("lock_successful", False):
            self.status["active"] = True
            self.status["target_locked"] = True
            self.status["power_level"] = targeting_result.get("optimal_power", 50.0)
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        return self.status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.initialized:
            return {"error": "System not initialized"}
        
        computation_type = input_data.get("computation", "")
        
        if computation_type == "targeting":
            # Neuromorphic targeting calculation
            return {
                "lock_successful": True,
                "optimal_power": 75.0,
                "estimated_effectiveness": 0.85,
                "atmospheric_compensation": True
            }
        
        return {"error": "Unknown computation type"}
    
    def train(self, training_data: Dict[str, Any]) -> bool:
        return True if self.initialized else False


class HighEnergyLaser(DirectedEnergyWeapon):
    """High energy laser weapon system."""
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "HEL-50":
            specs = DirectedEnergySpecs(
                weight=350.0,
                volume={"length": 1.5, "width": 0.8, "height": 0.6},
                power_requirements=150000.0,  # 150 kW
                mounting_points=["fuselage", "internal_bay"],
                energy_type="laser",
                power_output=50.0,  # 50 kW
                beam_divergence=0.2,
                wavelength=1064.0,
                cooling_capacity=60.0
            )
        else:
            raise ValueError(f"Unknown laser model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        self.firing_duration = 0.0
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        success = super().deploy(target_data)
        if success:
            self.firing_duration = target_data.get("duration", 5.0)
            # Neuromorphic thermal management
            self._manage_thermal_load()
        return success
    
    def _manage_thermal_load(self):
        # Use neuromorphic processing for thermal management
        thermal_result = self.process_data({
            "power_level": self.status["power_level"],
            "duration": self.firing_duration,
            "computation": "thermal_management"
        })
        self.status["temperature"] = thermal_result.get("temperature", 20.0)
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        base_result = super().process_data(input_data)
        
        if input_data.get("computation") == "thermal_management":
            # Neuromorphic thermal management
            return {
                "temperature": 85.0,
                "cooling_efficiency": 0.92,
                "max_continuous_operation": 12.5  # seconds
            }
        
        return base_result
"""
Environmental interaction system for propulsion adaptation to external conditions.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.propulsion.base import PropulsionInterface


class EnvironmentType(Enum):
    """Types of environments that affect propulsion systems."""
    STANDARD = 0
    HIGH_ALTITUDE = 1
    MARINE = 2
    URBAN = 3
    EXTREME_HEAT = 4
    EXTREME_COLD = 5
    STORM = 6


@dataclass
class EnvironmentalCondition:
    """Environmental condition parameters."""
    temperature: float  # Temperature in Kelvin
    pressure: float  # Pressure in Pascal
    humidity: float  # Relative humidity (0-1)
    wind_speed: float  # Wind speed in m/s
    wind_direction: np.ndarray  # Wind direction vector
    precipitation: float  # Precipitation rate in mm/h
    air_density: float  # Air density in kg/m³
    environment_type: EnvironmentType  # Type of environment


class EnvironmentalInteractionSystem:
    """System for managing propulsion interactions with environment."""
    
    def __init__(self):
        """Initialize environmental interaction system."""
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.current_condition = EnvironmentalCondition(
            temperature=288.15,  # 15°C
            pressure=101325.0,  # 1 atm
            humidity=0.5,
            wind_speed=0.0,
            wind_direction=np.array([0.0, 0.0, 0.0]),
            precipitation=0.0,
            air_density=1.225,
            environment_type=EnvironmentType.STANDARD
        )
        self.adaptation_history: List[Dict[str, Any]] = []
        
    def register_system(self, system_id: str, system: PropulsionInterface) -> bool:
        """Register a propulsion system for environmental adaptation."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        return True
        
    def update_environment(self, condition: EnvironmentalCondition) -> None:
        """Update current environmental conditions."""
        self.current_condition = condition
        
        # Record adaptation event
        self.adaptation_history.append({
            "timestamp": np.datetime64('now'),
            "temperature": condition.temperature,
            "pressure": condition.pressure,
            "environment_type": condition.environment_type.name
        })
        
        # Limit history size
        if len(self.adaptation_history) > 100:
            self.adaptation_history.pop(0)
    
    def get_adaptation_parameters(self, system_id: str) -> Dict[str, float]:
        """Get adaptation parameters for a specific propulsion system."""
        if system_id not in self.propulsion_systems:
            return {}
            
        # Calculate adaptation parameters based on current conditions
        condition = self.current_condition
        
        # Base parameters
        params = {
            "power_adjustment": 1.0,
            "thermal_adjustment": 1.0,
            "efficiency_adjustment": 1.0,
            "intake_adjustment": 1.0,
            "exhaust_adjustment": 1.0
        }
        
        # Adjust for temperature
        if condition.temperature > 310.0:  # Hot (>37°C)
            params["power_adjustment"] *= 0.9
            params["thermal_adjustment"] *= 1.2
            params["efficiency_adjustment"] *= 0.95
        elif condition.temperature < 270.0:  # Cold (<-3°C)
            params["power_adjustment"] *= 1.1
            params["thermal_adjustment"] *= 0.8
            params["intake_adjustment"] *= 0.9
            
        # Adjust for pressure/altitude
        if condition.pressure < 70000.0:  # High altitude
            params["power_adjustment"] *= 0.85
            params["intake_adjustment"] *= 1.2
            params["efficiency_adjustment"] *= 0.9
            
        # Adjust for wind
        if condition.wind_speed > 15.0:  # Strong wind
            params["power_adjustment"] *= 1.15
            params["efficiency_adjustment"] *= 0.9
            
        # Adjust for precipitation
        if condition.precipitation > 5.0:  # Significant rain
            params["intake_adjustment"] *= 0.9
            params["exhaust_adjustment"] *= 0.95
            
        # Adjust for environment type
        if condition.environment_type == EnvironmentType.MARINE:
            params["thermal_adjustment"] *= 1.1
            params["intake_adjustment"] *= 0.9
        elif condition.environment_type == EnvironmentType.URBAN:
            params["exhaust_adjustment"] *= 0.9
            params["efficiency_adjustment"] *= 0.95
        elif condition.environment_type == EnvironmentType.EXTREME_HEAT:
            params["thermal_adjustment"] *= 1.3
            params["power_adjustment"] *= 0.8
        elif condition.environment_type == EnvironmentType.EXTREME_COLD:
            params["thermal_adjustment"] *= 0.7
            params["power_adjustment"] *= 1.2
        elif condition.environment_type == EnvironmentType.STORM:
            params["power_adjustment"] *= 1.3
            params["intake_adjustment"] *= 0.8
            params["efficiency_adjustment"] *= 0.85
            
        return params
    
    def apply_environmental_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Apply environmental adaptations to all registered propulsion systems."""
        results = {}
        
        for system_id, system in self.propulsion_systems.items():
            # Get adaptation parameters
            params = self.get_adaptation_parameters(system_id)
            
            # Get current system status
            status = system.get_status()
            
            # Apply adaptations (simplified)
            # In a real implementation, this would call specific methods on the propulsion system
            adaptation_result = {
                "system_id": system_id,
                "adaptation_params": params,
                "environment": self.current_condition.environment_type.name,
                "success": True
            }
            
            results[system_id] = adaptation_result
            
        return results
    
    def get_environmental_report(self) -> Dict[str, Any]:
        """Get report on current environmental conditions and adaptations."""
        return {
            "current_condition": {
                "temperature": self.current_condition.temperature,
                "pressure": self.current_condition.pressure,
                "humidity": self.current_condition.humidity,
                "wind_speed": self.current_condition.wind_speed,
                "precipitation": self.current_condition.precipitation,
                "air_density": self.current_condition.air_density,
                "environment_type": self.current_condition.environment_type.name
            },
            "registered_systems": list(self.propulsion_systems.keys()),
            "adaptation_history_size": len(self.adaptation_history),
            "last_adaptation": self.adaptation_history[-1] if self.adaptation_history else None
        }
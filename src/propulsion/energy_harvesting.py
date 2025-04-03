"""
Energy harvesting system for capturing and utilizing ambient energy.
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

from src.propulsion.environmental_interaction import EnvironmentalCondition, EnvironmentalInteractionSystem
from src.propulsion.base import PropulsionInterface


class HarvestingType(Enum):
    """Types of energy harvesting methods."""
    SOLAR = 0
    THERMAL = 1
    VIBRATION = 2
    AIRFLOW = 3
    ELECTROMAGNETIC = 4
    PRESSURE = 5


@dataclass
class HarvesterSpecs:
    """Specifications for an energy harvester."""
    harvesting_type: HarvestingType
    max_output: float  # Maximum power output in kW
    efficiency: float  # Conversion efficiency (0-1)
    area: float  # Collection area in m²
    weight: float  # Weight in kg
    response_time: float  # Response time in seconds
    min_threshold: float  # Minimum energy threshold for harvesting


class EnergyHarvestingSystem:
    """System for harvesting energy from the environment."""
    
    def __init__(self, env_system: Optional[EnvironmentalInteractionSystem] = None):
        """Initialize energy harvesting system."""
        self.env_system = env_system
        self.harvesters: Dict[str, HarvesterSpecs] = {}
        self.current_outputs: Dict[str, float] = {}
        self.total_harvested = 0.0  # Total energy harvested in kWh
        self.harvesting_history: List[Dict[str, Any]] = []
        
    def register_harvester(self, harvester_id: str, specs: HarvesterSpecs) -> bool:
        """Register an energy harvester."""
        if harvester_id in self.harvesters:
            return False
            
        self.harvesters[harvester_id] = specs
        self.current_outputs[harvester_id] = 0.0
        return True
        
    def update(self, 
             env_condition: EnvironmentalCondition, 
             dt: float) -> Dict[str, float]:
        """
        Update energy harvesting based on environmental conditions.
        
        Args:
            env_condition: Current environmental conditions
            dt: Time step in seconds
            
        Returns:
            Dictionary of harvester outputs in kW
        """
        outputs = {}
        total_output = 0.0
        
        for harvester_id, specs in self.harvesters.items():
            # Calculate harvester output based on type and conditions
            if specs.harvesting_type == HarvestingType.SOLAR:
                # Solar harvesting based on sun angle and cloud cover
                sun_angle = 90.0  # Default sun angle (degrees)
                cloud_cover = 0.0  # Default cloud cover (0-1)
                
                # Extract from environment if available
                if hasattr(env_condition, 'sun_angle'):
                    sun_angle = getattr(env_condition, 'sun_angle')
                if hasattr(env_condition, 'cloud_cover'):
                    cloud_cover = getattr(env_condition, 'cloud_cover')
                
                # Calculate solar factor
                angle_factor = np.sin(np.radians(sun_angle))
                angle_factor = max(0.1, angle_factor)
                cloud_factor = 1.0 - (cloud_cover * 0.8)
                
                # Calculate output
                output = specs.max_output * angle_factor * cloud_factor * specs.efficiency
                
            elif specs.harvesting_type == HarvestingType.THERMAL:
                # Thermal harvesting based on temperature differential
                temp_diff = abs(env_condition.temperature - 293.15)  # Difference from 20°C
                if temp_diff < specs.min_threshold:
                    output = 0.0
                else:
                    output = specs.max_output * (temp_diff / 50.0) * specs.efficiency
                    output = min(output, specs.max_output)
                    
            elif specs.harvesting_type == HarvestingType.VIBRATION:
                # Vibration harvesting based on vibration level
                vibration = 0.0  # Default vibration level (m/s²)
                
                # Extract from environment if available
                if hasattr(env_condition, 'vibration'):
                    vibration = getattr(env_condition, 'vibration')
                
                if vibration < specs.min_threshold:
                    output = 0.0
                else:
                    output = specs.max_output * (vibration / 10.0) * specs.efficiency
                    output = min(output, specs.max_output)
                    
            elif specs.harvesting_type == HarvestingType.AIRFLOW:
                # Airflow harvesting based on wind speed
                if env_condition.wind_speed < specs.min_threshold:
                    output = 0.0
                else:
                    # Power is proportional to cube of wind speed
                    wind_factor = (env_condition.wind_speed / 10.0)**3
                    output = specs.max_output * wind_factor * specs.efficiency
                    output = min(output, specs.max_output)
                    
            elif specs.harvesting_type == HarvestingType.ELECTROMAGNETIC:
                # Electromagnetic harvesting (simplified)
                em_field = 0.0  # Default EM field strength
                
                # Extract from environment if available
                if hasattr(env_condition, 'em_field'):
                    em_field = getattr(env_condition, 'em_field')
                
                if em_field < specs.min_threshold:
                    output = 0.0
                else:
                    output = specs.max_output * (em_field / 5.0) * specs.efficiency
                    output = min(output, specs.max_output)
                    
            elif specs.harvesting_type == HarvestingType.PRESSURE:
                # Pressure differential harvesting
                pressure_diff = abs(env_condition.pressure - 101325) / 1000  # Difference from 1 atm in kPa
                if pressure_diff < specs.min_threshold:
                    output = 0.0
                else:
                    output = specs.max_output * (pressure_diff / 50.0) * specs.efficiency
                    output = min(output, specs.max_output)
            else:
                output = 0.0
                
            # Store and accumulate output
            self.current_outputs[harvester_id] = output
            outputs[harvester_id] = output
            total_output += output
            
        # Update total harvested energy
        self.total_harvested += total_output * (dt / 3600.0)  # Convert to kWh
        
        # Record harvesting data
        self.harvesting_history.append({
            "timestamp": np.datetime64('now'),
            "outputs": self.current_outputs.copy(),
            "total_output": total_output,
            "environment_type": env_condition.environment_type.name
        })
        
        # Limit history size
        if len(self.harvesting_history) > 100:
            self.harvesting_history.pop(0)
            
        return outputs
    
    def get_total_output(self) -> float:
        """Get current total output from all harvesters in kW."""
        return sum(self.current_outputs.values())
    
    def get_harvesting_report(self) -> Dict[str, Any]:
        """Get report on energy harvesting performance."""
        return {
            "current_outputs": self.current_outputs.copy(),
            "total_output": self.get_total_output(),
            "total_harvested": self.total_harvested,
            "harvester_count": len(self.harvesters),
            "harvester_types": {hid: h.harvesting_type.name for hid, h in self.harvesters.items()},
            "history_size": len(self.harvesting_history),
            "last_harvest": self.harvesting_history[-1] if self.harvesting_history else None
        }
    
    def integrate_with_propulsion(self, 
                                propulsion_system: PropulsionInterface,
                                flight_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Integrate harvested energy with propulsion system.
        
        Args:
            propulsion_system: Propulsion system to integrate with
            flight_conditions: Current flight conditions
            
        Returns:
            Integration results
        """
        # Get current total output
        total_output = self.get_total_output()
        
        # If no energy is being harvested, return early
        if total_output <= 0.0:
            return {
                "integrated": False,
                "reason": "no_energy_available",
                "harvested_power": 0.0
            }
            
        # Get propulsion system status
        status = propulsion_system.get_status()
        
        # Simple integration - just report the harvested energy
        # In a real implementation, this would modify propulsion parameters
        return {
            "integrated": True,
            "harvested_power": total_output,
            "propulsion_status": status
        }
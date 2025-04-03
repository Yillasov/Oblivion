#!/usr/bin/env python3
"""
Thermal management system for metamaterial-based infrared suppression.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.stealth.metamaterial.manufacturing_constraints import MetamaterialManufacturingProcess
from src.stealth.infrared.infrared_suppression import IRSuppressionParameters


class CoolingMethod(Enum):
    """Cooling methods for metamaterial thermal management."""
    PASSIVE = "passive"
    ACTIVE_FLUID = "active_fluid"
    THERMOELECTRIC = "thermoelectric"
    PHASE_CHANGE = "phase_change"
    RADIATIVE = "radiative"


@dataclass
class MetamaterialThermalProfile:
    """Thermal profile for metamaterial components."""
    max_operating_temp: float  # Maximum operating temperature in °C
    optimal_temp_range: Tuple[float, float]  # Optimal temperature range in °C
    thermal_conductivity: float  # W/(m·K)
    specific_heat_capacity: float  # J/(kg·K)
    emissivity: float  # 0.0-1.0
    current_temperature: float  # Current temperature in °C
    thermal_mass: float  # kg
    cooling_efficiency: float  # 0.0-1.0


class MetamaterialThermalManager:
    """
    Thermal management system for metamaterial-based infrared suppression.
    Integrates with existing IR suppression systems to optimize thermal performance.
    """
    
    def __init__(self, ir_params: Optional[IRSuppressionParameters] = None):
        """
        Initialize metamaterial thermal manager.
        
        Args:
            ir_params: Optional IR suppression parameters
        """
        self.ir_params = ir_params
        self.components: Dict[str, MetamaterialThermalProfile] = {}
        self.cooling_systems: Dict[str, Dict[str, Any]] = {}
        self.ambient_temperature = 20.0  # °C
        self.power_consumption = 0.0  # kW
        self.active = False
    
    def register_component(self, component_id: str, profile: MetamaterialThermalProfile) -> None:
        """Register a metamaterial component for thermal management."""
        self.components[component_id] = profile
    
    def register_cooling_system(self, 
                              system_id: str, 
                              method: CoolingMethod,
                              cooling_capacity: float,
                              power_requirement: float) -> None:
        """
        Register a cooling system.
        
        Args:
            system_id: Cooling system identifier
            method: Cooling method
            cooling_capacity: Cooling capacity in W
            power_requirement: Power requirement in kW
        """
        self.cooling_systems[system_id] = {
            "method": method,
            "cooling_capacity": cooling_capacity,
            "power_requirement": power_requirement,
            "active": False,
            "current_level": 0.0
        }
    
    def activate(self, power_level: float = 0.8) -> bool:
        """
        Activate thermal management system.
        
        Args:
            power_level: Power level (0.0-1.0)
            
        Returns:
            Success status
        """
        if not self.cooling_systems:
            return False
            
        self.active = True
        self.power_consumption = 0.0
        
        # Activate cooling systems
        for system_id, system in self.cooling_systems.items():
            system["active"] = True
            system["current_level"] = power_level
            self.power_consumption += system["power_requirement"] * power_level
            
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate thermal management system.
        
        Returns:
            Success status
        """
        self.active = False
        self.power_consumption = 0.0
        
        # Deactivate cooling systems
        for system_id, system in self.cooling_systems.items():
            system["active"] = False
            system["current_level"] = 0.0
            
        return True
    
    def update_thermal_state(self, 
                           time_delta: float, 
                           power_levels: Dict[str, float]) -> Dict[str, Any]:
        """
        Update thermal state of metamaterial components.
        
        Args:
            time_delta: Time delta in seconds
            power_levels: Power levels for each component
            
        Returns:
            Updated thermal state
        """
        if not self.active:
            return {"status": "inactive"}
            
        results = {}
        
        # Calculate total cooling capacity
        total_cooling = 0.0
        for system_id, system in self.cooling_systems.items():
            if system["active"]:
                total_cooling += system["cooling_capacity"] * system["current_level"]
        
        # Distribute cooling capacity proportionally to component needs
        for component_id, profile in self.components.items():
            # Get power level for this component
            power_level = power_levels.get(component_id, 0.0)
            
            # Calculate heat generation (simplified model)
            heat_generation = power_level * 100.0  # W
            
            # Calculate temperature change without cooling
            temp_change_rate = heat_generation / (profile.thermal_mass * profile.specific_heat_capacity)
            natural_temp_change = temp_change_rate * time_delta
            
            # Calculate passive cooling
            temp_diff = profile.current_temperature - self.ambient_temperature
            passive_cooling = temp_diff * profile.emissivity * 0.1 * time_delta
            
            # Calculate active cooling (proportional distribution)
            temp_over_optimal = max(0, profile.current_temperature - profile.optimal_temp_range[1])
            cooling_priority = temp_over_optimal / max(1.0, profile.max_operating_temp - profile.optimal_temp_range[1])
            
            # Apply cooling proportionally to priority
            active_cooling = 0.0
            if total_cooling > 0 and cooling_priority > 0:
                active_cooling = (total_cooling * cooling_priority * profile.cooling_efficiency * time_delta) / 100.0
            
            # Calculate net temperature change
            net_temp_change = natural_temp_change - passive_cooling - active_cooling
            
            # Update temperature
            profile.current_temperature += net_temp_change
            
            # Record results
            results[component_id] = {
                "temperature": profile.current_temperature,
                "temp_change": net_temp_change,
                "cooling_applied": active_cooling + passive_cooling,
                "heat_generated": heat_generation,
                "status": "normal" if profile.current_temperature <= profile.max_operating_temp else "overheating"
            }
        
        return {
            "status": "active",
            "power_consumption": self.power_consumption,
            "components": results,
            "total_cooling_capacity": total_cooling
        }
    
    def get_ir_suppression_factor(self) -> float:
        """
        Calculate IR suppression factor based on thermal management.
        
        Returns:
            IR suppression factor (0.0-1.0)
        """
        if not self.active or not self.components:
            return 0.0
            
        # Calculate average temperature reduction from optimal
        total_reduction = 0.0
        count = 0
        
        for component_id, profile in self.components.items():
            optimal_temp = (profile.optimal_temp_range[0] + profile.optimal_temp_range[1]) / 2
            temp_diff = max(0, optimal_temp - profile.current_temperature)
            total_reduction += temp_diff
            count += 1
        
        avg_reduction = total_reduction / max(1, count)
        
        # Calculate suppression factor (simplified model)
        # Higher temperature reduction = better IR suppression
        suppression_factor = min(avg_reduction / 50.0, 0.9)  # Max 90% suppression
        
        return suppression_factor
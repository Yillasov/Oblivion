"""
Thermal Management System for Electric Propulsion Components.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.propulsion.hybrid_electric import HybridElectricController
from src.propulsion.solar_electric import SolarElectricSystem
from src.propulsion.hydrogen_fuel_cell import HydrogenFuelCellManager
from src.propulsion.power_distribution import PowerDistributionOptimizer


@dataclass
class ThermalProfile:
    """Thermal characteristics of a propulsion component."""
    name: str
    max_temperature: float  # Maximum safe temperature in K
    optimal_temperature: float  # Optimal operating temperature in K
    cooling_rate: float  # Passive cooling rate in K/s
    heating_rate: float  # Heating rate at full power in K/s
    thermal_mass: float  # Thermal mass in J/K
    current_temperature: float = 293.0  # Current temperature in K


class ThermalManagementSystem:
    """Simple thermal management system for electric propulsion."""
    
    def __init__(self, power_optimizer: Optional[PowerDistributionOptimizer] = None):
        """Initialize thermal management system."""
        self.power_optimizer = power_optimizer
        self.thermal_profiles: Dict[str, ThermalProfile] = {}
        self.cooling_systems: Dict[str, Dict[str, float]] = {}
        self.thermal_history: Dict[str, List[Dict[str, float]]] = {}
        self.ambient_temperature = 293.0  # Default ambient temperature in K
        
    def register_component(self, component_id: str, profile: ThermalProfile) -> None:
        """Register a component for thermal management."""
        self.thermal_profiles[component_id] = profile
        self.thermal_history[component_id] = []
        
    def register_cooling_system(self, 
                              cooling_id: str, 
                              max_cooling_rate: float,
                              power_consumption: float) -> None:
        """Register a cooling system."""
        self.cooling_systems[cooling_id] = {
            "max_cooling_rate": max_cooling_rate,  # K/s
            "power_consumption": power_consumption,  # kW
            "current_level": 0.0  # 0-1 scale
        }
        
    def update_ambient_temperature(self, temperature: float) -> None:
        """Update ambient temperature based on flight conditions."""
        self.ambient_temperature = temperature
        
    def calculate_temperature_change(self, 
                                   component_id: str, 
                                   power_level: float,
                                   dt: float) -> float:
        """
        Calculate temperature change for a component.
        
        Args:
            component_id: Component identifier
            power_level: Current power level (0-1)
            dt: Time step in seconds
            
        Returns:
            Temperature change in K
        """
        if component_id not in self.thermal_profiles:
            return 0.0
            
        profile = self.thermal_profiles[component_id]
        current_temp = profile.current_temperature
        
        # Calculate heating from power usage
        heating = profile.heating_rate * power_level * dt
        
        # Calculate passive cooling (proportional to temperature difference)
        temp_diff = current_temp - self.ambient_temperature
        passive_cooling = profile.cooling_rate * temp_diff * dt
        
        # Calculate active cooling from cooling systems
        active_cooling = 0.0
        for cooling_id, cooling_system in self.cooling_systems.items():
            cooling_level = cooling_system["current_level"]
            max_rate = cooling_system["max_cooling_rate"]
            active_cooling += cooling_level * max_rate * dt
        
        # Net temperature change
        net_change = heating - passive_cooling - active_cooling
        
        return net_change
        
    def update_temperatures(self, 
                          power_levels: Dict[str, float], 
                          dt: float) -> Dict[str, Dict[str, float]]:
        """
        Update temperatures for all components.
        
        Args:
            power_levels: Current power levels for each component
            dt: Time step in seconds
            
        Returns:
            Updated thermal status
        """
        thermal_status = {}
        
        for component_id, profile in self.thermal_profiles.items():
            power_level = power_levels.get(component_id, 0.0)
            
            # Calculate temperature change
            temp_change = self.calculate_temperature_change(component_id, power_level, dt)
            
            # Update temperature
            new_temp = profile.current_temperature + temp_change
            profile.current_temperature = new_temp
            
            # Calculate thermal margin
            thermal_margin = (profile.max_temperature - new_temp) / (
                profile.max_temperature - profile.optimal_temperature
            )
            
            # Record thermal data
            thermal_data = {
                "temperature": new_temp,
                "power_level": power_level,
                "thermal_margin": thermal_margin,
                "timestamp": dt  # Assuming dt is cumulative time
            }
            self.thermal_history[component_id].append(thermal_data)
            
            # Prepare status report
            thermal_status[component_id] = {
                "temperature": new_temp,
                "optimal_temperature": profile.optimal_temperature,
                "max_temperature": profile.max_temperature,
                "thermal_margin": thermal_margin,
                "critical": thermal_margin < 0.2  # Critical if less than 20% margin
            }
            
        return thermal_status
        
    def adjust_cooling_systems(self, thermal_status: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Adjust cooling systems based on thermal status.
        
        Args:
            thermal_status: Current thermal status
            
        Returns:
            Cooling system levels
        """
        # Find the component with the lowest thermal margin
        critical_components = []
        for component_id, status in thermal_status.items():
            if status["critical"]:
                critical_components.append((component_id, status["thermal_margin"]))
        
        # Sort by thermal margin (lowest first)
        critical_components.sort(key=lambda x: x[1])
        
        # Adjust cooling based on critical components
        cooling_levels = {}
        
        if critical_components:
            # Activate cooling systems at levels proportional to criticality
            for cooling_id, cooling_system in self.cooling_systems.items():
                # Simple proportional control - more cooling for lower margin
                margin = critical_components[0][1]  # Margin of most critical component
                cooling_level = max(0.0, min(1.0, 1.0 - margin))
                
                # Update cooling system
                self.cooling_systems[cooling_id]["current_level"] = cooling_level
                cooling_levels[cooling_id] = cooling_level
        else:
            # No critical components, reduce cooling to save power
            for cooling_id, cooling_system in self.cooling_systems.items():
                # Gradually reduce cooling
                current_level = cooling_system["current_level"]
                new_level = max(0.0, current_level - 0.1)
                
                # Update cooling system
                self.cooling_systems[cooling_id]["current_level"] = new_level
                cooling_levels[cooling_id] = new_level
                
        return cooling_levels
        
    def get_cooling_power_consumption(self) -> float:
        """Calculate total power consumption of cooling systems."""
        total_power = 0.0
        for cooling_id, cooling_system in self.cooling_systems.items():
            power = cooling_system["power_consumption"] * cooling_system["current_level"]
            total_power += power
        return total_power
        
    def manage_thermal_conditions(self, 
                                flight_conditions: Dict[str, float],
                                power_levels: Dict[str, float],
                                dt: float) -> Dict[str, Any]:
        """
        Main thermal management function.
        
        Args:
            flight_conditions: Current flight conditions
            power_levels: Current power levels for each component
            dt: Time step in seconds
            
        Returns:
            Thermal management results
        """
        # Update ambient temperature based on flight conditions
        altitude = flight_conditions.get("altitude", 0)
        # Simple atmospheric model: temperature decreases with altitude
        if altitude < 11000:  # Troposphere
            self.ambient_temperature = 288.15 - 0.0065 * altitude
        else:  # Stratosphere (simplified)
            self.ambient_temperature = 216.65
            
        # Update component temperatures
        thermal_status = self.update_temperatures(power_levels, dt)
        
        # Adjust cooling systems
        cooling_levels = self.adjust_cooling_systems(thermal_status)
        
        # Calculate cooling power consumption
        cooling_power = self.get_cooling_power_consumption()
        
        # If we have a power optimizer, inform it about cooling power needs
        if self.power_optimizer:
            # This would be implemented in a real system to request power for cooling
            pass
            
        return {
            "thermal_status": thermal_status,
            "cooling_levels": cooling_levels,
            "cooling_power": cooling_power,
            "ambient_temperature": self.ambient_temperature
        }
        
    def get_component_temperature(self, component_id: str) -> float:
        """Get current temperature of a component."""
        if component_id in self.thermal_profiles:
            return self.thermal_profiles[component_id].current_temperature
        return 0.0
        
    def is_system_in_safe_range(self) -> bool:
        """Check if all components are within safe temperature range."""
        for component_id, profile in self.thermal_profiles.items():
            if profile.current_temperature > profile.max_temperature:
                return False
        return True
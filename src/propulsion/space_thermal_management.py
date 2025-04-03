"""
Space-Grade Thermal Management System for vacuum environments.
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

from src.propulsion.thermal_management import ThermalManagementSystem, ThermalProfile
from src.propulsion.radiation_shielding import RadiationShieldingCalculator, RadiationType


class SpaceEnvironment(Enum):
    """Space environment types affecting thermal management."""
    LOW_EARTH_ORBIT = 0
    GEOSTATIONARY = 1
    LUNAR = 2
    INTERPLANETARY = 3
    DEEP_SPACE = 4


@dataclass
class SpaceThermalProfile(ThermalProfile):
    """Extended thermal profile for space environments."""
    radiation_absorption: float = 0.8  # Radiation absorption coefficient (0-1)
    emissivity: float = 0.9  # Surface emissivity (0-1)
    surface_area: float = 1.0  # Surface area in m²
    view_factor: float = 0.5  # View factor to space (0-1)


class SpaceThermalManager:
    """Thermal management system for space environments."""
    
    def __init__(self, base_thermal_system: ThermalManagementSystem):
        """Initialize space thermal management system."""
        self.base_system = base_thermal_system
        self.environment = SpaceEnvironment.LOW_EARTH_ORBIT
        self.space_profiles: Dict[str, SpaceThermalProfile] = {}
        self.radiation_calc = RadiationShieldingCalculator()
        self.solar_flux = 1361.0  # W/m² at Earth distance
        self.albedo_factor = 0.3  # Earth's albedo
        self.earth_IR = 237.0  # W/m² Earth IR radiation
        
    def register_component(self, component_id: str, profile: SpaceThermalProfile) -> None:
        """Register a component for space thermal management."""
        # Register with base system
        self.base_system.register_component(component_id, profile)
        # Store space-specific profile
        self.space_profiles[component_id] = profile
        
    def set_environment(self, environment: SpaceEnvironment) -> None:
        """Set space environment for thermal calculations."""
        self.environment = environment
        
        # Update environmental parameters based on environment
        if environment == SpaceEnvironment.LOW_EARTH_ORBIT:
            self.solar_flux = 1361.0
            self.albedo_factor = 0.3
            self.earth_IR = 237.0
        elif environment == SpaceEnvironment.GEOSTATIONARY:
            self.solar_flux = 1361.0
            self.albedo_factor = 0.15
            self.earth_IR = 5.0
        elif environment == SpaceEnvironment.LUNAR:
            self.solar_flux = 1361.0
            self.albedo_factor = 0.12
            self.earth_IR = 0.0
        elif environment == SpaceEnvironment.INTERPLANETARY:
            # Adjust solar flux based on distance (default to Mars distance)
            self.solar_flux = 589.0
            self.albedo_factor = 0.0
            self.earth_IR = 0.0
        elif environment == SpaceEnvironment.DEEP_SPACE:
            self.solar_flux = 0.0
            self.albedo_factor = 0.0
            self.earth_IR = 0.0
    
    def calculate_radiative_heat_transfer(self, component_id: str) -> float:
        """
        Calculate radiative heat transfer in space.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Net radiative heat transfer in W
        """
        if component_id not in self.space_profiles:
            return 0.0
            
        profile = self.space_profiles[component_id]
        
        # Stefan-Boltzmann constant
        sigma = 5.67e-8  # W/(m²·K⁴)
        
        # Calculate radiative heat emission (cooling)
        T = profile.current_temperature
        emission = sigma * profile.emissivity * profile.surface_area * T**4 * profile.view_factor
        
        # Calculate heat absorption from solar radiation
        solar_absorption = self.solar_flux * profile.radiation_absorption * profile.surface_area * (1 - profile.view_factor)
        
        # Add Earth albedo and IR for near-Earth environments
        earth_radiation = 0.0
        if self.environment in [SpaceEnvironment.LOW_EARTH_ORBIT, SpaceEnvironment.GEOSTATIONARY]:
            earth_radiation = (self.solar_flux * self.albedo_factor + self.earth_IR) * profile.radiation_absorption * profile.surface_area * (1 - profile.view_factor)
        
        # Net radiative heat transfer (positive = heating, negative = cooling)
        net_radiation = solar_absorption + earth_radiation - emission
        
        return net_radiation
    
    def manage_space_thermal_conditions(self, 
                                      flight_conditions: Dict[str, float],
                                      power_levels: Dict[str, float],
                                      dt: float) -> Dict[str, Any]:
        """
        Manage thermal conditions in space environment.
        
        Args:
            flight_conditions: Current flight conditions
            power_levels: Current power levels for each component
            dt: Time step in seconds
            
        Returns:
            Thermal management results
        """
        # First, let the base system handle standard thermal management
        base_results = self.base_system.manage_thermal_conditions(
            flight_conditions, power_levels, dt
        )
        
        # Then apply space-specific thermal adjustments
        space_adjustments = {}
        
        for component_id, profile in self.space_profiles.items():
            # Calculate radiative heat transfer
            rad_heat = self.calculate_radiative_heat_transfer(component_id)
            
            # Convert to temperature change
            if profile.thermal_mass > 0:
                temp_change = rad_heat * dt / profile.thermal_mass
            else:
                temp_change = 0.0
                
            # Apply temperature change
            profile.current_temperature += temp_change
            
            # Update the base system's profile as well
            if component_id in self.base_system.thermal_profiles:
                self.base_system.thermal_profiles[component_id].current_temperature = profile.current_temperature
            
            # Store adjustment data
            space_adjustments[component_id] = {
                "radiative_heat": rad_heat,
                "temperature_change": temp_change,
                "final_temperature": profile.current_temperature
            }
        
        # Combine results
        results = {
            **base_results,
            "space_adjustments": space_adjustments,
            "environment": self.environment.name,
            "solar_flux": self.solar_flux
        }
        
        return results
    
    def recommend_thermal_controls(self) -> Dict[str, Any]:
        """Recommend thermal control measures for space environment."""
        recommendations = {}
        
        for component_id, profile in self.space_profiles.items():
            current_temp = profile.current_temperature
            optimal_temp = profile.optimal_temperature
            max_temp = profile.max_temperature
            
            # Calculate temperature deviation from optimal
            temp_deviation = abs(current_temp - optimal_temp)
            
            if current_temp > max_temp:
                status = "CRITICAL"
                actions = ["Deploy emergency radiators", "Reduce power immediately"]
            elif current_temp > optimal_temp * 1.1:
                status = "WARNING"
                actions = ["Increase radiator exposure", "Adjust component orientation"]
            elif current_temp < optimal_temp * 0.9:
                status = "WARNING"
                actions = ["Reduce radiator exposure", "Increase power to heaters"]
            else:
                status = "NOMINAL"
                actions = ["Maintain current configuration"]
                
            # Add radiation considerations
            if self.environment in [SpaceEnvironment.INTERPLANETARY, SpaceEnvironment.DEEP_SPACE]:
                actions.append("Monitor radiation shielding integrity")
                
            recommendations[component_id] = {
                "status": status,
                "temperature_deviation": temp_deviation,
                "actions": actions
            }
            
        return recommendations
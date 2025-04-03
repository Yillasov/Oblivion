"""
Solar-Electric Propulsion Integration for UCAV platforms.
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

from src.propulsion.hybrid_electric import HybridElectricController
from src.propulsion.optimization import OptimizationConstraints


@dataclass
class SolarPanelSpecs:
    """Specifications for solar panel system."""
    area: float  # Solar panel area in m²
    efficiency: float  # Panel efficiency (0-1)
    weight: float  # Weight in kg
    max_output: float  # Maximum power output in kW
    degradation_rate: float  # Annual degradation rate


class SolarElectricSystem:
    """Solar-electric propulsion integration system."""
    
    def __init__(self, 
                 hybrid_controller: HybridElectricController,
                 solar_specs: SolarPanelSpecs):
        """Initialize solar-electric system."""
        self.hybrid_controller = hybrid_controller
        self.solar_specs = solar_specs
        self.solar_output = 0.0  # Current solar power output in kW
        self.charging_battery = False
        self.solar_history: List[Dict[str, float]] = []
        
    def calculate_solar_output(self, flight_conditions: Dict[str, float]) -> float:
        """Calculate solar power output based on flight conditions."""
        # Extract relevant conditions
        altitude = flight_conditions.get("altitude", 0)
        cloud_cover = flight_conditions.get("cloud_cover", 0)
        sun_angle = flight_conditions.get("sun_angle", 90)
        
        # Base output calculation
        # Higher altitude = better solar irradiance
        altitude_factor = min(1.2, 1.0 + (altitude / 10000) * 0.2)
        
        # Cloud cover reduces output (0-1)
        cloud_factor = 1.0 - (cloud_cover * 0.8)
        
        # Sun angle affects output (optimal at 90 degrees)
        angle_factor = np.sin(np.radians(sun_angle))
        angle_factor = max(0.1, angle_factor)
        
        # Calculate output
        base_output = self.solar_specs.area * self.solar_specs.efficiency
        current_output = base_output * altitude_factor * cloud_factor * angle_factor
        
        # Limit to max output
        self.solar_output = min(current_output, self.solar_specs.max_output)
        
        # Record solar data
        self.solar_history.append({
            "altitude": altitude,
            "cloud_cover": cloud_cover,
            "sun_angle": sun_angle,
            "output": self.solar_output
        })
        
        return self.solar_output
    
    def integrate_with_hybrid_system(self, 
                                   flight_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Integrate solar power with hybrid electric system."""
        # Calculate solar output
        solar_output = self.calculate_solar_output(flight_conditions)
        
        # Get current battery level
        battery_level = self.hybrid_controller.battery_level
        
        # Determine if we should charge the battery or power the system directly
        self.charging_battery = battery_level < 0.8 and self.hybrid_controller.current_mode != "power_boost"
        
        if self.charging_battery:
            # Use solar power to charge battery
            # Simple charging model: 80% efficiency
            charging_rate = solar_output * 0.8
            new_battery_level = min(1.0, battery_level + (charging_rate * 0.01))  # Assuming 0.01 time units
            
            # Update battery level
            self.hybrid_controller.update_energy_levels(new_battery_level, self.hybrid_controller.fuel_level)
            
            return {
                "solar_output": solar_output,
                "mode": "battery_charging",
                "charging_rate": charging_rate,
                "battery_level": new_battery_level
            }
        else:
            # Use solar power to supplement electric power
            # Get current power distribution
            requested_power = flight_conditions.get("power_request", 0.5)
            power_distribution = self.hybrid_controller.calculate_power_distribution(requested_power)
            
            # Reduce electric power draw by solar output
            electric_power = power_distribution["electric_power"]
            solar_contribution = min(electric_power, solar_output)
            adjusted_electric = max(0, electric_power - solar_contribution)
            
            # Update power distribution
            power_distribution["electric_power"] = adjusted_electric
            power_distribution["solar_power"] = solar_contribution
            power_distribution["total_power"] = adjusted_electric + power_distribution["combustion_power"] + solar_contribution
            
            return {
                "solar_output": solar_output,
                "mode": "direct_power",
                "power_distribution": power_distribution,
                "solar_contribution": solar_contribution
            }
    
    def optimize_solar_usage(self, 
                           flight_conditions: Dict[str, float],
                           mission_duration: float) -> Dict[str, Any]:
        """Optimize solar power usage based on mission parameters."""
        # Calculate expected solar output over mission
        current_output = self.calculate_solar_output(flight_conditions)
        
        # Simple prediction model
        time_of_day = flight_conditions.get("time_of_day", 12)  # 24-hour format
        remaining_daylight = max(0, 18 - time_of_day)  # Assuming daylight until 6 PM
        
        # Estimate total available solar energy for the mission
        available_solar_energy = current_output * min(remaining_daylight, mission_duration)
        
        # Determine optimal usage strategy
        if available_solar_energy > mission_duration * 0.3:
            # Plenty of solar energy available - use it directly
            strategy = "maximize_direct_usage"
            recommended_mode = "full_electric" if self.hybrid_controller.battery_level > 0.5 else "balanced"
        elif self.hybrid_controller.battery_level < 0.4:
            # Low battery - prioritize charging
            strategy = "prioritize_charging"
            recommended_mode = "fuel_saving"
        else:
            # Balanced approach
            strategy = "balanced_usage"
            recommended_mode = "balanced"
        
        # Set recommended mode
        self.hybrid_controller.set_mode(recommended_mode)
        
        return {
            "strategy": strategy,
            "recommended_mode": recommended_mode,
            "available_solar_energy": available_solar_energy,
            "mission_duration": mission_duration,
            "current_output": current_output
        }
"""
Hydrogen Fuel Cell System Manager for UCAV platforms.
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


@dataclass
class FuelCellSpecs:
    """Specifications for hydrogen fuel cell system."""
    power_output: float  # Maximum power output in kW
    efficiency: float  # Conversion efficiency (0-1)
    weight: float  # Weight in kg
    hydrogen_capacity: float  # Hydrogen storage capacity in kg
    startup_time: float  # Time to reach operational state in seconds
    operating_temperature: Tuple[float, float]  # Min/max operating temperature in K


class HydrogenFuelCellManager:
    """Simple hydrogen fuel cell system manager."""
    
    def __init__(self, 
                 hybrid_controller: HybridElectricController,
                 fuel_cell_specs: FuelCellSpecs):
        """Initialize hydrogen fuel cell manager."""
        self.hybrid_controller = hybrid_controller
        self.specs = fuel_cell_specs
        self.hydrogen_level = 1.0  # Normalized hydrogen level (0-1)
        self.current_power = 0.0  # Current power output in kW
        self.temperature = 293.0  # Current temperature in K
        self.operational = False  # Whether the fuel cell is operational
        self.startup_progress = 0.0  # Startup progress (0-1)
        self.performance_history: List[Dict[str, float]] = []
        
    def update_hydrogen_level(self, new_level: float) -> None:
        """Update hydrogen level."""
        self.hydrogen_level = max(0.0, min(1.0, new_level))
        
        # If hydrogen is critically low, notify the hybrid controller
        if self.hydrogen_level < 0.1:
            # Update hybrid controller to rely less on fuel cell
            self.hybrid_controller.update_energy_levels(
                self.hybrid_controller.battery_level,
                0.1  # Low fuel level to trigger mode change
            )
    
    def start_system(self) -> Dict[str, Any]:
        """Start the fuel cell system."""
        if self.operational:
            return {"status": "already_running", "power": self.current_power}
            
        if self.hydrogen_level <= 0.05:
            return {"status": "insufficient_hydrogen", "power": 0.0}
            
        # Begin startup sequence
        self.startup_progress = 0.01
        self.operational = False
        
        return {
            "status": "starting",
            "estimated_time": self.specs.startup_time,
            "power": 0.0
        }
    
    def update(self, dt: float, power_request: float) -> Dict[str, Any]:
        """
        Update fuel cell system state.
        
        Args:
            dt: Time step in seconds
            power_request: Requested power in kW
            
        Returns:
            Dict with system status
        """
        # Handle startup sequence
        if self.startup_progress > 0.0 and self.startup_progress < 1.0:
            self.startup_progress += dt / self.specs.startup_time
            
            if self.startup_progress >= 1.0:
                self.operational = True
                self.startup_progress = 1.0
        
        # If not operational, return zero power
        if not self.operational:
            return {
                "status": "starting" if self.startup_progress > 0.0 else "offline",
                "power": 0.0,
                "hydrogen_level": self.hydrogen_level,
                "temperature": self.temperature,
                "startup_progress": self.startup_progress
            }
        
        # Calculate available power based on hydrogen level
        available_power = self.specs.power_output * self.hydrogen_level
        
        # Limit requested power to available power
        actual_power = min(power_request, available_power)
        
        # Calculate hydrogen consumption
        # Simple model: consumption proportional to power output
        # with efficiency factor
        hydrogen_consumption = (actual_power / self.specs.efficiency) * dt / 3600.0
        
        # Update hydrogen level
        new_hydrogen_level = self.hydrogen_level - (hydrogen_consumption / self.specs.hydrogen_capacity)
        self.update_hydrogen_level(new_hydrogen_level)
        
        # Update temperature (simple model)
        target_temp = self.specs.operating_temperature[0] + (
            (self.specs.operating_temperature[1] - self.specs.operating_temperature[0]) * 
            (actual_power / self.specs.power_output)
        )
        self.temperature += (target_temp - self.temperature) * 0.1 * dt
        
        # Update current power
        self.current_power = actual_power
        
        # Record performance data
        self.performance_history.append({
            "power": actual_power,
            "hydrogen_level": self.hydrogen_level,
            "temperature": self.temperature,
            "efficiency": self.specs.efficiency * (0.9 + 0.1 * self.hydrogen_level)
        })
        
        return {
            "status": "operational",
            "power": actual_power,
            "hydrogen_level": self.hydrogen_level,
            "temperature": self.temperature,
            "efficiency": self.specs.efficiency * (0.9 + 0.1 * self.hydrogen_level)
        }
    
    def shutdown(self) -> Dict[str, Any]:
        """Shutdown the fuel cell system."""
        if not self.operational and self.startup_progress == 0.0:
            return {"status": "already_shutdown"}
            
        self.operational = False
        self.startup_progress = 0.0
        self.current_power = 0.0
        
        return {"status": "shutdown_complete"}
    
    def integrate_with_hybrid_system(self, 
                                   flight_conditions: Dict[str, float],
                                   dt: float) -> Dict[str, Any]:
        """Integrate fuel cell with hybrid electric system."""
        # Get power request from flight conditions
        power_request = flight_conditions.get("power_request", 0.5)
        
        # Update fuel cell system
        fuel_cell_status = self.update(dt, power_request * self.specs.power_output)
        
        # If fuel cell is providing power, update hybrid controller
        if fuel_cell_status["status"] == "operational":
            # Get current power distribution
            power_distribution = self.hybrid_controller.calculate_power_distribution(power_request)
            
            # Reduce combustion power by fuel cell contribution
            combustion_power = power_distribution["combustion_power"]
            fuel_cell_contribution = min(combustion_power, fuel_cell_status["power"])
            adjusted_combustion = max(0, combustion_power - fuel_cell_contribution)
            
            # Update power distribution
            power_distribution["combustion_power"] = adjusted_combustion
            power_distribution["fuel_cell_power"] = fuel_cell_contribution
            power_distribution["total_power"] = (
                power_distribution["electric_power"] + 
                adjusted_combustion + 
                fuel_cell_contribution
            )
            
            # If we have solar power in the distribution, include it
            if "solar_power" in power_distribution:
                power_distribution["total_power"] += power_distribution["solar_power"]
            
            return {
                "fuel_cell_status": fuel_cell_status,
                "power_distribution": power_distribution,
                "fuel_cell_contribution": fuel_cell_contribution
            }
        else:
            # Fuel cell not contributing power
            return {
                "fuel_cell_status": fuel_cell_status,
                "power_distribution": self.hybrid_controller.calculate_power_distribution(power_request),
                "fuel_cell_contribution": 0.0
            }
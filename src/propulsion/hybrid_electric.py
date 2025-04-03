"""
Hybrid Electric Propulsion Controller for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.propulsion.base import NeuromorphicPropulsion, PropulsionSpecs, PropulsionType
from src.propulsion.optimization import PropulsionOptimizer, OptimizationConstraints


class HybridElectricController:
    """Controller for hybrid electric propulsion systems."""
    
    def __init__(self, system_id: str, optimizer: Optional[PropulsionOptimizer] = None):
        """Initialize the hybrid electric controller."""
        self.system_id = system_id
        self.optimizer = optimizer
        self.electric_ratio = 0.5  # Default electric/combustion ratio
        self.battery_level = 1.0   # Normalized battery level (0-1)
        self.fuel_level = 1.0      # Normalized fuel level (0-1)
        
        # Operating modes
        self.modes = {
            "full_electric": {"electric_ratio": 1.0, "max_power": 0.8},
            "balanced": {"electric_ratio": 0.5, "max_power": 1.0},
            "power_boost": {"electric_ratio": 0.3, "max_power": 1.0},
            "fuel_saving": {"electric_ratio": 0.7, "max_power": 0.9},
            "emergency": {"electric_ratio": 0.0, "max_power": 1.0}
        }
        
        self.current_mode = "balanced"
        self.performance_history = []
        
    def set_mode(self, mode: str) -> bool:
        """Set the operating mode of the hybrid system."""
        if mode not in self.modes:
            return False
            
        self.current_mode = mode
        self.electric_ratio = self.modes[mode]["electric_ratio"]
        return True
        
    def update_energy_levels(self, battery_level: float, fuel_level: float) -> None:
        """Update current energy levels."""
        self.battery_level = max(0.0, min(1.0, battery_level))
        self.fuel_level = max(0.0, min(1.0, fuel_level))
        
        # Auto-switch to emergency mode if both levels are critically low
        if self.battery_level < 0.1 and self.fuel_level < 0.1:
            self.set_mode("emergency")
        # Auto-switch to fuel saving if fuel is low but battery is good
        elif self.fuel_level < 0.2 and self.battery_level > 0.4:
            self.set_mode("fuel_saving")
        # Auto-switch to full electric if battery is good but fuel is low
        elif self.fuel_level < 0.1 and self.battery_level > 0.3:
            self.set_mode("full_electric")
            
    def calculate_power_distribution(self, 
                                   requested_power: float) -> Dict[str, float]:
        """Calculate power distribution between electric and combustion."""
        mode_config = self.modes[self.current_mode]
        max_power = mode_config["max_power"]
        electric_ratio = mode_config["electric_ratio"]
        
        # Limit requested power to max power for current mode
        actual_power = min(requested_power, max_power)
        
        # Calculate electric and combustion power
        electric_power = actual_power * electric_ratio
        combustion_power = actual_power - electric_power
        
        # Adjust for available energy
        if electric_power > 0 and self.battery_level < 0.05:
            # Redirect electric power to combustion if battery is critically low
            combustion_power += electric_power
            electric_power = 0
            
        if combustion_power > 0 and self.fuel_level < 0.05:
            # Redirect combustion power to electric if fuel is critically low
            # (up to what the electric system can handle)
            additional_electric = min(combustion_power, max_power * 0.8 - electric_power)
            electric_power += additional_electric
            combustion_power -= additional_electric
            
        return {
            "electric_power": electric_power,
            "combustion_power": combustion_power,
            "total_power": electric_power + combustion_power
        }
        
    def optimize_for_conditions(self, 
                              flight_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Optimize hybrid system for current flight conditions."""
        if not self.optimizer:
            return {"success": False, "error": "No optimizer available"}
            
        # Create constraints based on current mode and energy levels
        mode_config = self.modes[self.current_mode]
        constraints = OptimizationConstraints(
            max_power=mode_config["max_power"],
            max_temperature=1200.0,  # Default max temperature in K
            min_efficiency=0.7,      # Minimum acceptable efficiency
            max_fuel_consumption=self.fuel_level * 0.1  # Limit based on fuel level
        )
        
        # Run optimization
        result = self.optimizer.optimize(
            self.system_id,
            flight_conditions,
            constraints
        )
        
        if result.get("success", False):
            # Apply electric ratio to the optimized settings
            power_level = result["settings"]["power_level"]
            power_distribution = self.calculate_power_distribution(power_level)
            
            # Update result with hybrid-specific information
            result["hybrid"] = {
                "mode": self.current_mode,
                "electric_ratio": self.electric_ratio,
                "power_distribution": power_distribution
            }
            
            # Record performance
            self.performance_history.append({
                "flight_conditions": flight_conditions,
                "mode": self.current_mode,
                "power_distribution": power_distribution,
                "performance": result["performance"]
            })
            
        return result
        
    def get_recommended_mode(self, 
                           flight_conditions: Dict[str, float]) -> str:
        """Get recommended mode for current flight conditions."""
        altitude = flight_conditions.get("altitude", 0)
        speed = flight_conditions.get("speed", 0)
        
        # Simple rule-based mode selection
        if altitude > 8000:  # High altitude
            if self.battery_level > 0.6:
                return "full_electric"  # Electric is more efficient at high altitude
            else:
                return "balanced"
        elif speed > 250:  # High speed
            return "power_boost"  # More combustion power for high speed
        elif self.fuel_level < 0.3:  # Low fuel
            return "fuel_saving"
        else:
            return "balanced"  # Default balanced mode
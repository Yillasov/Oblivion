"""
Propulsion optimization framework for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

from src.propulsion.base import PropulsionInterface, PropulsionSpecs
from src.propulsion.neuromorphic_control import NeuromorphicPropulsionController


@dataclass
class OptimizationConstraints:
    """Constraints for propulsion optimization."""
    max_power: float  # Maximum power in kW
    max_temperature: float  # Maximum temperature in K
    min_efficiency: float  # Minimum efficiency (0-1)
    max_fuel_consumption: float  # Maximum fuel consumption rate


class PropulsionOptimizer:
    """Simple propulsion optimization framework."""
    
    def __init__(self, controller: Optional[NeuromorphicPropulsionController] = None):
        """Initialize the propulsion optimizer."""
        self.controller = controller
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.optimization_history: Dict[str, List[Dict[str, float]]] = {}
        
    def register_system(self, system_id: str, system: PropulsionInterface) -> bool:
        """Register a propulsion system for optimization."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        self.optimization_history[system_id] = []
        return True
        
    def optimize(self, 
               system_id: str, 
               flight_conditions: Dict[str, float],
               constraints: OptimizationConstraints) -> Dict[str, Any]:
        """
        Optimize propulsion settings for given flight conditions.
        
        Args:
            system_id: ID of the propulsion system
            flight_conditions: Current flight conditions
            constraints: Optimization constraints
            
        Returns:
            Optimized settings and performance metrics
        """
        if system_id not in self.propulsion_systems:
            return {"success": False, "error": "System not found"}
            
        system = self.propulsion_systems[system_id]
        specs = system.get_specifications()
        
        # Simple optimization strategy
        # 1. Calculate baseline performance
        baseline_performance = system.calculate_performance(flight_conditions)
        
        # 2. Find optimal power setting
        optimal_power = self._find_optimal_power(
            system, 
            flight_conditions, 
            constraints
        )
        
        # 3. Apply optimized settings
        optimized_settings = {
            "power_level": optimal_power,
            "mode": self._determine_optimal_mode(flight_conditions, specs)
        }
        
        # 4. Calculate expected performance with optimized settings
        power_state = {"power_level": optimal_power}
        system.set_power_state(power_state)
        optimized_performance = system.calculate_performance(flight_conditions)
        
        # 5. Record optimization results
        result = {
            "success": True,
            "settings": optimized_settings,
            "performance": optimized_performance,
            "improvement": {
                "efficiency": optimized_performance.get("efficiency", 0) - 
                              baseline_performance.get("efficiency", 0),
                "thrust": optimized_performance.get("thrust", 0) - 
                          baseline_performance.get("thrust", 0)
            }
        }
        
        self.optimization_history[system_id].append(result)
        return result
    
    def _find_optimal_power(self, 
                          system: PropulsionInterface, 
                          flight_conditions: Dict[str, float],
                          constraints: OptimizationConstraints) -> float:
        """Find optimal power setting within constraints."""
        # Simple grid search for optimal power level
        best_efficiency = 0
        optimal_power = 0.1  # Minimum power level
        
        # Try different power levels
        for power_level in np.linspace(0.1, 1.0, 10):
            # Set power state
            system.set_power_state({"power_level": power_level})
            
            # Calculate performance
            performance = system.calculate_performance(flight_conditions)
            
            # Check constraints
            if (performance.get("temperature", 0) <= constraints.max_temperature and
                performance.get("fuel_consumption", 0) <= constraints.max_fuel_consumption):
                
                # Check if this is better than current best
                if performance.get("efficiency", 0) > best_efficiency:
                    best_efficiency = performance.get("efficiency", 0)
                    optimal_power = power_level
        
        return optimal_power
    
    def _determine_optimal_mode(self, 
                              flight_conditions: Dict[str, float],
                              specs: PropulsionSpecs) -> str:
        """Determine optimal operating mode based on flight conditions."""
        altitude = flight_conditions.get("altitude", 0)
        speed = flight_conditions.get("speed", 0)
        
        # Simple mode selection based on altitude and speed
        if altitude > specs.altitude_limits.get("cruise_max", 10000):
            return "high_altitude"
        elif speed > specs.operational_envelope.get("speed", {}).get("cruise", 0):
            return "high_speed"
        else:
            return "standard"
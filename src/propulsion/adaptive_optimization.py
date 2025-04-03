"""
Adaptive propulsion optimization system that learns and improves over time.
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
from src.propulsion.optimization import OptimizationConstraints, PropulsionOptimizer


@dataclass
class AdaptiveOptimizationConfig:
    """Configuration for adaptive optimization."""
    learning_rate: float  # Learning rate for parameter updates
    exploration_rate: float  # Rate of exploration vs exploitation
    memory_size: int  # Size of memory buffer for past optimizations
    adaptation_threshold: float  # Minimum performance change to trigger adaptation
    max_iterations: int  # Maximum optimization iterations


class AdaptivePropulsionOptimizer:
    """Adaptive propulsion optimization system that learns from experience."""
    
    def __init__(self, config: AdaptiveOptimizationConfig):
        """Initialize adaptive propulsion optimizer."""
        self.config = config
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.base_optimizer = PropulsionOptimizer()
        self.optimization_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.learned_parameters: Dict[str, Dict[str, float]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
    def register_system(self, system_id: str, system: PropulsionInterface) -> bool:
        """Register a propulsion system for adaptive optimization."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        self.base_optimizer.register_system(system_id, system)
        self.optimization_memory[system_id] = []
        self.performance_history[system_id] = []
        
        # Initialize learned parameters
        self.learned_parameters[system_id] = {
            "power_bias": 0.0,
            "efficiency_weight": 1.0,
            "thermal_sensitivity": 1.0,
            "altitude_factor": 1.0,
            "speed_factor": 1.0
        }
        
        return True
        
    def optimize(self, 
                system_id: str, 
                flight_conditions: Dict[str, float],
                constraints: OptimizationConstraints) -> Dict[str, Any]:
        """
        Perform adaptive optimization for a propulsion system.
        
        Args:
            system_id: ID of the propulsion system
            flight_conditions: Current flight conditions
            constraints: Optimization constraints
            
        Returns:
            Optimized settings and performance metrics
        """
        if system_id not in self.propulsion_systems:
            return {"success": False, "error": "System not found"}
            
        # Apply learned parameters to adjust constraints
        adapted_constraints = self._adapt_constraints(system_id, constraints, flight_conditions)
        
        # Run base optimization with adapted constraints
        result = self.base_optimizer.optimize(system_id, flight_conditions, adapted_constraints)
        
        if result["success"]:
            # Store optimization result in memory
            self._update_memory(system_id, flight_conditions, result)
            
            # Update learned parameters based on performance
            self._update_parameters(system_id, flight_conditions, result)
            
            # Track performance
            efficiency = result["performance"].get("efficiency", 0.0)
            self.performance_history[system_id].append(efficiency)
            
            # Add adaptive parameters to result
            result["adaptive_parameters"] = self.learned_parameters[system_id].copy()
            
        return result
    
    def _adapt_constraints(self, 
                         system_id: str, 
                         constraints: OptimizationConstraints,
                         flight_conditions: Dict[str, float]) -> OptimizationConstraints:
        """Adapt constraints based on learned parameters."""
        params = self.learned_parameters[system_id]
        
        # Create a copy of constraints to modify
        adapted = OptimizationConstraints(
            max_power=constraints.max_power,
            max_temperature=constraints.max_temperature,
            min_efficiency=constraints.min_efficiency,
            max_fuel_consumption=constraints.max_fuel_consumption
        )
        
        # Apply altitude factor
        altitude = flight_conditions.get("altitude", 0.0)
        altitude_effect = 1.0 + (altitude / 10000.0) * (params["altitude_factor"] - 1.0)
        
        # Apply speed factor
        speed = flight_conditions.get("speed", 0.0)
        speed_effect = 1.0 + (speed / 100.0) * (params["speed_factor"] - 1.0)
        
        # Adjust constraints based on learned parameters
        adapted.max_power *= (1.0 + params["power_bias"])
        adapted.min_efficiency *= params["efficiency_weight"]
        adapted.max_temperature *= params["thermal_sensitivity"]
        
        # Apply environmental effects
        adapted.max_power *= altitude_effect
        adapted.min_efficiency *= speed_effect
        
        return adapted
    
    def _update_memory(self, 
                     system_id: str, 
                     conditions: Dict[str, float],
                     result: Dict[str, Any]) -> None:
        """Update optimization memory with new result."""
        memory = self.optimization_memory[system_id]
        
        # Create memory entry
        entry = {
            "conditions": conditions.copy(),
            "settings": result["settings"],
            "performance": result["performance"],
            "parameters": self.learned_parameters[system_id].copy()
        }
        
        # Add to memory
        memory.append(entry)
        
        # Limit memory size
        if len(memory) > self.config.memory_size:
            memory.pop(0)
    
    def _update_parameters(self, 
                         system_id: str, 
                         conditions: Dict[str, float],
                         result: Dict[str, Any]) -> None:
        """Update learned parameters based on optimization result."""
        # Extract performance metrics
        efficiency = result["performance"].get("efficiency", 0.0)
        temperature = result["performance"].get("temperature", 0.0)
        power = result["settings"].get("power_level", 0.0)
        
        # Get current parameters
        params = self.learned_parameters[system_id]
        
        # Calculate performance improvement
        history = self.performance_history[system_id]
        if len(history) > 1:
            improvement = efficiency - history[-2]
        else:
            improvement = 0.0
            
        # Only adapt if improvement is significant
        if abs(improvement) < self.config.adaptation_threshold:
            return
            
        # Update parameters based on performance
        lr = self.config.learning_rate
        
        # If efficiency improved, reinforce current parameter changes
        if improvement > 0:
            # Adjust power bias based on power level
            if power > 0.8:
                params["power_bias"] += lr * 0.1  # Increase power if high power worked well
            else:
                params["power_bias"] -= lr * 0.1  # Decrease power if low power worked well
                
            # Adjust efficiency weight
            params["efficiency_weight"] += lr * 0.05
            
            # Adjust environmental factors
            altitude = conditions.get("altitude", 0.0)
            if altitude > 5000:
                params["altitude_factor"] += lr * 0.1
                
            speed = conditions.get("speed", 0.0)
            if speed > 50:
                params["speed_factor"] += lr * 0.1
        else:
            # If performance decreased, reverse direction
            params["power_bias"] -= lr * 0.05 * np.sign(params["power_bias"])
            params["efficiency_weight"] -= lr * 0.02 * np.sign(params["efficiency_weight"] - 1.0)
            params["thermal_sensitivity"] += lr * 0.05  # Increase thermal sensitivity
            
        # Apply constraints to parameters
        params["power_bias"] = max(-0.2, min(0.2, params["power_bias"]))
        params["efficiency_weight"] = max(0.8, min(1.2, params["efficiency_weight"]))
        params["thermal_sensitivity"] = max(0.9, min(1.1, params["thermal_sensitivity"]))
        params["altitude_factor"] = max(0.8, min(1.2, params["altitude_factor"]))
        params["speed_factor"] = max(0.8, min(1.2, params["speed_factor"]))
    
    def get_optimization_stats(self, system_id: str) -> Dict[str, Any]:
        """Get optimization statistics for a system."""
        if system_id not in self.propulsion_systems:
            return {"success": False, "error": "System not found"}
            
        history = self.performance_history[system_id]
        
        if not history:
            return {"success": True, "stats": {}, "parameters": {}}
            
        # Calculate statistics
        avg_efficiency = sum(history) / len(history)
        improvement = history[-1] - history[0] if len(history) > 1 else 0
        
        return {
            "success": True,
            "stats": {
                "average_efficiency": avg_efficiency,
                "total_improvement": improvement,
                "optimization_count": len(history)
            },
            "parameters": self.learned_parameters[system_id]
        }
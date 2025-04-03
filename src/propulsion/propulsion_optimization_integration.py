"""
Propulsion-specific optimization integration system.
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

from src.propulsion.base import PropulsionInterface, PropulsionSpecs
from src.propulsion.optimization import PropulsionOptimizer, OptimizationConstraints
from src.propulsion.adaptive_optimization import AdaptivePropulsionOptimizer
from src.propulsion.combustion_optimization import CombustionOptimizer
from src.propulsion.environmental_interaction import EnvironmentalInteractionSystem
from src.propulsion.energy_harvesting import EnergyHarvestingSystem
from src.propulsion.energy_payload_integration import EnergyPayloadIntegrator


class OptimizationStrategy(Enum):
    """Optimization strategies for propulsion systems."""
    EFFICIENCY = 0
    THRUST = 1
    THERMAL = 2
    FUEL_ECONOMY = 3
    STEALTH = 4
    HYBRID = 5


@dataclass
class PropulsionOptimizationConfig:
    """Configuration for propulsion optimization integration."""
    strategy: OptimizationStrategy
    adaptation_weight: float  # Weight for adaptive optimization (0-1)
    environmental_weight: float  # Weight for environmental adaptation (0-1)
    energy_harvesting_weight: float  # Weight for energy harvesting integration (0-1)
    max_iterations: int  # Maximum optimization iterations
    convergence_threshold: float  # Convergence threshold for optimization


class PropulsionOptimizationIntegrator:
    """Integrates multiple optimization systems for propulsion."""
    
    def __init__(self, 
                 config: PropulsionOptimizationConfig,
                 adaptive_optimizer: Optional[AdaptivePropulsionOptimizer] = None,
                 env_system: Optional[EnvironmentalInteractionSystem] = None,
                 energy_system: Optional[EnergyHarvestingSystem] = None,
                 energy_payload_integrator: Optional[EnergyPayloadIntegrator] = None):
        """Initialize propulsion optimization integrator."""
        self.config = config
        self.adaptive_optimizer = adaptive_optimizer
        self.env_system = env_system
        self.energy_system = energy_system
        self.energy_payload_integrator = energy_payload_integrator
        
        # Create base optimizer if needed
        self.base_optimizer = PropulsionOptimizer()
        self.combustion_optimizer = CombustionOptimizer()
        
        # Track systems and optimization history
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = {}
        self.strategy_performance: Dict[OptimizationStrategy, List[float]] = {
            strategy: [] for strategy in OptimizationStrategy
        }
        
    def register_system(self, system_id: str, system: PropulsionInterface) -> bool:
        """Register a propulsion system for integrated optimization."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        self.optimization_history[system_id] = []
        
        # Register with component optimizers
        self.base_optimizer.register_system(system_id, system)
        
        if self.adaptive_optimizer:
            self.adaptive_optimizer.register_system(system_id, system)
            
        if self.env_system:
            self.env_system.register_system(system_id, system)
            
        if system.get_specifications().propulsion_type.name == "COMBUSTION":
            self.combustion_optimizer.register_system(system_id, system)
            
        return True
        
    def set_optimization_strategy(self, strategy: OptimizationStrategy) -> None:
        """Set the optimization strategy."""
        self.config.strategy = strategy
        
    def create_strategy_constraints(self, 
                                  base_constraints: OptimizationConstraints,
                                  system_id: str,
                                  flight_conditions: Dict[str, float]) -> OptimizationConstraints:
        """Create strategy-specific constraints."""
        specs = self.propulsion_systems[system_id].get_specifications()
        
        # Start with base constraints
        constraints = OptimizationConstraints(
            max_power=base_constraints.max_power,
            max_temperature=base_constraints.max_temperature,
            min_efficiency=base_constraints.min_efficiency,
            max_fuel_consumption=base_constraints.max_fuel_consumption
        )
        
        # Adjust based on strategy
        if self.config.strategy == OptimizationStrategy.EFFICIENCY:
            constraints.min_efficiency += 0.1  # Increase efficiency requirement
            constraints.max_power *= 0.9  # Reduce power to improve efficiency
            
        elif self.config.strategy == OptimizationStrategy.THRUST:
            constraints.max_power *= 1.1  # Allow more power for thrust
            constraints.min_efficiency -= 0.05  # Allow lower efficiency for thrust
            constraints.max_temperature *= 1.05  # Allow higher temperatures
            
        elif self.config.strategy == OptimizationStrategy.THERMAL:
            constraints.max_temperature *= 0.9  # Reduce temperature limit
            constraints.max_power *= 0.95  # Reduce power to manage heat
            
        elif self.config.strategy == OptimizationStrategy.FUEL_ECONOMY:
            constraints.max_fuel_consumption *= 0.8  # Reduce fuel consumption
            constraints.min_efficiency += 0.05  # Require higher efficiency
            
        elif self.config.strategy == OptimizationStrategy.STEALTH:
            # Stealth mode - reduce thermal signature and power
            constraints.max_temperature *= 0.85
            constraints.max_power *= 0.8
            
        # Apply environmental adaptations if available
        if self.env_system and self.config.environmental_weight > 0:
            env_params = self.env_system.get_adaptation_parameters(system_id)
            if env_params:
                constraints.max_power *= env_params.get("power_adjustment", 1.0) * self.config.environmental_weight
                constraints.max_temperature *= env_params.get("thermal_adjustment", 1.0) * self.config.environmental_weight
                constraints.min_efficiency *= env_params.get("efficiency_adjustment", 1.0) * self.config.environmental_weight
        
        # Apply energy harvesting adaptations if available
        if self.energy_system and self.config.energy_harvesting_weight > 0:
            harvested_power = self.energy_system.get_total_output()
            if harvested_power > 0:
                # Adjust power constraints based on harvested energy
                power_boost = harvested_power * self.config.energy_harvesting_weight
                constraints.max_power += power_boost
        
        return constraints
    
    def optimize(self, 
                system_id: str, 
                flight_conditions: Dict[str, float],
                base_constraints: OptimizationConstraints,
                env_condition: Any = None) -> Dict[str, Any]:
        """
        Perform integrated optimization for a propulsion system.
        
        Args:
            system_id: ID of the propulsion system
            flight_conditions: Current flight conditions
            base_constraints: Base optimization constraints
            env_condition: Environmental conditions (optional)
            
        Returns:
            Integrated optimization results
        """
        if system_id not in self.propulsion_systems:
            return {"success": False, "error": "System not found"}
        
        system = self.propulsion_systems[system_id]
        specs = system.get_specifications()
        
        # Update environmental system if provided
        if self.env_system and env_condition:
            self.env_system.update_environment(env_condition)
        
        # Update energy harvesting if available
        if self.energy_system and env_condition:
            self.energy_system.update(env_condition, 1.0)  # 1 second update
        
        # Create strategy-specific constraints
        strategy_constraints = self.create_strategy_constraints(
            base_constraints, system_id, flight_conditions
        )
        
        # Perform optimization based on propulsion type
        propulsion_type = specs.propulsion_type.name
        
        if propulsion_type == "COMBUSTION":
            # Optimize combustion parameters
            combustion_result = self.combustion_optimizer.optimize_combustion(
                system_id, flight_conditions, strategy_constraints
            )
            self.combustion_optimizer.apply_parameters(system_id)
        
        # Perform adaptive optimization if available
        if self.adaptive_optimizer and self.config.adaptation_weight > 0:
            adaptive_result = self.adaptive_optimizer.optimize(
                system_id, flight_conditions, strategy_constraints
            )
            optimization_result = adaptive_result
        else:
            # Use base optimizer
            optimization_result = self.base_optimizer.optimize(
                system_id, flight_conditions, strategy_constraints
            )
        
        # Apply optimized settings
        if optimization_result["success"]:
            system.set_power_state(optimization_result["settings"])
            
            # Track strategy performance
            if "performance" in optimization_result:
                efficiency = optimization_result["performance"].get("efficiency", 0.0)
                self.strategy_performance[self.config.strategy].append(efficiency)
        
        # Integrate with energy payload system if available
        if self.energy_payload_integrator and self.energy_system:
            if self.energy_payload_integrator.integration_active:
                energy_integration = self.energy_payload_integrator.update(env_condition, 1.0)
                optimization_result["energy_integration"] = energy_integration
        
        # Record optimization history
        self.optimization_history[system_id].append({
            "timestamp": np.datetime64('now'),
            "strategy": self.config.strategy.name,
            "constraints": {
                "max_power": strategy_constraints.max_power,
                "max_temperature": strategy_constraints.max_temperature,
                "min_efficiency": strategy_constraints.min_efficiency,
                "max_fuel_consumption": strategy_constraints.max_fuel_consumption
            },
            "result": optimization_result
        })
        
        # Limit history size
        if len(self.optimization_history[system_id]) > 100:
            self.optimization_history[system_id].pop(0)
            
        return optimization_result
    
    def get_optimization_report(self, system_id: str) -> Dict[str, Any]:
        """Get optimization report for a specific system."""
        if system_id not in self.propulsion_systems:
            return {"error": "System not found"}
            
        system = self.propulsion_systems[system_id]
        specs = system.get_specifications()
        
        # Get current system status
        status = system.get_status()
        
        # Get optimization history
        history = self.optimization_history[system_id]
        
        # Calculate strategy performance
        strategy_stats = {}
        for strategy, performance in self.strategy_performance.items():
            if performance:
                strategy_stats[strategy.name] = {
                    "avg_efficiency": np.mean(performance),
                    "max_efficiency": np.max(performance),
                    "samples": len(performance)
                }
        
        # Compile report
        return {
            "system_id": system_id,
            "propulsion_type": specs.propulsion_type.name,
            "current_status": status,
            "current_strategy": self.config.strategy.name,
            "optimization_count": len(history),
            "last_optimization": history[-1] if history else None,
            "strategy_performance": strategy_stats,
            "adaptive_optimization": self.adaptive_optimizer is not None,
            "environmental_adaptation": self.env_system is not None,
            "energy_harvesting": self.energy_system is not None
        }
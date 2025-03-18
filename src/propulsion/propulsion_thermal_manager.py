"""
Integrated Thermal Management System for Electric and Combustion Propulsion.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.propulsion.thermal_management import ThermalManagementSystem, ThermalProfile
from src.propulsion.combustion_optimization import CombustionOptimizer, CombustionParameters
from src.propulsion.hybrid_electric import HybridElectricController
from src.propulsion.base import PropulsionInterface
from src.propulsion.optimization import OptimizationConstraints  # Import the OptimizationConstraints class


class CoolingMode(Enum):
    """Cooling modes for propulsion thermal management."""
    PASSIVE = 0
    ACTIVE_LOW = 1
    ACTIVE_HIGH = 2
    EMERGENCY = 3


class PropulsionThermalManager:
    """Integrated thermal management for electric and combustion propulsion."""
    
    def __init__(self, 
                thermal_system: ThermalManagementSystem,
                combustion_optimizer: Optional[CombustionOptimizer] = None):
        """Initialize propulsion thermal manager."""
        self.thermal_system = thermal_system
        self.combustion_optimizer = combustion_optimizer
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.cooling_mode = CoolingMode.PASSIVE
        self.thermal_limits: Dict[str, Dict[str, float]] = {}
        self.critical_components: List[str] = []
        
    def register_propulsion_system(self, 
                                 system_id: str, 
                                 system: PropulsionInterface,
                                 thermal_limits: Dict[str, float]) -> bool:
        """Register a propulsion system for thermal management."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        self.thermal_limits[system_id] = thermal_limits
        
        # Register with combustion optimizer if applicable
        if self.combustion_optimizer and hasattr(system, 'get_combustion_parameters'):
            self.combustion_optimizer.register_system(system_id, system)
            
        return True
        
    def register_critical_component(self, component_id: str) -> None:
        """Register a component as critical for thermal management."""
        if component_id not in self.critical_components:
            self.critical_components.append(component_id)
    
    def update_thermal_state(self, 
                           flight_conditions: Dict[str, float],
                           power_levels: Dict[str, float],
                           dt: float) -> Dict[str, Any]:
        """
        Update thermal state of all propulsion systems.
        
        Args:
            flight_conditions: Current flight conditions
            power_levels: Current power levels for each system
            dt: Time step in seconds
            
        Returns:
            Thermal management results
        """
        # Update thermal system
        thermal_results = self.thermal_system.manage_thermal_conditions(
            flight_conditions, power_levels, dt
        )
        
        # Check for critical thermal conditions
        critical_systems = []
        for component_id, status in thermal_results["thermal_status"].items():
            if status["critical"] and component_id in self.critical_components:
                critical_systems.append(component_id)
        
        # Adjust cooling mode based on thermal status
        if critical_systems:
            if len(critical_systems) > 1:
                self.cooling_mode = CoolingMode.EMERGENCY
            else:
                self.cooling_mode = CoolingMode.ACTIVE_HIGH
        elif any(status["thermal_margin"] < 0.4 
                for status in thermal_results["thermal_status"].values()):
            self.cooling_mode = CoolingMode.ACTIVE_LOW
        else:
            self.cooling_mode = CoolingMode.PASSIVE
            
        # Adjust combustion parameters if needed
        if self.combustion_optimizer and self.cooling_mode in [CoolingMode.ACTIVE_HIGH, CoolingMode.EMERGENCY]:
            for system_id, system in self.propulsion_systems.items():
                if hasattr(system, 'get_combustion_parameters'):
                    # Optimize combustion for thermal management
                    self._optimize_combustion_for_thermal(system_id, flight_conditions)
        
        return {
            "thermal_results": thermal_results,
            "cooling_mode": self.cooling_mode.name,
            "critical_systems": critical_systems,
            "cooling_power": thermal_results["cooling_power"]
        }
    
    def _optimize_combustion_for_thermal(self, 
                                       system_id: str, 
                                       flight_conditions: Dict[str, float]) -> None:
        """Optimize combustion parameters for thermal management."""
        if not self.combustion_optimizer:
            return
            
        # Create thermal constraints as an OptimizationConstraints object
        max_temp = self.thermal_limits.get(system_id, {}).get("max_temp", 1200.0)
        
        # Create a proper OptimizationConstraints object
        thermal_constraints = OptimizationConstraints(
            max_power=float('inf'),  # No power limit for thermal optimization
            max_temperature=max_temp,
            min_efficiency=0.0,  # No minimum efficiency for thermal optimization
            max_fuel_consumption=float('inf')  # No fuel consumption limit for thermal optimization
        )
        
        # Optimize combustion
        self.combustion_optimizer.optimize_combustion(
            system_id, 
            flight_conditions,
            thermal_constraints
        )
        
        # Apply optimized parameters
        self.combustion_optimizer.apply_parameters(system_id)
    
    def get_recommended_power_levels(self, 
                                   requested_power: Dict[str, float]) -> Dict[str, float]:
        """
        Get recommended power levels based on thermal constraints.
        
        Args:
            requested_power: Requested power levels for each system
            
        Returns:
            Recommended power levels
        """
        recommended_power = requested_power.copy()
        
        # Adjust power levels based on cooling mode
        if self.cooling_mode == CoolingMode.EMERGENCY:
            # Reduce power for all systems in emergency mode
            for system_id in recommended_power:
                recommended_power[system_id] *= 0.6
        elif self.cooling_mode == CoolingMode.ACTIVE_HIGH:
            # Reduce power for critical systems
            for component_id in self.critical_components:
                if component_id in recommended_power:
                    recommended_power[component_id] *= 0.8
                    
        return recommended_power
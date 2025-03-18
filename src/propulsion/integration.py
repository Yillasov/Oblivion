"""
Propulsion system integration framework for UCAV platforms.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.propulsion.base import PropulsionInterface, PropulsionSpecs, PropulsionType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


@dataclass
class PropulsionIntegrationConfig:
    """Configuration for propulsion system integration."""
    max_power_draw: float  # Maximum power draw in kW
    thermal_threshold: float  # Maximum thermal load in kW
    response_time_limit: float  # Maximum allowed response time in seconds
    safety_margin: float  # Safety margin for critical operations
    redundancy_level: int  # Level of system redundancy


class PropulsionIntegrator:
    """Framework for integrating propulsion systems with UCAV platform."""
    
    def __init__(self, 
                 config: PropulsionIntegrationConfig,
                 neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """
        Initialize the propulsion integrator.
        
        Args:
            config: Integration configuration
            neuromorphic_system: Neuromorphic system interface
        """
        self.config = config
        self.neuromorphic_system = neuromorphic_system
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.system_states: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[Dict[str, float]]] = {}
        self.active_configurations: Dict[str, bool] = {}
        
    def register_propulsion_system(self, 
                                 system_id: str, 
                                 system: PropulsionInterface) -> bool:
        """Register a propulsion system with the integrator."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        self.system_states[system_id] = {
            "initialized": False,
            "active": False,
            "health": 1.0,
            "thermal_load": 0.0,
            "power_draw": 0.0
        }
        self.performance_history[system_id] = []
        return True
        
    def initialize_systems(self) -> Dict[str, bool]:
        """Initialize all registered propulsion systems."""
        results = {}
        for system_id, system in self.propulsion_systems.items():
            success = system.initialize()
            self.system_states[system_id]["initialized"] = success
            results[system_id] = success
        return results
        
    def configure_neuromorphic_control(self, 
                                     system_id: str,
                                     control_params: Dict[str, Any]) -> bool:
        """Configure neuromorphic control for a specific system."""
        if not self.neuromorphic_system or system_id not in self.propulsion_systems:
            return False
            
        system = self.propulsion_systems[system_id]
        specs = system.get_specifications()
        
        # Configure neural network for propulsion control
        control_config = {
            "input_dimensions": len(control_params["input_mapping"]),
            "output_dimensions": len(control_params["output_mapping"]),
            "response_time": specs.thermal_response_time,
            "control_frequency": control_params.get("control_frequency", 100),
            "adaptation_rate": control_params.get("adaptation_rate", 0.1)
        }
        
        return self.neuromorphic_system.add_component(
            f"propulsion_control_{system_id}",
            control_config
        )
        
    def monitor_thermal_conditions(self) -> Dict[str, Dict[str, float]]:
        """Monitor thermal conditions of all systems."""
        thermal_status = {}
        for system_id, system in self.propulsion_systems.items():
            status = system.get_status()
            specs = system.get_specifications()
            
            thermal_load = status.get("temperature", 0.0)
            thermal_limit = specs.thermal_limits.get("max_operating", float('inf'))
            thermal_margin = (thermal_limit - thermal_load) / thermal_limit
            
            thermal_status[system_id] = {
                "current_load": thermal_load,
                "limit": thermal_limit,
                "margin": thermal_margin,
                "critical": thermal_margin < self.config.safety_margin
            }
        return thermal_status
        
    def optimize_power_distribution(self) -> Dict[str, float]:
        """Optimize power distribution across propulsion systems."""
        power_allocation = {}
        total_power_available = self.config.max_power_draw
        
        # Get power requirements and priorities
        power_requirements = {}
        for system_id, system in self.propulsion_systems.items():
            if self.system_states[system_id]["active"]:
                specs = system.get_specifications()
                power_requirements[system_id] = specs.power_rating
                
        # Allocate power based on priorities and requirements
        remaining_power = total_power_available
        for system_id, required_power in power_requirements.items():
            allocated = min(required_power, remaining_power)
            power_allocation[system_id] = allocated
            remaining_power -= allocated
            
        return power_allocation
        
    def update_system_states(self, 
                           flight_conditions: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Update and return the state of all propulsion systems."""
        states = {}
        for system_id, system in self.propulsion_systems.items():
            if self.system_states[system_id]["active"]:
                performance = system.calculate_performance(flight_conditions)
                status = system.get_status()
                
                states[system_id] = {
                    "performance": performance,
                    "status": status,
                    "health": self.system_states[system_id]["health"]
                }
                
                # Update performance history
                self.performance_history[system_id].append(performance)
                
        return states
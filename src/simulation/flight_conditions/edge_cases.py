"""
Advanced Flight Conditions and Edge Cases Handler

Provides support for simulating extreme flight conditions and edge cases
in the Oblivion simulation framework.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import logging

from src.core.utils.logging_framework import get_logger
from src.simulation.core.scheduler import TaskConfig

logger = get_logger("flight_conditions")

class EdgeCaseType(Enum):
    """Types of flight edge cases that can be simulated."""
    EXTREME_TURBULENCE = auto()
    ICING = auto()
    SENSOR_FAILURE = auto()
    ENGINE_FLAMEOUT = auto()
    CONTROL_SURFACE_JAM = auto()
    SUDDEN_CROSSWIND = auto()
    BIRD_STRIKE = auto()
    GPS_DENIAL = auto()
    COMMUNICATION_LOSS = auto()
    RADAR_JAMMING = auto()

@dataclass
class EdgeCaseConfig:
    """Configuration for an edge case scenario."""
    type: EdgeCaseType
    severity: float  # 0.0 to 1.0
    duration: float  # seconds
    trigger_condition: Optional[Dict[str, Any]] = None
    recovery_behavior: Optional[str] = None

class FlightConditionsManager:
    """Manages advanced flight conditions and edge cases."""
    
    def __init__(self):
        self.active_edge_cases = {}
        self.registered_handlers = {}
        self.flight_conditions = {}
        self.recovery_strategies = {}
        self.shared_state = {}
        
    def register_edge_case_handler(self, edge_case_type: EdgeCaseType, 
                                  handler: Callable[[Dict[str, Any], float], Dict[str, Any]]) -> None:
        """Register a handler for a specific edge case type."""
        self.registered_handlers[edge_case_type] = handler
        
    def register_recovery_strategy(self, edge_case_type: EdgeCaseType,
                                  strategy: Callable[[Dict[str, Any]], None]) -> None:
        """Register a recovery strategy for an edge case."""
        self.recovery_strategies[edge_case_type] = strategy
        
    def trigger_edge_case(self, edge_case: EdgeCaseConfig) -> bool:
        """Trigger a specific edge case."""
        if edge_case.type in self.active_edge_cases:
            logger.warning(f"Edge case {edge_case.type.name} already active")
            return False
            
        self.active_edge_cases[edge_case.type] = {
            "config": edge_case,
            "start_time": 0.0,
            "end_time": edge_case.duration
        }
        
        logger.info(f"Triggered edge case: {edge_case.type.name} (severity: {edge_case.severity})")
        return True
        
    def update(self, sim_time: float, flight_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update flight conditions with active edge cases.
        
        Args:
            sim_time: Current simulation time
            flight_data: Current flight data
            
        Returns:
            Updated flight data with edge case effects
        """
        # Store current base conditions
        self.flight_conditions = flight_data.copy()
        
        # Process active edge cases
        edge_cases_to_remove = []
        for edge_type, edge_data in self.active_edge_cases.items():
            # Update timing
            if edge_data["start_time"] == 0.0:
                edge_data["start_time"] = sim_time
                edge_data["end_time"] = sim_time + edge_data["config"].duration
            
            # Check if edge case has expired
            if sim_time >= edge_data["end_time"]:
                edge_cases_to_remove.append(edge_type)
                # Apply recovery if available
                if edge_type in self.recovery_strategies:
                    self.recovery_strategies[edge_type](flight_data)
                continue
                
            # Apply edge case effect if handler exists
            if edge_type in self.registered_handlers:
                flight_data = self.registered_handlers[edge_type](
                    flight_data, 
                    edge_data["config"].severity
                )
        
        # Remove expired edge cases
        for edge_type in edge_cases_to_remove:
            logger.info(f"Edge case ended: {edge_type.name}")
            del self.active_edge_cases[edge_type]
            
        return flight_data

    def get_task_config(self) -> TaskConfig:
        """Get scheduler task configuration for the manager."""
        return TaskConfig(
            name="update_flight_conditions",
            update_rate=50.0,
            priority=2,
            group="physics"
        )

    def update_shared_state(self, data: Dict[str, Any]) -> None:
        """
        Update the shared flight data state.
        
        Args:
            data: Flight data to update in the shared state
        """
        self.shared_state.update({"flight_data": data})
        
    def get_shared_state(self) -> Dict[str, Any]:
        """
        Get the current shared flight data state.
        
        Returns:
            Current flight data from shared state
        """
        return self.shared_state.get("flight_data", {})

# Default handlers for common edge cases
def handle_turbulence(flight_data: Dict[str, Any], severity: float) -> Dict[str, Any]:
    """Handle extreme turbulence edge case."""
    # Add random velocity perturbations based on severity
    if "platform" in flight_data and "velocity" in flight_data["platform"]:
        max_perturbation = 10.0 * severity  # m/s
        perturbation = np.random.normal(0, max_perturbation, 3)
        flight_data["platform"]["velocity"] = [
            v + p for v, p in zip(flight_data["platform"]["velocity"], perturbation)
        ]
    return flight_data

def handle_icing(flight_data: Dict[str, Any], severity: float) -> Dict[str, Any]:
    """Handle icing conditions edge case."""
    # Reduce lift and increase drag based on severity
    if "aerodynamics" not in flight_data:
        flight_data["aerodynamics"] = {}
    
    flight_data["aerodynamics"]["lift_degradation"] = severity * 0.5  # Up to 50% lift loss
    flight_data["aerodynamics"]["drag_increase"] = severity * 0.3     # Up to 30% drag increase
    
    return flight_data
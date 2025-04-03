"""
Fuel Flow Control System for UCAV propulsion systems.
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


class FuelFlowMode(Enum):
    """Operating modes for fuel flow control."""
    IDLE = 0
    NORMAL = 1
    PERFORMANCE = 2
    ECONOMY = 3
    EMERGENCY = 4


@dataclass
class FuelFlowProfile:
    """Fuel flow characteristics for a propulsion system."""
    min_flow_rate: float  # Minimum fuel flow rate in kg/s
    max_flow_rate: float  # Maximum fuel flow rate in kg/s
    optimal_flow_rate: float  # Optimal fuel flow rate in kg/s
    response_time: float  # Flow rate change response time in seconds
    pressure_range: Tuple[float, float]  # Min/max fuel pressure in kPa


class FuelFlowController:
    """Simple fuel flow control system for propulsion."""
    
    def __init__(self):
        """Initialize fuel flow controller."""
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.flow_profiles: Dict[str, FuelFlowProfile] = {}
        self.current_flow_rates: Dict[str, float] = {}
        self.current_modes: Dict[str, FuelFlowMode] = {}
        self.flow_history: Dict[str, List[Dict[str, float]]] = {}
        
    def register_system(self, 
                       system_id: str, 
                       system: PropulsionInterface,
                       flow_profile: FuelFlowProfile) -> bool:
        """Register a propulsion system for fuel flow control."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        self.flow_profiles[system_id] = flow_profile
        self.current_flow_rates[system_id] = 0.0
        self.current_modes[system_id] = FuelFlowMode.IDLE
        self.flow_history[system_id] = []
        
        return True
        
    def set_flow_mode(self, system_id: str, mode: FuelFlowMode) -> bool:
        """Set fuel flow mode for a propulsion system."""
        if system_id not in self.propulsion_systems:
            return False
            
        self.current_modes[system_id] = mode
        
        # Adjust flow rate based on mode
        profile = self.flow_profiles[system_id]
        if mode == FuelFlowMode.IDLE:
            target_flow = profile.min_flow_rate
        elif mode == FuelFlowMode.NORMAL:
            target_flow = profile.optimal_flow_rate
        elif mode == FuelFlowMode.PERFORMANCE:
            target_flow = profile.max_flow_rate
        elif mode == FuelFlowMode.ECONOMY:
            target_flow = profile.optimal_flow_rate * 0.8
        elif mode == FuelFlowMode.EMERGENCY:
            target_flow = profile.min_flow_rate * 1.5
            
        self.set_flow_rate(system_id, target_flow)
        return True
    
    def set_flow_rate(self, system_id: str, flow_rate: float) -> float:
        """Set fuel flow rate for a propulsion system."""
        if system_id not in self.propulsion_systems:
            return 0.0
            
        profile = self.flow_profiles[system_id]
        
        # Clamp flow rate to valid range
        clamped_flow = max(profile.min_flow_rate, 
                          min(flow_rate, profile.max_flow_rate))
        
        self.current_flow_rates[system_id] = clamped_flow
        
        # Apply to propulsion system using a more generic approach
        system = self.propulsion_systems[system_id]
        try:
            # Try direct method if available
            if hasattr(system, 'set_fuel_flow'):
                system.set_fuel_flow(clamped_flow)
            # Alternative: use set_power_state with fuel flow parameter
            elif hasattr(system, 'set_power_state'):
                system.set_power_state({"fuel_flow_rate": clamped_flow})
        except Exception as e:
            # Log error but continue operation
            print(f"Warning: Could not set fuel flow for {system_id}: {e}")
        
        return clamped_flow
    
    def update_flow_rates(self, 
                        flight_conditions: Dict[str, float],
                        power_demands: Dict[str, float],
                        dt: float) -> Dict[str, Dict[str, float]]:
        """
        Update fuel flow rates based on flight conditions and power demands.
        
        Args:
            flight_conditions: Current flight conditions
            power_demands: Power demand for each system
            dt: Time step in seconds
            
        Returns:
            Updated flow rates and status for each system
        """
        results = {}
        
        for system_id, system in self.propulsion_systems.items():
            if system_id not in power_demands:
                continue
                
            profile = self.flow_profiles[system_id]
            mode = self.current_modes[system_id]
            
            # Calculate target flow rate based on power demand and mode
            power_level = power_demands[system_id]
            
            if mode == FuelFlowMode.PERFORMANCE:
                # Higher flow rate for performance mode
                target_flow = profile.min_flow_rate + power_level * (
                    profile.max_flow_rate - profile.min_flow_rate)
            elif mode == FuelFlowMode.ECONOMY:
                # Lower flow rate for economy mode
                max_economy_flow = profile.optimal_flow_rate * 0.9
                target_flow = profile.min_flow_rate + power_level * (
                    max_economy_flow - profile.min_flow_rate)
            else:
                # Normal flow rate calculation
                target_flow = profile.min_flow_rate + power_level * (
                    profile.optimal_flow_rate - profile.min_flow_rate)
            
            # Apply gradual change based on response time
            current_flow = self.current_flow_rates[system_id]
            max_change = dt / profile.response_time * (profile.max_flow_rate - profile.min_flow_rate)
            
            if abs(target_flow - current_flow) <= max_change:
                new_flow = target_flow
            else:
                new_flow = current_flow + max_change * np.sign(target_flow - current_flow)
                
            # Set the new flow rate
            actual_flow = self.set_flow_rate(system_id, new_flow)
            
            # Calculate fuel efficiency
            if mode == FuelFlowMode.ECONOMY:
                efficiency_factor = 1.2  # Better efficiency in economy mode
            elif mode == FuelFlowMode.PERFORMANCE:
                efficiency_factor = 0.8  # Lower efficiency in performance mode
            else:
                efficiency_factor = 1.0
                
            # Record flow data
            flow_data = {
                "timestamp": flight_conditions.get("timestamp", 0.0),
                "flow_rate": actual_flow,
                "power_level": power_level,
                "mode": mode.name,
                "efficiency_factor": efficiency_factor
            }
            self.flow_history[system_id].append(flow_data)
            
            # Prepare result
            results[system_id] = {
                "flow_rate": actual_flow,
                "mode": mode.name,
                "efficiency": efficiency_factor,
                "min_flow": profile.min_flow_rate,
                "max_flow": profile.max_flow_rate,
                "optimal_flow": profile.optimal_flow_rate
            }
            
        return results
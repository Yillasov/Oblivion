"""
Scramjet Simulation and Control System for UCAV platforms.
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

from src.propulsion.base import PropulsionInterface, PropulsionSpecs, PropulsionType


class ScramjetMode(Enum):
    """Operating modes for scramjet engine."""
    STARTUP = 0
    SUBSONIC = 1
    TRANSONIC = 2
    SUPERSONIC = 3
    HYPERSONIC = 4
    SHUTDOWN = 5


@dataclass
class ScramjetSpecs:
    """Specifications for scramjet propulsion system."""
    max_thrust: float  # Maximum thrust in kN
    specific_impulse: float  # Specific impulse in seconds
    min_mach: float  # Minimum operational Mach number
    max_mach: float  # Maximum operational Mach number
    inlet_area: float  # Inlet area in m²
    weight: float  # Weight in kg
    fuel_type: str  # Type of fuel used


class ScramjetController:
    """Simple controller for scramjet propulsion system."""
    
    def __init__(self, specs: ScramjetSpecs):
        """Initialize scramjet controller."""
        self.specs = specs
        self.mode = ScramjetMode.SHUTDOWN
        self.throttle = 0.0
        self.current_mach = 0.0
        self.fuel_flow = 0.0
        self.inlet_temperature = 288.0  # K
        self.combustion_temperature = 288.0  # K
        self.nozzle_temperature = 288.0  # K
        self.performance_history: List[Dict[str, float]] = []
        
    def set_mode(self, mode: ScramjetMode) -> bool:
        """Set operating mode of scramjet."""
        # Check if mode transition is valid
        if mode == ScramjetMode.STARTUP and self.mode == ScramjetMode.SHUTDOWN:
            if self.current_mach < 0.5:
                return False  # Need minimum speed for startup
        
        # Set new mode
        self.mode = mode
        
        # Adjust parameters based on mode
        if mode == ScramjetMode.SHUTDOWN:
            self.throttle = 0.0
            self.fuel_flow = 0.0
        
        return True
        
    def update(self, flight_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Update scramjet state based on flight conditions."""
        # Extract relevant conditions
        self.current_mach = flight_conditions.get("mach", 0.0)
        altitude = flight_conditions.get("altitude", 0.0)
        dynamic_pressure = flight_conditions.get("dynamic_pressure", 0.0)
        
        # Check if within operational envelope
        if self.mode != ScramjetMode.SHUTDOWN and (
            self.current_mach < self.specs.min_mach or 
            self.current_mach > self.specs.max_mach
        ):
            self.set_mode(ScramjetMode.SHUTDOWN)
            return {"status": "shutdown", "reason": "outside_envelope"}
        
        # Determine appropriate mode based on Mach number
        if self.mode != ScramjetMode.SHUTDOWN and self.mode != ScramjetMode.STARTUP:
            if self.current_mach < 1.0:
                new_mode = ScramjetMode.SUBSONIC
            elif self.current_mach < 2.0:
                new_mode = ScramjetMode.TRANSONIC
            elif self.current_mach < 5.0:
                new_mode = ScramjetMode.SUPERSONIC
            else:
                new_mode = ScramjetMode.HYPERSONIC
                
            if new_mode != self.mode:
                self.set_mode(new_mode)
        
        # Calculate temperatures
        # Simple model: inlet temperature increases with Mach number
        self.inlet_temperature = 288.0 * (1 + 0.2 * self.current_mach**2)
        
        # Combustion temperature based on fuel flow and inlet temperature
        if self.mode != ScramjetMode.SHUTDOWN:
            self.combustion_temperature = self.inlet_temperature * (1 + self.throttle * 5.0)
            self.nozzle_temperature = self.combustion_temperature * 0.8
        
        # Calculate thrust
        thrust = self._calculate_thrust(self.current_mach, altitude, self.throttle)
        
        # Record performance data
        performance = {
            "mach": self.current_mach,
            "mode": self.mode.name,
            "thrust": thrust,
            "inlet_temp": self.inlet_temperature,
            "combustion_temp": self.combustion_temperature,
            "fuel_flow": self.fuel_flow
        }
        self.performance_history.append(performance)
        
        return performance
    
    def set_throttle(self, throttle: float) -> float:
        """Set throttle level and return actual throttle setting."""
        if self.mode == ScramjetMode.SHUTDOWN:
            self.throttle = 0.0
            self.fuel_flow = 0.0
            return 0.0
            
        # Limit throttle based on mode
        if self.mode == ScramjetMode.STARTUP:
            max_throttle = 0.3
        elif self.mode == ScramjetMode.SUBSONIC:
            max_throttle = 0.5
        else:
            max_throttle = 1.0
            
        self.throttle = max(0.0, min(throttle, max_throttle))
        
        # Calculate fuel flow based on throttle and Mach
        self.fuel_flow = self.throttle * (0.5 + 0.5 * self.current_mach / self.specs.max_mach)
        
        return self.throttle
    
    def _calculate_thrust(self, mach: float, altitude: float, throttle: float) -> float:
        """Calculate thrust based on flight conditions and throttle."""
        if self.mode == ScramjetMode.SHUTDOWN or mach < self.specs.min_mach:
            return 0.0
            
        # Basic thrust model
        # Thrust increases with Mach number up to a point, then decreases
        mach_factor = min(1.0, mach / 3.0) * (2.0 - min(1.0, mach / self.specs.max_mach))
        
        # Altitude effect (decreases with altitude due to air density)
        if altitude < 20000:
            altitude_factor = 1.0 - altitude / 40000
        else:
            altitude_factor = 0.5 * (1.0 - (altitude - 20000) / 60000)
        altitude_factor = max(0.1, altitude_factor)
        
        # Calculate thrust
        thrust = self.specs.max_thrust * throttle * mach_factor * altitude_factor
        
        return thrust
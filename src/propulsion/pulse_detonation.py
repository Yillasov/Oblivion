"""
Pulse Detonation Engine Controller for UCAV platforms.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.propulsion.base import PropulsionInterface, PropulsionSpecs, PropulsionType


class PDEMode(Enum):
    """Operating modes for pulse detonation engine."""
    STANDBY = 0
    STARTUP = 1
    LOW_FREQUENCY = 2
    HIGH_FREQUENCY = 3
    SHUTDOWN = 4


@dataclass
class PDESpecs:
    """Specifications for pulse detonation engine."""
    max_thrust: float  # Maximum thrust in kN
    specific_impulse: float  # Specific impulse in seconds
    min_frequency: float  # Minimum detonation frequency in Hz
    max_frequency: float  # Maximum detonation frequency in Hz
    chamber_volume: float  # Detonation chamber volume in m³
    weight: float  # Weight in kg
    fuel_type: str  # Type of fuel used


class PulseDetonationController:
    """Simple controller for pulse detonation engine."""
    
    def __init__(self, specs: PDESpecs):
        """Initialize pulse detonation engine controller."""
        self.specs = specs
        self.mode = PDEMode.STANDBY
        self.thrust_level = 0.0
        self.current_frequency = 0.0
        self.fuel_flow = 0.0
        self.chamber_temperature = 300.0  # K
        self.chamber_pressure = 101.3  # kPa
        self.performance_history: List[Dict[str, float]] = []
        self.cycle_count = 0
        
    def set_mode(self, mode: PDEMode) -> bool:
        """Set operating mode of PDE."""
        # Set new mode
        self.mode = mode
        
        # Adjust parameters based on mode
        if mode == PDEMode.STANDBY or mode == PDEMode.SHUTDOWN:
            self.thrust_level = 0.0
            self.current_frequency = 0.0
            self.fuel_flow = 0.0
        elif mode == PDEMode.STARTUP:
            self.thrust_level = 0.1
            self.current_frequency = self.specs.min_frequency
        elif mode == PDEMode.LOW_FREQUENCY:
            self.current_frequency = self.specs.min_frequency + 0.2 * (
                self.specs.max_frequency - self.specs.min_frequency)
        elif mode == PDEMode.HIGH_FREQUENCY:
            self.current_frequency = self.specs.max_frequency
        
        return True
        
    def update(self, flight_conditions: Dict[str, float], dt: float) -> Dict[str, Any]:
        """Update PDE state based on flight conditions."""
        # Extract relevant conditions
        altitude = flight_conditions.get("altitude", 0.0)
        mach = flight_conditions.get("mach", 0.0)
        
        # Update cycle count
        if self.mode != PDEMode.STANDBY and self.mode != PDEMode.SHUTDOWN:
            self.cycle_count += int(self.current_frequency * dt)
        
        # Calculate chamber conditions
        if self.mode != PDEMode.STANDBY and self.mode != PDEMode.SHUTDOWN:
            # Simple model for chamber temperature during detonation
            base_temp = 300.0 + altitude * (-0.0065)  # Approximate ambient temperature
            self.chamber_temperature = base_temp + 2000.0 * self.thrust_level
            
            # Simple model for chamber pressure during detonation
            ambient_pressure = 101.3 * np.exp(-altitude / 8000)  # Approximate ambient pressure
            self.chamber_pressure = ambient_pressure * (1 + 20.0 * self.thrust_level)
            
            # Calculate fuel flow
            self.fuel_flow = self.thrust_level * self.current_frequency / self.specs.max_frequency
        else:
            # Reset to ambient conditions when not operating
            self.chamber_temperature = 300.0 + altitude * (-0.0065)
            self.chamber_pressure = 101.3 * np.exp(-altitude / 8000)
            self.fuel_flow = 0.0
        
        # Calculate thrust
        thrust = self._calculate_thrust(altitude, mach)
        
        # Record performance data
        performance = {
            "mode": self.mode.name,
            "thrust": thrust,
            "frequency": self.current_frequency,
            "chamber_temp": self.chamber_temperature,
            "chamber_pressure": self.chamber_pressure,
            "fuel_flow": self.fuel_flow,
            "cycle_count": self.cycle_count
        }
        self.performance_history.append(performance)
        
        return performance
    
    def set_thrust(self, thrust_level: float) -> float:
        """Set thrust level and return actual thrust setting."""
        if self.mode == PDEMode.STANDBY or self.mode == PDEMode.SHUTDOWN:
            self.thrust_level = 0.0
            return 0.0
            
        # Limit thrust based on mode
        if self.mode == PDEMode.STARTUP:
            max_thrust = 0.3
        elif self.mode == PDEMode.LOW_FREQUENCY:
            max_thrust = 0.7
        else:
            max_thrust = 1.0
            
        self.thrust_level = max(0.0, min(thrust_level, max_thrust))
        
        # Adjust frequency based on thrust level
        if self.mode == PDEMode.LOW_FREQUENCY or self.mode == PDEMode.HIGH_FREQUENCY:
            freq_range = self.specs.max_frequency - self.specs.min_frequency
            self.current_frequency = self.specs.min_frequency + self.thrust_level * freq_range
            
            # Switch modes based on frequency
            if self.current_frequency < self.specs.min_frequency + 0.5 * freq_range:
                self.mode = PDEMode.LOW_FREQUENCY
            else:
                self.mode = PDEMode.HIGH_FREQUENCY
        
        return self.thrust_level
    
    def _calculate_thrust(self, altitude: float, mach: float) -> float:
        """Calculate thrust based on flight conditions and engine settings."""
        if self.mode == PDEMode.STANDBY or self.mode == PDEMode.SHUTDOWN:
            return 0.0
            
        # Basic thrust model
        # Thrust proportional to frequency and thrust level
        freq_factor = self.current_frequency / self.specs.max_frequency
        
        # Altitude effect (decreases with altitude due to air density)
        altitude_factor = np.exp(-altitude / 20000)
        
        # Mach effect (slight increase with mach due to ram effect)
        mach_factor = 1.0 + 0.1 * min(2.0, mach)
        
        # Calculate thrust
        thrust = self.specs.max_thrust * self.thrust_level * freq_factor * altitude_factor * mach_factor
        
        return thrust
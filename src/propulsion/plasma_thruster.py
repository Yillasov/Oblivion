"""
Plasma Thruster Management System for advanced space propulsion.
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
from src.propulsion.ion_thruster import IonThrusterController, IonThrusterSpecs


class PlasmaMode(Enum):
    """Operating modes for plasma thrusters."""
    OFF = 0
    STANDBY = 1
    PULSED = 2
    CONTINUOUS = 3
    HIGH_POWER = 4


@dataclass
class PlasmaThrusterSpecs(IonThrusterSpecs):
    """Specifications for plasma thruster systems."""
    plasma_density_range: Tuple[float, float]  # Plasma density range in particles/m³
    magnetic_field_strength: float  # Magnetic field strength in Tesla
    discharge_voltage: float  # Discharge voltage in V
    pulse_frequency_range: Tuple[float, float]  # Min/max pulse frequency in Hz


class PlasmaThrusterManager:
    """Simple plasma thruster management system."""
    
    def __init__(self, thruster: IonThrusterController, specs: PlasmaThrusterSpecs):
        """Initialize plasma thruster manager."""
        self.thruster = thruster
        self.specs = specs
        self.mode = PlasmaMode.OFF
        self.plasma_density = 0.0
        self.pulse_frequency = 0.0
        self.magnetic_field = specs.magnetic_field_strength
        self.discharge_current = 0.0
        
    def set_mode(self, mode: PlasmaMode) -> bool:
        """Set plasma thruster operating mode."""
        if mode == self.mode:
            return True
            
        # Map plasma mode to ion thruster mode
        if mode == PlasmaMode.OFF:
            self.thruster.set_mode(self.thruster.mode.__class__.OFF)
            self.pulse_frequency = 0.0
        elif mode == PlasmaMode.STANDBY:
            self.thruster.set_mode(self.thruster.mode.__class__.STANDBY)
            self.pulse_frequency = 0.0
        elif mode == PlasmaMode.PULSED:
            self.thruster.set_mode(self.thruster.mode.__class__.NORMAL)
            self.pulse_frequency = self.specs.pulse_frequency_range[0]
        elif mode == PlasmaMode.CONTINUOUS:
            self.thruster.set_mode(self.thruster.mode.__class__.NORMAL)
            self.pulse_frequency = 0.0  # Continuous mode
        elif mode == PlasmaMode.HIGH_POWER:
            self.thruster.set_mode(self.thruster.mode.__class__.HIGH_POWER)
            self.pulse_frequency = self.specs.pulse_frequency_range[1]
            
        self.mode = mode
        self._update_plasma_parameters()
        return True
    
    def set_power_level(self, power_level: float) -> bool:
        """Set power level for plasma thruster."""
        # Set power level on underlying ion thruster
        result = self.thruster.set_power_state({"power_level": power_level})
        self._update_plasma_parameters()
        return result
    
    def set_pulse_frequency(self, frequency: float) -> None:
        """Set pulse frequency for pulsed mode."""
        if self.mode != PlasmaMode.PULSED:
            return
            
        # Clamp to valid range
        self.pulse_frequency = max(self.specs.pulse_frequency_range[0],
                                 min(frequency, self.specs.pulse_frequency_range[1]))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of plasma thruster."""
        ion_status = self.thruster.get_status()
        
        return {
            "mode": self.mode.name,
            "power_level": ion_status["power_level"],
            "plasma_density": self.plasma_density,
            "magnetic_field": self.magnetic_field,
            "pulse_frequency": self.pulse_frequency,
            "discharge_current": self.discharge_current,
            "thrust_estimate": self._calculate_thrust(),
            "propellant_type": self.specs.propellant_type,
            "temperature": ion_status["temperature"]
        }
    
    def _update_plasma_parameters(self) -> None:
        """Update plasma parameters based on current state."""
        power_level = self.thruster.power_level
        
        # Calculate plasma density based on power level
        density_range = self.specs.plasma_density_range
        self.plasma_density = density_range[0] + power_level * (
            density_range[1] - density_range[0])
            
        # Calculate discharge current based on voltage and power
        power = power_level * self.specs.max_power * 1000  # Convert to W
        self.discharge_current = power / self.specs.discharge_voltage
    
    def _calculate_thrust(self) -> float:
        """Calculate thrust based on plasma parameters."""
        base_thrust = self.thruster._estimate_thrust()
        
        # Apply plasma-specific modifications
        if self.mode == PlasmaMode.PULSED:
            # Pulsed mode has different thrust characteristics
            pulse_factor = 0.8 + 0.4 * (
                self.pulse_frequency - self.specs.pulse_frequency_range[0]
            ) / (self.specs.pulse_frequency_range[1] - self.specs.pulse_frequency_range[0])
            return base_thrust * pulse_factor
        elif self.mode == PlasmaMode.HIGH_POWER:
            # High power mode has enhanced thrust
            return base_thrust * 1.2
            
        return base_thrust
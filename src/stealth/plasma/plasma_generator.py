#!/usr/bin/env python3
"""
Plasma generation and control systems for stealth applications.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.stealth.plasma.plasma_system import PlasmaParameters, PlasmaStealthSystem


class PlasmaGenerationMethod(Enum):
    """Methods for generating plasma."""
    RF_DISCHARGE = "rf_discharge"
    MICROWAVE = "microwave"
    LASER_INDUCED = "laser_induced"
    ELECTRIC_DISCHARGE = "electric_discharge"
    MAGNETRON = "magnetron"


class PlasmaPulsePattern(Enum):
    """Pulse patterns for plasma generation."""
    CONTINUOUS = "continuous"
    REGULAR_PULSE = "regular_pulse"
    VARIABLE_PULSE = "variable_pulse"
    MODULATED = "modulated"
    ADAPTIVE = "adaptive"


@dataclass
class PlasmaGeneratorSpecs:
    """Specifications for plasma generator."""
    max_power_kw: float
    max_density: float  # particles/cm³
    generation_method: PlasmaGenerationMethod
    frequency_range: Tuple[float, float]  # GHz
    pulse_capabilities: List[PlasmaPulsePattern]
    startup_time_ms: float
    cooldown_time_ms: float
    weight_kg: float
    dimensions_cm: Tuple[float, float, float]  # length, width, height


class PlasmaGenerator:
    """Plasma generator for stealth applications."""
    
    def __init__(self, specs: PlasmaGeneratorSpecs):
        """
        Initialize plasma generator.
        
        Args:
            specs: Generator specifications
        """
        self.specs = specs
        self.current_power = 0.0
        self.current_frequency = specs.frequency_range[0]
        self.current_pulse_pattern = PlasmaPulsePattern.CONTINUOUS
        self.pulse_frequency = 0.0
        self.pulse_duty_cycle = 1.0  # 1.0 = continuous
        self.status = {
            "active": False,
            "power_level": 0.0,
            "frequency": self.current_frequency,
            "pulse_pattern": self.current_pulse_pattern.value,
            "pulse_frequency": self.pulse_frequency,
            "pulse_duty_cycle": self.pulse_duty_cycle,
            "temperature": 20.0,  # °C
            "plasma_density": 0.0,
            "plasma_stability": 1.0,  # 0-1 scale
            "error_code": None
        }
    
    def power_on(self) -> bool:
        """
        Power on the plasma generator.
        
        Returns:
            Success status
        """
        if self.status["active"]:
            return True
            
        self.status["active"] = True
        self.status["power_level"] = 0.1  # Start at 10% power
        self.status["plasma_density"] = self._calculate_plasma_density()
        return True
    
    def power_off(self) -> bool:
        """
        Power off the plasma generator.
        
        Returns:
            Success status
        """
        if not self.status["active"]:
            return True
            
        self.status["active"] = False
        self.status["power_level"] = 0.0
        self.status["plasma_density"] = 0.0
        return True
    
    def set_power_level(self, power_level: float) -> bool:
        """
        Set power level of the plasma generator.
        
        Args:
            power_level: Power level (0.0-1.0)
            
        Returns:
            Success status
        """
        if not self.status["active"]:
            return False
            
        if power_level < 0.0 or power_level > 1.0:
            return False
            
        self.status["power_level"] = power_level
        self.status["plasma_density"] = self._calculate_plasma_density()
        self.status["temperature"] = 20.0 + (power_level * 80.0)  # Simple temperature model
        return True
    
    def set_frequency(self, frequency_ghz: float) -> bool:
        """
        Set operating frequency.
        
        Args:
            frequency_ghz: Operating frequency in GHz
            
        Returns:
            Success status
        """
        min_freq, max_freq = self.specs.frequency_range
        if frequency_ghz < min_freq or frequency_ghz > max_freq:
            return False
            
        self.current_frequency = frequency_ghz
        self.status["frequency"] = frequency_ghz
        return True
    
    def set_pulse_pattern(self, pattern: PlasmaPulsePattern, 
                         frequency_hz: float = 1000.0,
                         duty_cycle: float = 0.5) -> bool:
        """
        Set pulse pattern for plasma generation.
        
        Args:
            pattern: Pulse pattern
            frequency_hz: Pulse frequency in Hz
            duty_cycle: Duty cycle (0.0-1.0)
            
        Returns:
            Success status
        """
        if pattern not in self.specs.pulse_capabilities:
            return False
            
        self.current_pulse_pattern = pattern
        self.pulse_frequency = frequency_hz
        self.pulse_duty_cycle = max(0.1, min(1.0, duty_cycle))
        
        self.status["pulse_pattern"] = pattern.value
        self.status["pulse_frequency"] = frequency_hz
        self.status["pulse_duty_cycle"] = self.pulse_duty_cycle
        
        # Adjust plasma density based on pulse pattern
        if pattern == PlasmaPulsePattern.CONTINUOUS:
            # No change for continuous operation
            pass
        else:
            # Pulsed operation affects average plasma density
            self.status["plasma_density"] = self._calculate_plasma_density() * self.pulse_duty_cycle
            
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the plasma generator.
        
        Returns:
            Status dictionary
        """
        return self.status
    
    def _calculate_plasma_density(self) -> float:
        """
        Calculate plasma density based on current settings.
        
        Returns:
            Plasma density in particles/cm³
        """
        if not self.status["active"]:
            return 0.0
            
        # Base density is proportional to power level
        base_density = self.specs.max_density * self.status["power_level"]
        
        # Different generation methods have different efficiency
        method_efficiency = {
            PlasmaGenerationMethod.RF_DISCHARGE: 0.8,
            PlasmaGenerationMethod.MICROWAVE: 0.9,
            PlasmaGenerationMethod.LASER_INDUCED: 0.7,
            PlasmaGenerationMethod.ELECTRIC_DISCHARGE: 0.85,
            PlasmaGenerationMethod.MAGNETRON: 0.95
        }
        
        efficiency = method_efficiency.get(self.specs.generation_method, 0.8)
        
        return base_density * efficiency


class PlasmaControlSystem:
    """Control system for plasma-based stealth."""
    
    def __init__(self, plasma_system: PlasmaStealthSystem, plasma_generator: PlasmaGenerator):
        """
        Initialize plasma control system.
        
        Args:
            plasma_system: Plasma stealth system
            plasma_generator: Plasma generator
        """
        self.plasma_system = plasma_system
        self.plasma_generator = plasma_generator
        self.adaptive_mode = False
        self.threat_response_active = False
        self.current_threat_frequency = None
        
    def initialize(self) -> bool:
        """
        Initialize the plasma control system.
        
        Returns:
            Success status
        """
        return self.plasma_system.initialize()
    
    def activate(self, power_level: float = 0.8, 
                pulse_mode: bool = False,
                pulse_frequency: float = 1000.0) -> bool:
        """
        Activate the plasma stealth system.
        
        Args:
            power_level: Power level (0.0-1.0)
            pulse_mode: Enable pulse mode
            pulse_frequency: Pulse frequency in Hz
            
        Returns:
            Success status
        """
        # First activate the generator
        if not self.plasma_generator.power_on():
            return False
            
        # Set generator parameters
        self.plasma_generator.set_power_level(power_level)
        
        if pulse_mode:
            self.plasma_generator.set_pulse_pattern(
                PlasmaPulsePattern.REGULAR_PULSE,
                pulse_frequency,
                0.5  # 50% duty cycle
            )
        else:
            self.plasma_generator.set_pulse_pattern(PlasmaPulsePattern.CONTINUOUS)
        
        # Then activate the stealth system
        activation_params = {
            "power_level": power_level,
            "pulse_mode": pulse_mode,
            "pulse_frequency": pulse_frequency
        }
        
        return self.plasma_system.activate(activation_params)
    
    def deactivate(self) -> bool:
        """
        Deactivate the plasma stealth system.
        
        Returns:
            Success status
        """
        # First deactivate the stealth system
        stealth_result = self.plasma_system.deactivate()
        
        # Then power off the generator
        generator_result = self.plasma_generator.power_off()
        
        return stealth_result and generator_result
    
    def adjust_for_threat(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust plasma parameters to counter a specific threat.
        
        Args:
            threat_data: Information about the threat
            
        Returns:
            Adjustment results
        """
        if not self.plasma_system.status["active"]:
            return {"success": False, "error": "System not active"}
            
        # Extract threat frequency
        threat_frequency = threat_data.get("frequency", 10.0)  # GHz
        
        # Store current threat frequency
        self.current_threat_frequency = threat_frequency
        self.threat_response_active = True
        
        # Adjust plasma frequency to counter the threat
        # For optimal absorption, plasma frequency should be close to threat frequency
        min_freq, max_freq = self.plasma_generator.specs.frequency_range
        target_frequency = min(max(threat_frequency, min_freq), max_freq)
        
        # Set generator frequency
        self.plasma_generator.set_frequency(target_frequency)
        
        # Adjust power level based on threat intensity
        threat_power = threat_data.get("power", 1000.0)  # W
        required_power_level = min(0.5 + (threat_power / 10000.0), 1.0)
        self.plasma_generator.set_power_level(required_power_level)
        
        # Update plasma system parameters
        self.plasma_system.adjust_parameters({
            "power_level": required_power_level,
            "frequency": target_frequency
        })
        
        # Calculate expected effectiveness
        env_conditions = {"temperature": 20.0, "altitude": 5000.0}
        effectiveness = self.plasma_system.calculate_effectiveness(threat_data, env_conditions)
        
        return {
            "success": True,
            "adjusted_frequency": target_frequency,
            "adjusted_power": required_power_level,
            "expected_effectiveness": effectiveness
        }
    
    def set_adaptive_mode(self, enabled: bool) -> bool:
        """
        Enable or disable adaptive mode.
        
        Args:
            enabled: Enable adaptive mode
            
        Returns:
            Success status
        """
        self.adaptive_mode = enabled
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get combined status of plasma control system.
        
        Returns:
            Status dictionary
        """
        generator_status = self.plasma_generator.get_status()
        system_status = self.plasma_system.get_status()
        
        return {
            "stealth_system": system_status,
            "plasma_generator": generator_status,
            "control_system": {
                "adaptive_mode": self.adaptive_mode,
                "threat_response_active": self.threat_response_active,
                "current_threat_frequency": self.current_threat_frequency
            }
        }
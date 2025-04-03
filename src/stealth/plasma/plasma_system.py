#!/usr/bin/env python3
"""
Plasma Stealth system implementation for Oblivion SDK.
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

from src.stealth.base.interfaces import NeuromorphicStealth, StealthSpecs, StealthType
from src.stealth.base.config import StealthSystemConfig, StealthOperationalMode


@dataclass
class PlasmaParameters:
    """Parameters for plasma stealth operation."""
    density: float  # Plasma density in particles/cm³
    temperature: float  # Plasma temperature in eV
    frequency: float  # Operating frequency in GHz
    power_level: float  # Power level (0.0-1.0)
    pulse_mode: bool  # Pulsed or continuous
    pulse_frequency: float  # Pulse frequency in Hz (if in pulse mode)


class PlasmaStealthSystem(NeuromorphicStealth):
    """Plasma Stealth system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """
        Initialize Plasma Stealth system.
        
        Args:
            config: System configuration
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.config = config
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.PLASMA_STEALTH,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=0.2,  # Plasma can achieve very low RCS
            infrared_signature=1.5,  # Plasma generates heat
            acoustic_signature=1.2,  # Plasma generators produce some noise
            activation_time=config.activation_time_seconds,
            operational_duration=config.operational_duration_minutes,
            cooldown_period=config.cooldown_time_seconds / 60.0,  # Convert to minutes
            frequency_ranges=[{"min": 0.5, "max": 35.0}]  # Effective against wide frequency range
        )
        
        # Plasma system specific parameters
        self.plasma_params = PlasmaParameters(
            density=1.0e12,  # Default plasma density
            temperature=10.0,  # Default plasma temperature in eV
            frequency=10.0,   # Default operating frequency in GHz
            power_level=0.0,  # Initial power level
            pulse_mode=False, # Continuous mode by default
            pulse_frequency=0.0  # No pulsing initially
        )
        
        # System status
        self.status = {
            "active": False,
            "mode": "standby",
            "power_level": 0.0,
            "temperature": 20.0,
            "remaining_operation_time": config.operational_duration_minutes,
            "cooldown_remaining": 0.0,
            "plasma_density": 0.0,
            "plasma_temperature": 0.0,
            "frequency_coverage": {"min": 0.5, "max": 35.0}
        }
        
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the plasma stealth system."""
        self.initialized = True
        self.status["mode"] = "standby"
        return True
    
    def get_specifications(self) -> StealthSpecs:
        """Get the physical specifications of the stealth system."""
        return self.specs
    
    def calculate_effectiveness(self, 
                              threat_data: Dict[str, Any],
                              environmental_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate plasma stealth effectiveness against specific threats under given conditions.
        
        Args:
            threat_data: Information about the threat (radar type, frequency, etc.)
            environmental_conditions: Environmental conditions (temperature, humidity, etc.)
            
        Returns:
            Dictionary of effectiveness metrics
        """
        if not self.status["active"]:
            return {"rcs_reduction": 0.0, "detection_probability": 1.0}
        
        # Extract threat information
        radar_frequency = threat_data.get("frequency", 10.0)  # Default to 10 GHz
        radar_power = threat_data.get("power", 1000.0)  # Default to 1000W
        
        # Extract environmental conditions
        ambient_temperature = environmental_conditions.get("temperature", 20.0)  # °C
        altitude = environmental_conditions.get("altitude", 0.0)  # m
        
        # Calculate plasma frequency based on density
        # Formula: fp = 8.98 * sqrt(ne) Hz, where ne is electron density in cm^-3
        plasma_density = self.plasma_params.density * self.status["power_level"]
        plasma_frequency = 8.98 * np.sqrt(plasma_density) / 1.0e9  # Convert to GHz
        
        # Calculate effectiveness based on plasma frequency vs radar frequency
        if plasma_frequency > radar_frequency:
            # Plasma frequency higher than radar frequency - very effective
            base_effectiveness = 0.95
        else:
            # Plasma frequency lower than radar frequency - less effective
            ratio = plasma_frequency / radar_frequency
            base_effectiveness = 0.4 + (0.5 * ratio)
        
        # Adjust for power level
        power_factor = self.status["power_level"]
        
        # Adjust for altitude (plasma is more effective at higher altitudes)
        altitude_factor = 1.0
        if altitude > 10000:
            altitude_factor = 1.2
        elif altitude > 5000:
            altitude_factor = 1.1
        
        # Adjust for temperature (extreme temperatures reduce effectiveness)
        temp_diff = abs(ambient_temperature - 20.0)
        temp_factor = 1.0 - (0.005 * min(temp_diff, 40.0))
        
        # Calculate final effectiveness
        effectiveness = base_effectiveness * power_factor * altitude_factor * temp_factor
        
        # Calculate RCS reduction and detection probability
        rcs_reduction = min(effectiveness, 0.98)  # Cap at 98% reduction
        detection_probability = 1.0 - (rcs_reduction * 0.9)  # Even perfect RCS reduction has some detection probability
        
        return {
            "rcs_reduction": rcs_reduction,
            "detection_probability": detection_probability,
            "plasma_frequency_ghz": plasma_frequency,
            "effectiveness_factor": effectiveness
        }
    
    def activate(self, activation_params: Dict[str, Any] = {}) -> bool:
        """
        Activate the plasma stealth system.
        
        Args:
            activation_params: Parameters for activation
            
        Returns:
            Success status
        """
        if not self.initialized:
            return False
            
        if self.status["cooldown_remaining"] > 0:
            return False  # Still in cooldown
            
        # Set default parameters if none provided
        if activation_params is None:
            activation_params = {}
            
        # Extract activation parameters
        power_level = activation_params.get("power_level", 0.8)
        pulse_mode = activation_params.get("pulse_mode", False)
        pulse_frequency = activation_params.get("pulse_frequency", 1000.0)  # Hz
        
        # Update plasma parameters
        self.plasma_params.power_level = power_level
        self.plasma_params.pulse_mode = pulse_mode
        self.plasma_params.pulse_frequency = pulse_frequency
        
        # Update system status
        self.status["active"] = True
        self.status["mode"] = "active"
        self.status["power_level"] = power_level
        self.status["plasma_density"] = self.plasma_params.density * power_level
        self.status["plasma_temperature"] = self.plasma_params.temperature * power_level
        
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the plasma stealth system.
        
        Returns:
            Success status
        """
        if not self.initialized or not self.status["active"]:
            return False
            
        # Update system status
        self.status["active"] = False
        self.status["mode"] = "standby"
        self.status["power_level"] = 0.0
        self.status["plasma_density"] = 0.0
        self.status["plasma_temperature"] = 0.0
        self.status["cooldown_remaining"] = self.specs.cooldown_period
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Adjust operational parameters of the plasma stealth system.
        
        Args:
            parameters: New parameters to set
            
        Returns:
            Success status
        """
        if not self.initialized:
            return False
            
        if "power_level" in parameters:
            power_level = parameters["power_level"]
            if 0.0 <= power_level <= 1.0:
                self.status["power_level"] = power_level
                self.plasma_params.power_level = power_level
                self.status["plasma_density"] = self.plasma_params.density * power_level
                self.status["plasma_temperature"] = self.plasma_params.temperature * power_level
                
        if "pulse_mode" in parameters:
            self.plasma_params.pulse_mode = parameters["pulse_mode"]
            
        if "pulse_frequency" in parameters:
            self.plasma_params.pulse_frequency = parameters["pulse_frequency"]
            
        if "frequency" in parameters:
            self.plasma_params.frequency = parameters["frequency"]
            
        return True
    
    def update_system(self, time_delta: float) -> None:
        """
        Update system status based on time elapsed.
        
        Args:
            time_delta: Time elapsed in seconds
        """
        if not self.initialized:
            return
            
        if self.status["active"]:
            # Convert time_delta to minutes
            time_delta_min = time_delta / 60.0
            
            # Update remaining operation time
            self.status["remaining_operation_time"] -= time_delta_min
            
            # Check if operation time has expired
            if self.status["remaining_operation_time"] <= 0:
                self.deactivate()
                self.status["remaining_operation_time"] = 0.0
                
            # Update temperature based on power level
            base_temp = 20.0
            max_temp_increase = 80.0
            self.status["temperature"] = base_temp + (max_temp_increase * self.status["power_level"])
                
        elif self.status["cooldown_remaining"] > 0:
            # Update cooldown time
            time_delta_min = time_delta / 60.0
            self.status["cooldown_remaining"] -= time_delta_min
            
            if self.status["cooldown_remaining"] <= 0:
                self.status["cooldown_remaining"] = 0.0
                self.status["remaining_operation_time"] = self.config.operational_duration_minutes
                
            # Temperature decreases during cooldown
            cooldown_progress = 1.0 - (self.status["cooldown_remaining"] / self.specs.cooldown_period)
            self.status["temperature"] = 20.0 + (80.0 * (1.0 - cooldown_progress))
#!/usr/bin/env python3
"""
Acoustic Signature Reduction system implementation for Oblivion SDK.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.stealth.base.interfaces import NeuromorphicStealth, StealthSpecs, StealthType
from src.stealth.base.config import StealthSystemConfig, StealthOperationalMode


@dataclass
class AcousticParameters:
    """Parameters for acoustic signature reduction."""
    frequency_range: Dict[str, float]  # Frequency range in Hz
    max_attenuation: float  # Maximum attenuation in dB
    response_time: float  # Response time in seconds
    coverage_area: float  # Coverage area in m²
    active_damping_capability: float  # Active damping capability (0.0-1.0)
    passive_damping_capability: float  # Passive damping capability (0.0-1.0)


class AcousticReductionSystem(NeuromorphicStealth):
    """Acoustic Signature Reduction system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """
        Initialize Acoustic Signature Reduction system.
        
        Args:
            config: System configuration
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.config = config
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.ACOUSTIC_REDUCTION,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=1.0,  # No effect on RCS
            infrared_signature=1.0,  # No effect on IR
            acoustic_signature=0.25,  # Significant reduction in acoustic signature
            activation_time=config.activation_time_seconds,
            operational_duration=config.operational_duration_minutes,
            cooldown_period=config.cooldown_time_seconds / 60.0  # Convert to minutes
        )
        
        # Acoustic reduction specific parameters
        self.acoustic_params = AcousticParameters(
            frequency_range={"min": 20.0, "max": 20000.0},  # 20Hz to 20kHz (human hearing range)
            max_attenuation=35.0,  # 35 dB maximum attenuation
            response_time=0.05,  # 50ms response time
            coverage_area=25.0,  # 25 m² coverage
            active_damping_capability=0.85,  # 85% active damping capability
            passive_damping_capability=0.60   # 60% passive damping capability
        )
        
        # System status
        self.status = {
            "active": False,
            "mode": "standby",
            "power_level": 0.0,
            "current_attenuation": 0.0,  # Current attenuation in dB
            "target_frequencies": [],  # Target frequencies for active cancellation
            "remaining_operation_time": config.operational_duration_minutes,
            "cooldown_remaining": 0.0,
            "effectiveness": 0.0,
            "active_damping_enabled": False,
            "passive_damping_enabled": True
        }
        
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the acoustic reduction system."""
        self.initialized = True
        self.status["mode"] = "standby"
        self.status["passive_damping_enabled"] = True  # Passive damping is always on
        return True
    
    def get_specifications(self) -> StealthSpecs:
        """Get the physical specifications of the stealth system."""
        return self.specs
    
    def calculate_effectiveness(self, 
                              threat_data: Dict[str, Any],
                              environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate acoustic reduction effectiveness against specific threats.
        
        Args:
            threat_data: Information about the threat
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of effectiveness metrics
        """
        # Base passive effectiveness even when not active
        if not self.status["active"] and self.status["passive_damping_enabled"]:
            return {
                "acoustic_reduction": self.acoustic_params.passive_damping_capability * 0.8,
                "detection_probability": 0.7,
                "mode": "passive"
            }
        elif not self.status["active"]:
            return {"acoustic_reduction": 0.0, "detection_probability": 1.0, "mode": "inactive"}
        
        # Extract threat information
        threat_type = threat_data.get("type", "acoustic")
        threat_distance = threat_data.get("distance", 500.0)  # meters
        sensor_sensitivity = threat_data.get("sensitivity", 0.5)  # 0.0-1.0
        threat_frequencies = threat_data.get("frequencies", [100.0, 500.0, 1000.0])  # Hz
        
        # Extract environmental conditions
        wind_speed = environmental_conditions.get("wind_speed", 0.0)  # m/s
        precipitation = environmental_conditions.get("precipitation", 0.0)  # 0.0-1.0
        temperature = environmental_conditions.get("temperature", 20.0)  # °C
        
        # Calculate base effectiveness based on power level and mode
        power_factor = self.status["power_level"]
        
        # Calculate passive damping effectiveness
        passive_effectiveness = self.acoustic_params.passive_damping_capability
        
        # Calculate active damping effectiveness if enabled
        active_effectiveness = 0.0
        if self.status["active_damping_enabled"]:
            # Check how many threat frequencies are within our target range
            matching_frequencies = 0
            for freq in threat_frequencies:
                if freq in self.status["target_frequencies"] or not self.status["target_frequencies"]:
                    matching_frequencies += 1
            
            # Calculate frequency match factor
            if threat_frequencies:
                frequency_match = matching_frequencies / len(threat_frequencies)
            else:
                frequency_match = 1.0
                
            # Calculate active effectiveness
            active_effectiveness = self.acoustic_params.active_damping_capability * power_factor * frequency_match
        
        # Combined effectiveness (active + passive)
        combined_effectiveness = passive_effectiveness + (active_effectiveness * (1.0 - passive_effectiveness))
        
        # Environmental adjustments
        # Wind reduces effectiveness
        wind_factor = max(0.7, 1.0 - (wind_speed / 30.0))
        
        # Precipitation affects sound propagation
        precipitation_factor = 1.0 + (precipitation * 0.2)
        
        # Distance factor (acoustic detection decreases with distance)
        distance_factor = min(1.0 + (threat_distance / 1000.0), 1.5)
        
        # Calculate final effectiveness
        if threat_type == "acoustic" or threat_type == "sonar":
            type_factor = 1.0  # Full effectiveness against acoustic threats
        else:
            type_factor = 0.1  # Minimal effectiveness against other threats
            
        effectiveness = min(combined_effectiveness * wind_factor * precipitation_factor * 
                          distance_factor * type_factor, 0.95)
        
        # Calculate detection probability based on sensor sensitivity
        detection_probability = max(0.05, (1.0 - effectiveness) * sensor_sensitivity)
        
        # Update status
        self.status["effectiveness"] = effectiveness
        self.status["current_attenuation"] = effectiveness * self.acoustic_params.max_attenuation
        
        return {
            "acoustic_reduction": effectiveness,
            "attenuation_db": self.status["current_attenuation"],
            "detection_probability": detection_probability,
            "mode": "active" if self.status["active_damping_enabled"] else "passive",
            "effectiveness_factors": {
                "passive": passive_effectiveness,
                "active": active_effectiveness,
                "wind": wind_factor,
                "precipitation": precipitation_factor,
                "distance": distance_factor,
                "threat_type": type_factor
            }
        }
    
    def activate(self, activation_params: Dict[str, Any] = {}) -> bool:
        """
        Activate the acoustic reduction system.
        
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
        active_damping = activation_params.get("active_damping", True)
        target_frequencies = activation_params.get("target_frequencies", [])
        
        # Update system status
        self.status["active"] = True
        self.status["mode"] = "active"
        self.status["power_level"] = power_level
        self.status["active_damping_enabled"] = active_damping
        
        if target_frequencies:
            self.status["target_frequencies"] = target_frequencies
        else:
            # Default to broad spectrum noise reduction
            self.status["target_frequencies"] = [100.0, 500.0, 1000.0, 2000.0, 5000.0]
        
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the acoustic reduction system.
        
        Returns:
            Success status
        """
        if not self.initialized or not self.status["active"]:
            return False
            
        # Update system status
        self.status["active"] = False
        self.status["mode"] = "standby"
        self.status["power_level"] = 0.0
        self.status["active_damping_enabled"] = False
        self.status["current_attenuation"] = self.acoustic_params.passive_damping_capability * self.acoustic_params.max_attenuation
        self.status["effectiveness"] = self.acoustic_params.passive_damping_capability
        self.status["cooldown_remaining"] = self.specs.cooldown_period
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Adjust operational parameters of the acoustic reduction system.
        
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
                
        if "active_damping" in parameters:
            self.status["active_damping_enabled"] = bool(parameters["active_damping"])
                
        if "target_frequencies" in parameters:
            target_freqs = parameters["target_frequencies"]
            if isinstance(target_freqs, list):
                # Filter frequencies to ensure they're within our range
                min_freq = self.acoustic_params.frequency_range["min"]
                max_freq = self.acoustic_params.frequency_range["max"]
                self.status["target_frequencies"] = [
                    f for f in target_freqs if min_freq <= f <= max_freq
                ]
                
        if "passive_damping" in parameters:
            self.status["passive_damping_enabled"] = bool(parameters["passive_damping"])
                
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
                
        elif self.status["cooldown_remaining"] > 0:
            # Update cooldown time
            time_delta_min = time_delta / 60.0
            self.status["cooldown_remaining"] -= time_delta_min
            
            if self.status["cooldown_remaining"] <= 0:
                self.status["cooldown_remaining"] = 0.0
                self.status["remaining_operation_time"] = self.config.operational_duration_minutes
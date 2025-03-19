"""
Adaptive Thermal Camouflage system implementation for Oblivion SDK.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.stealth.base.interfaces import NeuromorphicStealth, StealthSpecs, StealthType
from src.stealth.base.config import StealthSystemConfig, StealthOperationalMode


@dataclass
class ThermalCamouflageParameters:
    """Parameters for thermal camouflage operation."""
    temperature_range: Dict[str, float]  # Temperature range in °C
    adaptation_speed: float  # Speed of adaptation (0.0-1.0)
    power_efficiency: float  # Power efficiency (0.0-1.0)
    coverage_area: float  # Coverage area in m²
    emissivity_control: float  # Emissivity control precision (0.0-1.0)


class AdaptiveThermalCamouflage(NeuromorphicStealth):
    """Adaptive Thermal Camouflage system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """
        Initialize Adaptive Thermal Camouflage system.
        
        Args:
            config: System configuration
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.config = config
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.THERMAL_CAMOUFLAGE,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=1.0,  # No effect on RCS
            infrared_signature=0.15,  # Significant reduction in IR signature
            acoustic_signature=1.0,  # No effect on acoustics
            activation_time=config.activation_time_seconds,
            operational_duration=config.operational_duration_minutes,
            cooldown_period=config.cooldown_time_seconds / 60.0  # Convert to minutes
        )
        
        # Thermal camouflage specific parameters
        self.thermal_params = ThermalCamouflageParameters(
            temperature_range={"min": -20.0, "max": 80.0},  # Operating temperature range
            adaptation_speed=0.8,  # Fast adaptation
            power_efficiency=0.7,  # Good power efficiency
            coverage_area=20.0,  # 20 m² coverage
            emissivity_control=0.9  # High precision emissivity control
        )
        
        # System status
        self.status = {
            "active": False,
            "mode": "standby",
            "power_level": 0.0,
            "current_temperature": 20.0,  # Current surface temperature in °C
            "target_temperature": 20.0,  # Target temperature in °C
            "ambient_temperature": 20.0,  # Ambient temperature in °C
            "adaptation_progress": 0.0,  # Adaptation progress (0.0-1.0)
            "remaining_operation_time": config.operational_duration_minutes,
            "cooldown_remaining": 0.0,
            "current_emissivity": 0.9,  # Current surface emissivity
            "target_emissivity": 0.9,  # Target surface emissivity
            "thermal_profile": "neutral"  # Current thermal profile
        }
        
        self.initialized = False
        self.thermal_profiles = {
            "neutral": {"temperature": 20.0, "emissivity": 0.9},
            "cold": {"temperature": -10.0, "emissivity": 0.3},
            "hot": {"temperature": 60.0, "emissivity": 0.7},
            "ambient_match": {"temperature": 20.0, "emissivity": 0.5},
            "low_emission": {"temperature": 20.0, "emissivity": 0.1}
        }
    
    def initialize(self) -> bool:
        """Initialize the thermal camouflage system."""
        self.initialized = True
        self.status["mode"] = "standby"
        return True
    
    def get_specifications(self) -> StealthSpecs:
        """Get the physical specifications of the stealth system."""
        return self.specs
    
    def calculate_effectiveness(self, 
                              threat_data: Dict[str, Any],
                              environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate thermal camouflage effectiveness against specific threats.
        
        Args:
            threat_data: Information about the threat
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of effectiveness metrics
        """
        if not self.status["active"]:
            return {"thermal_reduction": 0.0, "detection_probability": 1.0}
        
        # Extract threat information
        threat_type = threat_data.get("type", "infrared")
        threat_distance = threat_data.get("distance", 1000.0)  # meters
        sensor_sensitivity = threat_data.get("sensitivity", 0.5)  # 0.0-1.0
        
        # Extract environmental conditions
        ambient_temp = environmental_conditions.get("temperature", 20.0)  # °C
        humidity = environmental_conditions.get("humidity", 50.0)  # %
        precipitation = environmental_conditions.get("precipitation", 0.0)  # 0.0-1.0
        
        # Update ambient temperature
        self.status["ambient_temperature"] = ambient_temp
        
        # Calculate temperature difference between surface and ambient
        temp_diff = abs(self.status["current_temperature"] - ambient_temp)
        
        # Calculate base effectiveness based on temperature match and emissivity
        temp_match_factor = max(0.0, 1.0 - (temp_diff / 40.0))  # Normalize to 40°C max difference
        emissivity_factor = 1.0 - self.status["current_emissivity"]  # Lower emissivity is better
        
        # Adjust for power level
        power_factor = self.status["power_level"]
        
        # Adjust for adaptation progress
        adaptation_factor = self.status["adaptation_progress"]
        
        # Adjust for distance (more effective at greater distances)
        distance_factor = min(1.0 + (threat_distance / 2000.0), 1.5)
        
        # Adjust for environmental conditions
        humidity_factor = 1.0 - (abs(humidity - 50.0) / 100.0)  # Optimal at 50% humidity
        
        # Calculate final effectiveness
        if threat_type == "infrared" or threat_type == "thermal":
            type_factor = 1.0  # Full effectiveness against IR threats
        else:
            type_factor = 0.1  # Minimal effectiveness against other threats
            
        base_effectiveness = (temp_match_factor * 0.6 + emissivity_factor * 0.4) * power_factor * adaptation_factor
        effectiveness = min(base_effectiveness * distance_factor * humidity_factor * type_factor, 0.95)
        
        # Calculate detection probability based on sensor sensitivity
        detection_probability = max(0.05, (1.0 - effectiveness) * sensor_sensitivity)
        
        return {
            "thermal_reduction": effectiveness,
            "detection_probability": detection_probability,
            "temperature_match": temp_match_factor,
            "emissivity_effectiveness": emissivity_factor,
            "adaptation_level": adaptation_factor,
            "effectiveness_factors": {
                "temperature": temp_match_factor,
                "emissivity": emissivity_factor,
                "power": power_factor,
                "adaptation": adaptation_factor,
                "distance": distance_factor,
                "humidity": humidity_factor,
                "threat_type": type_factor
            }
        }
    
    def activate(self, activation_params: Dict[str, Any] = {}) -> bool:
        """
        Activate the thermal camouflage system.
        
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
        profile = activation_params.get("profile", "ambient_match")
        
        # Get profile settings
        if profile in self.thermal_profiles:
            profile_settings = self.thermal_profiles[profile]
            target_temp = profile_settings["temperature"]
            target_emissivity = profile_settings["emissivity"]
        else:
            # Default to ambient match
            profile = "ambient_match"
            target_temp = self.status["ambient_temperature"]
            target_emissivity = 0.5
        
        # Update system status
        self.status["active"] = True
        self.status["mode"] = "active"
        self.status["power_level"] = power_level
        self.status["target_temperature"] = target_temp
        self.status["target_emissivity"] = target_emissivity
        self.status["thermal_profile"] = profile
        self.status["adaptation_progress"] = 0.2  # Initial adaptation progress
        
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the thermal camouflage system.
        
        Returns:
            Success status
        """
        if not self.initialized or not self.status["active"]:
            return False
            
        # Update system status
        self.status["active"] = False
        self.status["mode"] = "standby"
        self.status["power_level"] = 0.0
        self.status["adaptation_progress"] = 0.0
        self.status["cooldown_remaining"] = self.specs.cooldown_period
        
        # Reset to neutral thermal profile
        self.status["thermal_profile"] = "neutral"
        self.status["target_temperature"] = self.thermal_profiles["neutral"]["temperature"]
        self.status["target_emissivity"] = self.thermal_profiles["neutral"]["emissivity"]
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Adjust operational parameters of the thermal camouflage system.
        
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
                
        if "profile" in parameters:
            profile = parameters["profile"]
            if profile in self.thermal_profiles:
                self.status["thermal_profile"] = profile
                self.status["target_temperature"] = self.thermal_profiles[profile]["temperature"]
                self.status["target_emissivity"] = self.thermal_profiles[profile]["emissivity"]
                # Reset adaptation progress when profile changes
                self.status["adaptation_progress"] = 0.2
                
        if "target_temperature" in parameters:
            temp = parameters["target_temperature"]
            if self.thermal_params.temperature_range["min"] <= temp <= self.thermal_params.temperature_range["max"]:
                self.status["target_temperature"] = temp
                self.status["thermal_profile"] = "custom"
                
        if "target_emissivity" in parameters:
            emissivity = parameters["target_emissivity"]
            if 0.0 <= emissivity <= 1.0:
                self.status["target_emissivity"] = emissivity
                self.status["thermal_profile"] = "custom"
                
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
            
            # Update adaptation progress
            if self.status["adaptation_progress"] < 1.0:
                adaptation_step = self.thermal_params.adaptation_speed * time_delta / 5.0  # Full adaptation in ~5 seconds
                self.status["adaptation_progress"] = min(1.0, self.status["adaptation_progress"] + adaptation_step)
            
            # Update current temperature (move toward target)
            temp_diff = self.status["target_temperature"] - self.status["current_temperature"]
            temp_step = self.thermal_params.adaptation_speed * time_delta * 5.0  # 5°C per second at full speed
            if abs(temp_diff) > temp_step:
                self.status["current_temperature"] += temp_step if temp_diff > 0 else -temp_step
            else:
                self.status["current_temperature"] = self.status["target_temperature"]
                
            # Update current emissivity (move toward target)
            emissivity_diff = self.status["target_emissivity"] - self.status["current_emissivity"]
            emissivity_step = self.thermal_params.adaptation_speed * time_delta * 0.2  # 0.2 per second at full speed
            if abs(emissivity_diff) > emissivity_step:
                self.status["current_emissivity"] += emissivity_step if emissivity_diff > 0 else -emissivity_step
            else:
                self.status["current_emissivity"] = self.status["target_emissivity"]
            
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
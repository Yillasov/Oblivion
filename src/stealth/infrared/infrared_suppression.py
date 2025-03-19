"""
Infrared Suppression system implementation for Oblivion SDK.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.stealth.base.interfaces import NeuromorphicStealth, StealthSpecs, StealthType
from src.stealth.base.config import StealthSystemConfig, StealthOperationalMode


@dataclass
class IRSuppressionParameters:
    """Parameters for infrared suppression operation."""
    cooling_capacity: float  # Cooling capacity in kW
    surface_emissivity: float  # Surface emissivity (0.0-1.0)
    temperature_delta: float  # Maximum temperature reduction in °C
    power_level: float  # Power level (0.0-1.0)
    coverage_area: float  # Coverage area in m²
    response_time: float  # Response time in seconds


class InfraredSuppressionSystem(NeuromorphicStealth):
    """Infrared Suppression system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """
        Initialize Infrared Suppression system.
        
        Args:
            config: System configuration
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.config = config
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.INFRARED_SUPPRESSION,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=1.0,  # No effect on RCS
            infrared_signature=0.3,  # Significant reduction in IR signature
            acoustic_signature=1.1,  # Slight increase due to cooling systems
            activation_time=config.activation_time_seconds,
            operational_duration=config.operational_duration_minutes,
            cooldown_period=config.cooldown_time_seconds / 60.0  # Convert to minutes
        )
        
        # IR suppression specific parameters
        self.ir_params = IRSuppressionParameters(
            cooling_capacity=25.0,  # 25 kW cooling capacity
            surface_emissivity=0.3,  # Low emissivity
            temperature_delta=80.0,  # Can reduce temperature by up to 80°C
            power_level=0.0,  # Initial power level
            coverage_area=15.0,  # 15 m² coverage
            response_time=2.5  # 2.5 seconds response time
        )
        
        # System status
        self.status = {
            "active": False,
            "mode": "standby",
            "power_level": 0.0,
            "current_temperature": 20.0,  # Ambient temperature in °C
            "target_temperature": 20.0,  # Target temperature in °C
            "cooling_power": 0.0,  # Current cooling power in kW
            "remaining_operation_time": config.operational_duration_minutes,
            "cooldown_remaining": 0.0,
            "effectiveness": 0.0
        }
        
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the infrared suppression system."""
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
        Calculate infrared suppression effectiveness against specific threats.
        
        Args:
            threat_data: Information about the threat
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of effectiveness metrics
        """
        if not self.status["active"]:
            return {"ir_reduction": 0.0, "detection_probability": 1.0}
        
        # Extract threat information
        threat_type = threat_data.get("type", "infrared")
        threat_distance = threat_data.get("distance", 1000.0)  # meters
        sensor_sensitivity = threat_data.get("sensitivity", 0.5)  # 0.0-1.0
        
        # Extract environmental conditions
        ambient_temp = environmental_conditions.get("temperature", 20.0)  # °C
        humidity = environmental_conditions.get("humidity", 0.5)  # 0.0-1.0
        precipitation = environmental_conditions.get("precipitation", 0.0)  # 0.0-1.0
        
        # Calculate temperature difference from ambient
        temp_diff = max(0, self.status["current_temperature"] - self.status["target_temperature"])
        
        # Base effectiveness depends on temperature reduction and power level
        temp_reduction_factor = min(temp_diff / self.ir_params.temperature_delta, 1.0)
        power_factor = self.status["power_level"]
        
        # Calculate base effectiveness
        base_effectiveness = temp_reduction_factor * power_factor
        
        # Adjust for environmental factors
        # Higher humidity reduces IR transmission
        humidity_factor = 1.0 + (humidity * 0.2)
        
        # Precipitation significantly reduces IR transmission
        precipitation_factor = 1.0 + (precipitation * 0.5)
        
        # Distance factor (IR detection decreases with distance)
        distance_factor = min(1.0 + (threat_distance / 2000.0), 1.5)
        
        # Calculate final effectiveness
        if threat_type == "infrared" or threat_type == "thermal":
            type_factor = 1.0  # Full effectiveness against IR threats
        else:
            type_factor = 0.1  # Minimal effectiveness against other threats
            
        effectiveness = min(base_effectiveness * humidity_factor * precipitation_factor * 
                          distance_factor * type_factor, 0.95)
        
        # Calculate detection probability based on sensor sensitivity
        detection_probability = max(0.05, (1.0 - effectiveness) * sensor_sensitivity)
        
        # Update status
        self.status["effectiveness"] = effectiveness
        
        return {
            "ir_reduction": effectiveness,
            "detection_probability": detection_probability,
            "temperature_reduction": temp_diff,
            "effectiveness_factors": {
                "base": base_effectiveness,
                "humidity": humidity_factor,
                "precipitation": precipitation_factor,
                "distance": distance_factor,
                "threat_type": type_factor
            }
        }
    
    def activate(self, activation_params: Dict[str, Any] = {}) -> bool:
        """
        Activate the infrared suppression system.
        
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
        target_temp = activation_params.get("target_temperature", 
                                          self.status["current_temperature"] - 50.0)
        
        # Update system status
        self.status["active"] = True
        self.status["mode"] = "active"
        self.status["power_level"] = power_level
        self.status["target_temperature"] = max(-10.0, target_temp)  # Prevent unrealistic temperatures
        self.status["cooling_power"] = power_level * self.ir_params.cooling_capacity
        
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the infrared suppression system.
        
        Returns:
            Success status
        """
        if not self.initialized or not self.status["active"]:
            return False
            
        # Update system status
        self.status["active"] = False
        self.status["mode"] = "standby"
        self.status["power_level"] = 0.0
        self.status["cooling_power"] = 0.0
        self.status["target_temperature"] = self.status["current_temperature"]
        self.status["effectiveness"] = 0.0
        self.status["cooldown_remaining"] = self.specs.cooldown_period
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Adjust operational parameters of the infrared suppression system.
        
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
                self.status["cooling_power"] = power_level * self.ir_params.cooling_capacity
                
        if "target_temperature" in parameters:
            target_temp = parameters["target_temperature"]
            # Ensure target temperature is realistic
            self.status["target_temperature"] = max(-10.0, target_temp)
                
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
                return
                
            # Update temperature based on cooling power
            temp_diff = self.status["current_temperature"] - self.status["target_temperature"]
            if temp_diff > 0:
                # Calculate temperature change
                cooling_efficiency = 0.8  # 80% efficiency
                temp_change_rate = (self.status["cooling_power"] * cooling_efficiency) / 10.0  # °C per second
                temp_change = min(temp_diff, temp_change_rate * time_delta)
                
                # Update current temperature
                self.status["current_temperature"] -= temp_change
            
        elif self.status["cooldown_remaining"] > 0:
            # Update cooldown time
            time_delta_min = time_delta / 60.0
            self.status["cooldown_remaining"] -= time_delta_min
            
            if self.status["cooldown_remaining"] <= 0:
                self.status["cooldown_remaining"] = 0.0
                self.status["remaining_operation_time"] = self.config.operational_duration_minutes
                
            # Temperature gradually returns to ambient during cooldown
            ambient_temp = 20.0  # Assume standard ambient temperature
            temp_diff = ambient_temp - self.status["current_temperature"]
            if abs(temp_diff) > 0.1:
                # Temperature change rate during cooldown
                temp_change_rate = 0.5  # °C per second
                temp_change = min(abs(temp_diff), temp_change_rate * time_delta) * np.sign(temp_diff)
                
                # Update current temperature
                self.status["current_temperature"] += temp_change
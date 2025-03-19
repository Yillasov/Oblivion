"""
Metamaterial Cloaking simulation for Oblivion SDK.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.stealth.base.interfaces import NeuromorphicStealth, StealthSpecs, StealthType
from src.stealth.base.config import StealthSystemConfig, StealthOperationalMode


@dataclass
class MetamaterialProperties:
    """Properties of metamaterial cloaking system."""
    permittivity: float  # Electric permittivity
    permeability: float  # Magnetic permeability
    resonant_frequency: float  # Resonant frequency in GHz
    bandwidth: float  # Operational bandwidth in GHz
    thickness_mm: float  # Material thickness
    operational_wavelengths: List[float]  # Operational wavelengths in mm


class MetamaterialCloaking(NeuromorphicStealth):
    """Metamaterial Cloaking system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """
        Initialize Metamaterial Cloaking system.
        
        Args:
            config: System configuration
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.config = config
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.METAMATERIAL_CLOAKING,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=0.05,  # Extremely low RCS
            infrared_signature=0.3,  # Significant IR reduction
            acoustic_signature=0.7,  # Some acoustic reduction
            activation_time=config.activation_time_seconds,
            operational_duration=config.operational_duration_minutes,
            cooldown_period=config.cooldown_time_seconds / 60.0  # Convert to minutes
        )
        
        # Metamaterial properties
        self.metamaterial = MetamaterialProperties(
            permittivity=-1.0,  # Negative permittivity for cloaking
            permeability=-1.0,  # Negative permeability for cloaking
            resonant_frequency=10.0,  # 10 GHz
            bandwidth=2.0,  # 2 GHz bandwidth
            thickness_mm=5.0,
            operational_wavelengths=[30.0, 15.0, 10.0]  # mm wavelengths
        )
        
        # System status
        self.status = {
            "active": False,
            "mode": "standby",
            "power_level": 0.0,
            "frequency_tuning": self.metamaterial.resonant_frequency,
            "effectiveness": 0.0,
            "remaining_operation_time": config.operational_duration_minutes,
            "cooldown_remaining": 0.0,
            "temperature": 20.0  # Celsius
        }
        
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the metamaterial cloaking system."""
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
        Calculate metamaterial cloaking effectiveness against specific threats.
        
        Args:
            threat_data: Information about the threat
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of effectiveness metrics
        """
        if not self.status["active"]:
            return {"cloaking_effectiveness": 0.0, "detection_probability": 1.0}
        
        # Extract threat information
        threat_frequency = threat_data.get("frequency", 10.0)  # GHz
        threat_type = threat_data.get("type", "radar")
        
        # Extract environmental conditions
        temperature = environmental_conditions.get("temperature", 20.0)  # °C
        
        # Calculate frequency match (how close threat frequency is to resonant frequency)
        frequency_diff = abs(threat_frequency - self.status["frequency_tuning"])
        frequency_match = max(0.0, 1.0 - (frequency_diff / self.metamaterial.bandwidth))
        
        # Calculate temperature effect (metamaterials are sensitive to temperature)
        temp_diff = abs(temperature - 20.0)  # Difference from optimal temperature
        temp_factor = max(0.7, 1.0 - (temp_diff / 100.0))  # At most 30% reduction
        
        # Calculate power factor
        power_factor = self.status["power_level"]
        
        # Calculate base effectiveness based on threat type
        if threat_type == "radar":
            base_effectiveness = 0.95  # Very effective against radar
        elif threat_type == "infrared":
            base_effectiveness = 0.7   # Moderately effective against IR
        elif threat_type == "visual":
            base_effectiveness = 0.4   # Less effective against visual
        else:
            base_effectiveness = 0.3   # Limited effectiveness against other threats
        
        # Calculate final effectiveness
        effectiveness = base_effectiveness * frequency_match * temp_factor * power_factor
        
        # Calculate detection probability
        detection_probability = max(0.05, 1.0 - (effectiveness * 0.95))
        
        # Update status
        self.status["effectiveness"] = effectiveness
        
        return {
            "cloaking_effectiveness": effectiveness,
            "detection_probability": detection_probability,
            "frequency_match": frequency_match,
            "temperature_factor": temp_factor,
            "power_factor": power_factor,
            "base_effectiveness": base_effectiveness
        }
    
    def activate(self, activation_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Activate the metamaterial cloaking system.
        
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
        frequency = activation_params.get("frequency", self.metamaterial.resonant_frequency)
        
        # Update system status
        self.status["active"] = True
        self.status["mode"] = "active"
        self.status["power_level"] = power_level
        self.status["frequency_tuning"] = frequency
        
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the metamaterial cloaking system.
        
        Returns:
            Success status
        """
        if not self.initialized or not self.status["active"]:
            return False
            
        # Update system status
        self.status["active"] = False
        self.status["mode"] = "standby"
        self.status["power_level"] = 0.0
        self.status["effectiveness"] = 0.0
        self.status["cooldown_remaining"] = self.specs.cooldown_period
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Adjust operational parameters of the metamaterial cloaking system.
        
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
                
        if "frequency_tuning" in parameters:
            frequency = parameters["frequency_tuning"]
            # Check if frequency is within operational range
            if self.metamaterial.resonant_frequency - self.metamaterial.bandwidth <= frequency <= self.metamaterial.resonant_frequency + self.metamaterial.bandwidth:
                self.status["frequency_tuning"] = frequency
                
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
                
            # Simulate temperature increase during operation
            self.status["temperature"] += time_delta * 0.01 * self.status["power_level"]
            
        elif self.status["cooldown_remaining"] > 0:
            # Update cooldown time
            time_delta_min = time_delta / 60.0
            self.status["cooldown_remaining"] -= time_delta_min
            
            if self.status["cooldown_remaining"] <= 0:
                self.status["cooldown_remaining"] = 0.0
                self.status["remaining_operation_time"] = self.config.operational_duration_minutes
                
            # Simulate temperature decrease during cooldown
            self.status["temperature"] = max(20.0, self.status["temperature"] - time_delta * 0.02)
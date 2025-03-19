"""
Electromagnetic Pulse (EMP) Shielding system implementation for Oblivion SDK.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

from src.stealth.base.interfaces import NeuromorphicStealth, StealthSpecs, StealthType
from src.stealth.base.config import StealthSystemConfig


@dataclass
class EMPShieldingParameters:
    """Parameters for EMP shielding operation."""
    max_attenuation: float  # Maximum attenuation in dB
    frequency_range: Dict[str, float]  # Operating frequency range in Hz
    recovery_time: float  # Recovery time after EMP hit in seconds
    faraday_cage_integrity: float  # Integrity of Faraday cage (0.0-1.0)
    surge_protection_level: float  # Surge protection level in kV


class EMPShieldingSystem(NeuromorphicStealth):
    """Electromagnetic Pulse Shielding system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """Initialize EMP Shielding system."""
        super().__init__(hardware_interface)
        self.config = config
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.ELECTROMAGNETIC_SHIELDING,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=1.0,  # No effect on RCS
            infrared_signature=1.0,  # No effect on IR
            acoustic_signature=1.0,  # No effect on acoustics
            activation_time=config.activation_time_seconds,
            operational_duration=config.operational_duration_minutes,
            cooldown_period=config.cooldown_time_seconds / 60.0
        )
        
        # EMP shielding specific parameters
        self.emp_params = EMPShieldingParameters(
            max_attenuation=80.0,  # 80 dB maximum attenuation
            frequency_range={"min": 100.0, "max": 10.0e9},  # 100 Hz to 10 GHz
            recovery_time=5.0,  # 5 seconds recovery time
            faraday_cage_integrity=0.95,  # 95% integrity
            surge_protection_level=50.0  # 50 kV surge protection
        )
        
        # System status
        self.status = {
            "active": False,
            "mode": "standby",
            "power_level": 0.0,
            "shielding_integrity": 1.0,
            "last_emp_timestamp": 0.0,
            "recovering": False,
            "remaining_operation_time": config.operational_duration_minutes,
            "effectiveness": 0.0
        }
        
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the EMP shielding system."""
        self.initialized = True
        self.status["mode"] = "standby"
        return True
    
    def get_specifications(self) -> StealthSpecs:
        """Get the physical specifications of the stealth system."""
        return self.specs
    
    def calculate_effectiveness(self, 
                              threat_data: Dict[str, Any],
                              environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate EMP shielding effectiveness against specific threats."""
        # Base effectiveness even when not active
        passive_effectiveness = 0.5  # Passive shielding from Faraday cage
        
        if not self.status["active"]:
            return {
                "emp_protection": passive_effectiveness,
                "mode": "passive"
            }
        
        # Extract threat information
        threat_type = threat_data.get("type", "emp")
        threat_strength = threat_data.get("strength", 10.0)  # kV/m
        threat_frequency = threat_data.get("frequency", 1.0e6)  # Hz
        
        # Calculate base effectiveness based on power level
        power_factor = self.status["power_level"]
        
        # Check if frequency is within our operating range
        min_freq = self.emp_params.frequency_range["min"]
        max_freq = self.emp_params.frequency_range["max"]
        
        if min_freq <= threat_frequency <= max_freq:
            frequency_match = 1.0
        else:
            # Effectiveness drops outside operating range
            frequency_match = 0.5
        
        # Calculate active effectiveness
        active_effectiveness = self.emp_params.faraday_cage_integrity * power_factor * frequency_match
        
        # Combined effectiveness (active + passive)
        combined_effectiveness = passive_effectiveness + (active_effectiveness * (1.0 - passive_effectiveness))
        
        # Adjust for threat strength
        if threat_strength > self.emp_params.surge_protection_level:
            # Effectiveness decreases with stronger EMPs
            strength_factor = self.emp_params.surge_protection_level / threat_strength
            combined_effectiveness *= strength_factor
        
        # Calculate final effectiveness
        if threat_type == "emp" or threat_type == "electromagnetic":
            type_factor = 1.0  # Full effectiveness against EMP threats
        else:
            type_factor = 0.2  # Limited effectiveness against other threats
            
        effectiveness = min(combined_effectiveness * type_factor, 0.98)
        
        # Update status
        self.status["effectiveness"] = effectiveness
        
        return {
            "emp_protection": effectiveness,
            "attenuation_db": effectiveness * self.emp_params.max_attenuation,
            "mode": "active",
            "protected_frequency_range": f"{min_freq} Hz - {max_freq} Hz",
            "max_protected_surge": f"{self.emp_params.surge_protection_level} kV"
        }
    
    def activate(self, activation_params: Dict[str, Any] = {}) -> bool:
        """Activate the EMP shielding system."""
        if not self.initialized:
            return False
            
        # Set default parameters if none provided
        if activation_params is None:
            activation_params = {}
            
        # Extract activation parameters
        power_level = activation_params.get("power_level", 0.8)
        
        # Update system status
        self.status["active"] = True
        self.status["mode"] = "active"
        self.status["power_level"] = power_level
        
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the EMP shielding system."""
        if not self.initialized or not self.status["active"]:
            return False
            
        # Update system status
        self.status["active"] = False
        self.status["mode"] = "standby"
        self.status["power_level"] = 0.0
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Adjust operational parameters of the EMP shielding system."""
        if not self.initialized:
            return False
            
        if "power_level" in parameters:
            power_level = parameters["power_level"]
            if 0.0 <= power_level <= 1.0:
                self.status["power_level"] = power_level
                
        return True
    
    def register_emp_hit(self, strength: float, timestamp: float) -> Dict[str, Any]:
        """
        Register an EMP hit and calculate damage/recovery.
        
        Args:
            strength: EMP strength in kV/m
            timestamp: Current timestamp
            
        Returns:
            Hit results and system status
        """
        # Calculate protection based on current effectiveness
        protection = self.status["effectiveness"]
        
        # Calculate damage to shielding
        damage = 0.0
        if strength > self.emp_params.surge_protection_level * protection:
            # Calculate damage to shielding integrity
            excess_strength = strength - (self.emp_params.surge_protection_level * protection)
            damage = min(0.5, excess_strength / (self.emp_params.surge_protection_level * 2))
            
            # Apply damage
            self.status["shielding_integrity"] = max(0.0, self.status["shielding_integrity"] - damage)
            
            # Enter recovery mode
            self.status["recovering"] = True
            self.status["last_emp_timestamp"] = timestamp
        
        return {
            "hit_registered": True,
            "damage": damage,
            "remaining_integrity": self.status["shielding_integrity"],
            "recovery_time": self.emp_params.recovery_time,
            "recovering": self.status["recovering"]
        }
    
    def update_system(self, current_time: float) -> None:
        """
        Update system status based on time elapsed.
        
        Args:
            current_time: Current timestamp
        """
        if not self.initialized:
            return
            
        # Check if system is recovering from EMP hit
        if self.status["recovering"]:
            time_since_hit = current_time - self.status["last_emp_timestamp"]
            
            if time_since_hit >= self.emp_params.recovery_time:
                # Recovery complete
                self.status["recovering"] = False
                
                # Partial recovery of integrity
                self.status["shielding_integrity"] = min(1.0, self.status["shielding_integrity"] + 0.2)
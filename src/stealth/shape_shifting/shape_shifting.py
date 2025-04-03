#!/usr/bin/env python3
"""
Shape-Shifting Surfaces implementation for Oblivion SDK.
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
class ShapeShiftingParameters:
    """Parameters for shape-shifting surfaces."""
    max_deformation: float  # Maximum surface deformation in mm
    response_time: float  # Response time in seconds
    precision: float  # Deformation precision in mm
    energy_density: float  # Energy required per unit area (J/m²)
    memory_positions: int  # Number of memorized positions
    material_elasticity: float  # Material elasticity (0.0-1.0)


class ShapeShiftingSurfaces(NeuromorphicStealth):
    """Shape-Shifting Surfaces system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """
        Initialize Shape-Shifting Surfaces system.
        
        Args:
            config: System configuration
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.config = config
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.SHAPE_SHIFTING,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=0.2,  # Significant RCS reduction
            infrared_signature=0.7,  # Moderate IR reduction
            acoustic_signature=0.6,  # Moderate acoustic reduction
            activation_time=config.activation_time_seconds,
            operational_duration=config.operational_duration_minutes,
            cooldown_period=config.cooldown_time_seconds / 60.0  # Convert to minutes
        )
        
        # Shape-shifting specific parameters
        self.shape_params = ShapeShiftingParameters(
            max_deformation=50.0,  # 50mm maximum deformation
            response_time=1.2,  # 1.2 seconds response time
            precision=0.5,  # 0.5mm precision
            energy_density=120.0,  # 120 J/m²
            memory_positions=5,  # 5 memorized positions
            material_elasticity=0.85  # 85% elasticity
        )
        
        # System status
        self.status = {
            "active": False,
            "mode": "standby",
            "power_level": 0.0,
            "current_deformation": 0.0,  # Current deformation in mm
            "target_profile": "neutral",  # Current target profile
            "remaining_operation_time": config.operational_duration_minutes,
            "cooldown_remaining": 0.0,
            "effectiveness": 0.0,
            "stored_profiles": ["neutral"]  # Default neutral profile
        }
        
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the shape-shifting system."""
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
        Calculate shape-shifting effectiveness against specific threats.
        
        Args:
            threat_data: Information about the threat
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of effectiveness metrics
        """
        if not self.status["active"]:
            return {"shape_shifting_effectiveness": 0.0, "detection_probability": 1.0}
        
        # Extract threat information
        threat_type = threat_data.get("type", "radar")
        threat_frequency = threat_data.get("frequency", 10.0)  # GHz
        threat_angle = threat_data.get("angle", 0.0)  # Degrees
        
        # Extract environmental conditions
        temperature = environmental_conditions.get("temperature", 20.0)  # °C
        wind_speed = environmental_conditions.get("wind_speed", 0.0)  # m/s
        
        # Calculate base effectiveness based on power level and deformation
        power_factor = self.status["power_level"]
        deformation_factor = min(self.status["current_deformation"] / self.shape_params.max_deformation, 1.0)
        
        # Calculate profile match (how well current profile matches threat)
        profile_match = 0.7  # Default moderate match
        if self.status["target_profile"] == "radar_deflection" and threat_type == "radar":
            profile_match = 0.9
        elif self.status["target_profile"] == "infrared_reduction" and threat_type == "infrared":
            profile_match = 0.9
        elif self.status["target_profile"] == "acoustic_diffusion" and threat_type == "acoustic":
            profile_match = 0.9
        
        # Environmental adjustments
        # High winds reduce effectiveness
        wind_factor = max(0.6, 1.0 - (wind_speed / 50.0))
        
        # Temperature affects material properties
        temp_diff = abs(temperature - 20.0)  # Difference from optimal temperature
        temp_factor = max(0.7, 1.0 - (temp_diff / 100.0))
        
        # Calculate final effectiveness
        base_effectiveness = power_factor * deformation_factor * profile_match
        effectiveness = min(base_effectiveness * wind_factor * temp_factor, 0.95)
        
        # Calculate detection probability based on threat type
        if threat_type == "radar":
            detection_probability = max(0.05, 1.0 - (effectiveness * 0.9))
        elif threat_type == "infrared":
            detection_probability = max(0.1, 1.0 - (effectiveness * 0.7))
        elif threat_type == "acoustic":
            detection_probability = max(0.1, 1.0 - (effectiveness * 0.8))
        else:
            detection_probability = max(0.2, 1.0 - (effectiveness * 0.5))
        
        # Update status
        self.status["effectiveness"] = effectiveness
        
        return {
            "shape_shifting_effectiveness": effectiveness,
            "detection_probability": detection_probability,
            "profile_match": profile_match,
            "effectiveness_factors": {
                "power": power_factor,
                "deformation": deformation_factor,
                "profile_match": profile_match,
                "wind": wind_factor,
                "temperature": temp_factor
            }
        }
    
    def activate(self, activation_params: Dict[str, Any] = {}) -> bool:
        """
        Activate the shape-shifting system.
        
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
        target_profile = activation_params.get("profile", "neutral")
        target_deformation = activation_params.get("deformation", self.shape_params.max_deformation * 0.7)
        
        # Validate target profile
        if target_profile not in self.status["stored_profiles"]:
            target_profile = "neutral"
        
        # Update system status
        self.status["active"] = True
        self.status["mode"] = "active"
        self.status["power_level"] = power_level
        self.status["target_profile"] = target_profile
        self.status["current_deformation"] = min(target_deformation, self.shape_params.max_deformation)
        
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the shape-shifting system.
        
        Returns:
            Success status
        """
        if not self.initialized or not self.status["active"]:
            return False
            
        # Update system status
        self.status["active"] = False
        self.status["mode"] = "standby"
        self.status["power_level"] = 0.0
        self.status["current_deformation"] = 0.0
        self.status["target_profile"] = "neutral"
        self.status["effectiveness"] = 0.0
        self.status["cooldown_remaining"] = self.specs.cooldown_period
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Adjust operational parameters of the shape-shifting system.
        
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
            if profile in self.status["stored_profiles"]:
                self.status["target_profile"] = profile
                
        if "deformation" in parameters:
            deformation = parameters["deformation"]
            if 0.0 <= deformation <= self.shape_params.max_deformation:
                self.status["current_deformation"] = deformation
                
        return True
    
    def store_profile(self, profile_name: str, profile_data: Dict[str, Any]) -> bool:
        """
        Store a new shape-shifting profile.
        
        Args:
            profile_name: Name of the profile
            profile_data: Profile configuration data
            
        Returns:
            Success status
        """
        if not self.initialized:
            return False
            
        # Check if we've reached the maximum number of stored profiles
        if len(self.status["stored_profiles"]) >= self.shape_params.memory_positions:
            return False
            
        # Add profile to stored profiles
        if profile_name not in self.status["stored_profiles"]:
            self.status["stored_profiles"].append(profile_name)
            
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
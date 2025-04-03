#!/usr/bin/env python3
"""
Low-Observable Nozzle implementation for stealth propulsion integration.
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
from src.propulsion.base import PropulsionInterface


@dataclass
class NozzleParameters:
    """Parameters for low-observable nozzle operation."""
    ir_suppression_level: float  # IR suppression level (0.0-1.0)
    radar_cross_section: float  # RCS in square meters
    thrust_loss: float  # Thrust loss due to stealth features (0.0-1.0)
    noise_reduction: float  # Noise reduction in dB
    temperature_reduction: float  # Temperature reduction in %
    aspect_ratio: float  # Nozzle aspect ratio
    cooling_efficiency: float  # Cooling efficiency (0.0-1.0)


class LowObservableNozzle(NeuromorphicStealth):
    """Low-Observable Nozzle system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """
        Initialize Low-Observable Nozzle system.
        
        Args:
            config: System configuration
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.config = config
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.LOW_OBSERVABLE_NOZZLE,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=0.05,  # Very low RCS
            infrared_signature=0.3,  # Significant reduction in IR signature
            acoustic_signature=0.4,  # Significant noise reduction
            activation_time=config.activation_time_seconds,
            operational_duration=config.operational_duration_minutes,
            cooldown_period=config.cooldown_time_seconds / 60.0  # Convert to minutes
        )
        
        # Nozzle-specific parameters
        self.nozzle_params = NozzleParameters(
            ir_suppression_level=0.7,  # 70% IR suppression
            radar_cross_section=0.05,  # Very low RCS in m²
            thrust_loss=0.15,  # 15% thrust loss
            noise_reduction=12.0,  # 12 dB noise reduction
            temperature_reduction=40.0,  # 40% temperature reduction
            aspect_ratio=4.5,  # Aspect ratio for rectangular nozzle
            cooling_efficiency=0.8  # 80% cooling efficiency
        )
        
        # System status
        self.status = {
            "active": False,
            "mode": "standby",
            "power_level": 0.0,
            "ir_suppression_active": False,
            "shape_adaptation_active": False,
            "cooling_active": False,
            "exhaust_temperature": 0.0,  # Current exhaust temperature in °C
            "exhaust_velocity": 0.0,  # Current exhaust velocity in m/s
            "propulsion_integration_status": "disconnected",
            "remaining_operation_time": config.operational_duration_minutes,
            "cooldown_remaining": 0.0
        }
        
        self.initialized = False
        self.propulsion_system = None
    
    def initialize(self) -> bool:
        """Initialize the low-observable nozzle system."""
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
        Calculate low-observable nozzle effectiveness against specific threats.
        
        Args:
            threat_data: Information about the threat
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of effectiveness metrics
        """
        if not self.status["active"]:
            return {"ir_reduction": 0.0, "radar_reduction": 0.0, "acoustic_reduction": 0.0}
        
        # Extract threat information
        threat_type = threat_data.get("type", "infrared")
        threat_distance = threat_data.get("distance", 1000.0)  # meters
        
        # Extract environmental conditions
        ambient_temp = environmental_conditions.get("temperature", 20.0)  # °C
        humidity = environmental_conditions.get("humidity", 50.0)  # %
        
        # Calculate base effectiveness based on nozzle parameters
        ir_effectiveness = self.nozzle_params.ir_suppression_level * self.status["power_level"]
        radar_effectiveness = 1.0 - (self.nozzle_params.radar_cross_section / 0.5)  # Normalize to 0.5m²
        acoustic_effectiveness = self.nozzle_params.noise_reduction / 20.0  # Normalize to 20dB
        
        # Adjust for power level
        power_factor = self.status["power_level"]
        
        # Adjust for distance (more effective at greater distances)
        distance_factor = min(1.0 + (threat_distance / 2000.0), 1.5)
        
        # Calculate final effectiveness based on threat type
        if threat_type == "infrared" or threat_type == "thermal":
            type_factor = 1.0  # Full effectiveness against IR threats
            effectiveness = ir_effectiveness * power_factor * distance_factor
        elif threat_type == "radar":
            type_factor = 0.8  # Good effectiveness against radar
            effectiveness = radar_effectiveness * power_factor * distance_factor
        elif threat_type == "acoustic":
            type_factor = 0.9  # Very good effectiveness against acoustic
            effectiveness = acoustic_effectiveness * power_factor * distance_factor
        else:
            type_factor = 0.3  # Limited effectiveness against other threats
            effectiveness = 0.3 * power_factor * distance_factor
            
        # Calculate thrust impact
        thrust_impact = self.nozzle_params.thrust_loss * power_factor
        
        return {
            "ir_reduction": ir_effectiveness * type_factor,
            "radar_reduction": radar_effectiveness * type_factor,
            "acoustic_reduction": acoustic_effectiveness * type_factor,
            "overall_effectiveness": effectiveness,
            "thrust_impact": thrust_impact,
            "effectiveness_factors": {
                "ir_suppression": ir_effectiveness,
                "radar_cross_section": radar_effectiveness,
                "acoustic_reduction": acoustic_effectiveness,
                "power": power_factor,
                "distance": distance_factor,
                "threat_type": type_factor
            }
        }
    
    def activate(self, activation_params: Dict[str, Any] = {}) -> bool:
        """
        Activate the low-observable nozzle system.
        
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
        ir_suppression = activation_params.get("ir_suppression", True)
        shape_adaptation = activation_params.get("shape_adaptation", True)
        cooling = activation_params.get("cooling", True)
        
        # Update system status
        self.status["active"] = True
        self.status["mode"] = "active"
        self.status["power_level"] = power_level
        self.status["ir_suppression_active"] = ir_suppression
        self.status["shape_adaptation_active"] = shape_adaptation
        self.status["cooling_active"] = cooling
        
        # Update from propulsion system if connected
        if self.propulsion_system:
            self.update_from_propulsion()
        
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the low-observable nozzle system.
        
        Returns:
            Success status
        """
        if not self.initialized or not self.status["active"]:
            return False
            
        # Update system status
        self.status["active"] = False
        self.status["mode"] = "standby"
        self.status["power_level"] = 0.0
        self.status["ir_suppression_active"] = False
        self.status["shape_adaptation_active"] = False
        self.status["cooling_active"] = False
        self.status["cooldown_remaining"] = self.specs.cooldown_period
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Adjust operational parameters of the low-observable nozzle system.
        
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
                
        if "ir_suppression_active" in parameters:
            self.status["ir_suppression_active"] = bool(parameters["ir_suppression_active"])
            
        if "shape_adaptation_active" in parameters:
            self.status["shape_adaptation_active"] = bool(parameters["shape_adaptation_active"])
            
        if "cooling_active" in parameters:
            self.status["cooling_active"] = bool(parameters["cooling_active"])
            
        if "ir_suppression_level" in parameters:
            level = parameters["ir_suppression_level"]
            if 0.0 <= level <= 1.0:
                self.nozzle_params.ir_suppression_level = level
                
        if "aspect_ratio" in parameters:
            ratio = parameters["aspect_ratio"]
            if 1.0 <= ratio <= 10.0:
                self.nozzle_params.aspect_ratio = ratio
                
        return True
    
    def connect_propulsion_system(self, propulsion_system: PropulsionInterface) -> bool:
        """
        Connect to a propulsion system for integration.
        
        Args:
            propulsion_system: Propulsion system to connect
            
        Returns:
            Success status
        """
        if not self.initialized:
            return False
            
        self.propulsion_system = propulsion_system
        self.status["propulsion_integration_status"] = "connected"
        
        # Initial update from propulsion
        self.update_from_propulsion()
        
        return True
    
    def update_from_propulsion(self) -> bool:
        """
        Update nozzle parameters based on current propulsion system state.
        
        Returns:
            Success status
        """
        if not self.propulsion_system or self.status["propulsion_integration_status"] != "connected":
            return False
            
        # Get current propulsion status
        try:
            propulsion_status = self.propulsion_system.get_status()
            
            # Update exhaust parameters
            self.status["exhaust_temperature"] = propulsion_status.get("exhaust_temperature", 800.0)
            self.status["exhaust_velocity"] = propulsion_status.get("exhaust_velocity", 500.0)
            
            # Adjust nozzle parameters based on propulsion power
            power_level = propulsion_status.get("power_level", 0.5)
            
            # If propulsion is at high power, we may need to adjust our stealth capabilities
            if power_level > 0.8 and self.status["active"]:
                # At high power, IR suppression becomes more challenging
                self.nozzle_params.ir_suppression_level = max(0.4, 0.7 - (power_level - 0.8) * 0.5)
                # Thrust loss increases at high power
                self.nozzle_params.thrust_loss = min(0.25, 0.15 + (power_level - 0.8) * 0.2)
            else:
                # Reset to default values
                self.nozzle_params.ir_suppression_level = 0.7
                self.nozzle_params.thrust_loss = 0.15
                
            return True
        except Exception:
            return False
    
    def get_propulsion_impact(self) -> Dict[str, float]:
        """
        Get the impact of the nozzle on propulsion performance.
        
        Returns:
            Dictionary of impact metrics
        """
        if not self.status["active"]:
            return {"thrust_loss": 0.0, "fuel_efficiency_impact": 0.0}
            
        # Calculate impact based on current settings
        thrust_loss = self.nozzle_params.thrust_loss
        
        # Shape adaptation can reduce thrust loss
        if self.status["shape_adaptation_active"]:
            thrust_loss *= 0.8
            
        # Calculate fuel efficiency impact
        # Low-observable features can slightly improve fuel efficiency
        fuel_efficiency_impact = -0.05 if self.status["shape_adaptation_active"] else 0.0
        
        return {
            "thrust_loss": thrust_loss,
            "fuel_efficiency_impact": fuel_efficiency_impact
        }
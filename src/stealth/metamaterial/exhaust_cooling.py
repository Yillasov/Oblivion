"""
Exhaust cooling and mixing system for IR signature reduction.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.stealth.metamaterial.thermal_management import MetamaterialThermalManager, CoolingMethod


class ExhaustMixingMode(Enum):
    """Exhaust mixing modes for IR signature reduction."""
    PASSIVE = "passive"
    ACTIVE_DILUTION = "active_dilution"
    BYPASS_MIXING = "bypass_mixing"
    MULTI_STAGE = "multi_stage"
    EMERGENCY = "emergency"


@dataclass
class ExhaustProfile:
    """Exhaust thermal and flow characteristics."""
    temperature: float  # Exhaust temperature in °C
    flow_rate: float  # Flow rate in kg/s
    velocity: float  # Exhaust velocity in m/s
    diameter: float  # Exhaust diameter in m
    ir_emissivity: float  # IR emissivity (0.0-1.0)


class ExhaustCoolingSystem:
    """
    Exhaust cooling and mixing system for IR signature reduction.
    Integrates with thermal management to reduce propulsion thermal signatures.
    """
    
    def __init__(self, thermal_manager: Optional[MetamaterialThermalManager] = None):
        """
        Initialize exhaust cooling system.
        
        Args:
            thermal_manager: Optional thermal manager for integration
        """
        self.thermal_manager = thermal_manager
        self.mixing_mode = ExhaustMixingMode.PASSIVE
        self.exhaust_profiles: Dict[str, ExhaustProfile] = {}
        self.cooling_efficiency = 0.0
        self.bypass_ratio = 0.0
        self.active = False
        self.power_consumption = 0.0
        
    def register_exhaust(self, exhaust_id: str, profile: ExhaustProfile) -> None:
        """Register an exhaust for cooling and mixing."""
        self.exhaust_profiles[exhaust_id] = profile
        
    def set_mixing_mode(self, mode: ExhaustMixingMode) -> bool:
        """
        Set exhaust mixing mode.
        
        Args:
            mode: Mixing mode
            
        Returns:
            Success status
        """
        self.mixing_mode = mode
        
        # Configure system based on mode
        if mode == ExhaustMixingMode.PASSIVE:
            self.bypass_ratio = 0.2
            self.cooling_efficiency = 0.3
            self.power_consumption = 0.0
        elif mode == ExhaustMixingMode.ACTIVE_DILUTION:
            self.bypass_ratio = 0.5
            self.cooling_efficiency = 0.6
            self.power_consumption = 2.0
        elif mode == ExhaustMixingMode.BYPASS_MIXING:
            self.bypass_ratio = 0.8
            self.cooling_efficiency = 0.5
            self.power_consumption = 1.5
        elif mode == ExhaustMixingMode.MULTI_STAGE:
            self.bypass_ratio = 0.7
            self.cooling_efficiency = 0.8
            self.power_consumption = 3.0
        elif mode == ExhaustMixingMode.EMERGENCY:
            self.bypass_ratio = 0.9
            self.cooling_efficiency = 0.9
            self.power_consumption = 5.0
            
        return True
        
    def activate(self) -> bool:
        """
        Activate exhaust cooling system.
        
        Returns:
            Success status
        """
        self.active = True
        return True
        
    def deactivate(self) -> bool:
        """
        Deactivate exhaust cooling system.
        
        Returns:
            Success status
        """
        self.active = False
        self.power_consumption = 0.0
        return True
    
    def calculate_ir_reduction(self, exhaust_id: str) -> Dict[str, Any]:
        """
        Calculate IR signature reduction for an exhaust.
        
        Args:
            exhaust_id: Exhaust identifier
            
        Returns:
            IR reduction metrics
        """
        if not self.active or exhaust_id not in self.exhaust_profiles:
            return {"reduction_factor": 0.0, "status": "inactive"}
            
        profile = self.exhaust_profiles[exhaust_id]
        
        # Calculate temperature reduction
        ambient_temp = 20.0  # Default ambient temperature
        if self.thermal_manager:
            ambient_temp = self.thermal_manager.ambient_temperature
            
        max_temp_reduction = profile.temperature - ambient_temp
        actual_reduction = max_temp_reduction * self.cooling_efficiency
        
        # Calculate mixed exhaust temperature
        mixed_temp = profile.temperature - actual_reduction
        
        # Calculate IR signature reduction (simplified model)
        # Based on Stefan-Boltzmann law (T^4 relationship)
        original_emission = profile.ir_emissivity * (profile.temperature + 273.15)**4
        reduced_emission = profile.ir_emissivity * (mixed_temp + 273.15)**4
        
        reduction_factor = 1.0 - (reduced_emission / original_emission)
        
        # Calculate plume characteristics
        plume_length_reduction = self.bypass_ratio * 0.8
        plume_visibility_reduction = self.cooling_efficiency * 0.9
        
        return {
            "original_temp": profile.temperature,
            "mixed_temp": mixed_temp,
            "temp_reduction": actual_reduction,
            "reduction_factor": reduction_factor,
            "plume_length_reduction": plume_length_reduction,
            "plume_visibility_reduction": plume_visibility_reduction,
            "power_consumption": self.power_consumption,
            "status": "active"
        }
    
    def update_exhaust_mixing(self, 
                            exhaust_id: str, 
                            thrust_level: float,
                            ambient_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Update exhaust mixing based on thrust and ambient conditions.
        
        Args:
            exhaust_id: Exhaust identifier
            thrust_level: Current thrust level (0.0-1.0)
            ambient_conditions: Ambient conditions
            
        Returns:
            Updated exhaust parameters
        """
        if not self.active or exhaust_id not in self.exhaust_profiles:
            return {"success": False, "status": "inactive"}
            
        profile = self.exhaust_profiles[exhaust_id]
        
        # Extract ambient conditions
        ambient_temp = ambient_conditions.get("temperature", 20.0)
        ambient_pressure = ambient_conditions.get("pressure", 101.3)
        wind_speed = ambient_conditions.get("wind_speed", 0.0)
        
        # Calculate base exhaust parameters based on thrust
        base_temp = 100.0 + (900.0 * thrust_level)  # 100-1000°C based on thrust
        base_flow = profile.flow_rate * thrust_level
        base_velocity = profile.velocity * thrust_level
        
        # Apply mixing effects
        mixed_temp = base_temp - (base_temp - ambient_temp) * self.cooling_efficiency
        mixed_flow = base_flow * (1.0 + self.bypass_ratio)
        mixed_velocity = base_velocity * (1.0 / (1.0 + self.bypass_ratio))
        
        # Update profile
        updated_profile = ExhaustProfile(
            temperature=mixed_temp,
            flow_rate=mixed_flow,
            velocity=mixed_velocity,
            diameter=profile.diameter,
            ir_emissivity=profile.ir_emissivity
        )
        
        self.exhaust_profiles[exhaust_id] = updated_profile
        
        # Calculate IR reduction
        ir_reduction = self.calculate_ir_reduction(exhaust_id)
        
        return {
            "success": True,
            "status": "active",
            "mode": self.mixing_mode.value,
            "original_temp": base_temp,
            "mixed_temp": mixed_temp,
            "flow_increase": self.bypass_ratio * 100.0,
            "velocity_reduction": (1.0 - (mixed_velocity / base_velocity)) * 100.0,
            "ir_reduction": ir_reduction["reduction_factor"] * 100.0,
            "power_consumption": self.power_consumption
        }
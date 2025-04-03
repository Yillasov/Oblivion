"""
Conventional weapon payload systems for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, field

from src.payload.base import NeuromorphicPayload, PayloadSpecs
from src.payload.types import WeaponType, PayloadCategory, PayloadMountType


@dataclass
class WeaponSpecs(PayloadSpecs):
    """Extended specifications for weapon systems."""
    weapon_type: WeaponType
    range: float  # Maximum effective range in km
    payload_weight: float  # Weight of the actual warhead/explosive in kg
    guidance_system: str  # Type of guidance system
    launch_parameters: Dict[str, Any] = field(default_factory=dict)  # Launch requirements


class ConventionalWeapon(NeuromorphicPayload):
    """Base class for conventional weapon systems."""
    
    def __init__(self, specs: WeaponSpecs, hardware_interface=None):
        """
        Initialize a conventional weapon system.
        
        Args:
            specs: Weapon specifications
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.specs = specs
        self.status = {
            "armed": False,
            "ready": False,
            "target_locked": False,
            "deployment_progress": 0.0
        }
    
    def get_specifications(self) -> PayloadSpecs:
        """Get the physical specifications of the weapon."""
        return self.specs
    
    def calculate_impact(self) -> Dict[str, float]:
        """Calculate the impact of this weapon on UCAV performance."""
        # Basic impact calculation
        return {
            "weight_impact": self.specs.weight,
            "drag_coefficient": 0.05,  # Default value, will be refined
            "power_consumption": self.specs.power_requirements
        }
    
    def arm(self) -> bool:
        """Arm the weapon system."""
        if self.initialized:
            self.status["armed"] = True
            return True
        return False
    
    def lock_target(self, target_data: Dict[str, Any]) -> bool:
        """
        Lock onto a target.
        
        Args:
            target_data: Data about the target
            
        Returns:
            Success status
        """
        if not self.initialized or not self.status["armed"]:
            return False
            
        # Use neuromorphic processing for target validation
        target_valid = self.process_data({
            "target": target_data,
            "computation": "target_validation"
        })
        
        self.status["target_locked"] = target_valid.get("valid", False)
        return self.status["target_locked"]
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy the weapon.
        
        Args:
            target_data: Data about the target
            
        Returns:
            Success status
        """
        if not self.status["armed"] or not self.status["target_locked"]:
            return False
            
        # Simulate deployment
        self.status["deployment_progress"] = 100.0
        self.status["armed"] = False
        self.status["target_locked"] = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the weapon."""
        return self.status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using neuromorphic computing.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Dict containing processed results
        """
        if not self.hardware_interface or not self.initialized:
            return {"error": "Hardware interface not initialized"}
            
        # Process data using neuromorphic hardware
        computation_type = input_data.get("computation", "")
        
        if computation_type == "target_validation":
            # Validate target using neuromorphic processing
            return {
                "valid": True,  # Simplified for now
                "confidence": 0.95,
                "estimated_success_probability": 0.85
            }
        elif computation_type == "trajectory_calculation":
            # Calculate optimal trajectory
            return {
                "trajectory": [
                    {"x": 0, "y": 0, "z": 0, "t": 0},
                    {"x": 100, "y": 100, "z": 50, "t": 5},
                    {"x": 200, "y": 200, "z": 100, "t": 10}
                ],
                "impact_time": 10.5,
                "energy_required": 500.0
            }
        
        return {"error": "Unknown computation type"}
    
    def train(self, training_data: Dict[str, Any]) -> bool:
        """
        Train the neuromorphic components of the weapon.
        
        Args:
            training_data: Data for training
            
        Returns:
            Success status
        """
        if not self.hardware_interface or not self.initialized:
            return False
            
        # Simplified training process
        return True


class AirToAirMissile(ConventionalWeapon):
    """Air-to-air missile system."""
    
    def __init__(self, model: str, hardware_interface=None):
        """
        Initialize an air-to-air missile.
        
        Args:
            model: Missile model identifier
            hardware_interface: Interface to neuromorphic hardware
        """
        # Define specifications based on model
        if model == "AIM-120":
            specs = WeaponSpecs(
                weight=152.0,
                volume={"length": 3.7, "diameter": 0.178},
                power_requirements=500.0,
                mounting_points=["external_hardpoint", "internal_bay"],
                weapon_type=WeaponType.MISSILE,
                range=75.0,
                payload_weight=22.5,
                guidance_system="active_radar"
            )
        elif model == "AIM-9X":
            specs = WeaponSpecs(
                weight=85.0,
                volume={"length": 3.0, "diameter": 0.127},
                power_requirements=350.0,
                mounting_points=["external_hardpoint", "internal_bay"],
                weapon_type=WeaponType.MISSILE,
                range=35.0,
                payload_weight=9.4,
                guidance_system="infrared"
            )
        else:
            raise ValueError(f"Unknown missile model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
    
    def calculate_impact(self) -> Dict[str, float]:
        """Calculate the impact of this missile on UCAV performance."""
        base_impact = super().calculate_impact()
        
        # Add missile-specific impacts
        if self.model == "AIM-120":
            base_impact["drag_coefficient"] = 0.08
        elif self.model == "AIM-9X":
            base_impact["drag_coefficient"] = 0.06
            
        return base_impact
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using neuromorphic computing for missile guidance."""
        base_result = super().process_data(input_data)
        
        # Add missile-specific processing
        if input_data.get("computation") == "target_validation":
            # Enhanced target validation for missiles
            base_result["intercept_probability"] = 0.92
            base_result["evasion_difficulty"] = 0.75
            
        return base_result


class GuidedBomb(ConventionalWeapon):
    """Guided bomb system."""
    
    def __init__(self, model: str, hardware_interface=None):
        """
        Initialize a guided bomb.
        
        Args:
            model: Bomb model identifier
            hardware_interface: Interface to neuromorphic hardware
        """
        # Define specifications based on model
        if model == "GBU-31":
            specs = WeaponSpecs(
                weight=925.0,
                volume={"length": 3.9, "diameter": 0.457},
                power_requirements=200.0,
                mounting_points=["external_hardpoint", "internal_bay"],
                weapon_type=WeaponType.BOMB,
                range=28.0,
                payload_weight=429.0,
                guidance_system="gps_ins"
            )
        elif model == "GBU-39":
            specs = WeaponSpecs(
                weight=129.0,
                volume={"length": 1.8, "diameter": 0.19},
                power_requirements=150.0,
                mounting_points=["external_hardpoint", "internal_bay"],
                weapon_type=WeaponType.BOMB,
                range=110.0,
                payload_weight=17.0,
                guidance_system="gps_ins"
            )
        else:
            raise ValueError(f"Unknown bomb model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
    
    def calculate_impact(self) -> Dict[str, float]:
        """Calculate the impact of this bomb on UCAV performance."""
        base_impact = super().calculate_impact()
        
        # Add bomb-specific impacts
        if self.model == "GBU-31":
            base_impact["drag_coefficient"] = 0.12
        elif self.model == "GBU-39":
            base_impact["drag_coefficient"] = 0.05
            
        return base_impact
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using neuromorphic computing for bomb guidance."""
        base_result = super().process_data(input_data)
        
        # Add bomb-specific processing
        if input_data.get("computation") == "trajectory_calculation":
            # Enhanced trajectory calculation for bombs
            base_result["wind_compensation"] = True
            base_result["precision"] = 3.0  # CEP in meters
            
        return base_result


class GunSystem(ConventionalWeapon):
    """Aircraft gun system."""
    
    def __init__(self, model: str, hardware_interface=None):
        """
        Initialize an aircraft gun system.
        
        Args:
            model: Gun model identifier
            hardware_interface: Interface to neuromorphic hardware
        """
        # Define specifications based on model
        if model == "M61A2":
            specs = WeaponSpecs(
                weight=92.0,
                volume={"length": 1.9, "width": 0.3, "height": 0.3},
                power_requirements=800.0,
                mounting_points=["fuselage"],
                weapon_type=WeaponType.GUN,
                range=6.0,
                payload_weight=100.0,  # Ammunition weight
                guidance_system="ballistic"
            )
        else:
            raise ValueError(f"Unknown gun model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        self.ammunition = 500  # Rounds of ammunition
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Fire the gun.
        
        Args:
            target_data: Data about the target
            
        Returns:
            Success status
        """
        if not self.status["armed"] or self.ammunition <= 0:
            return False
            
        # Calculate number of rounds to fire
        rounds = min(target_data.get("burst_length", 50), self.ammunition)
        self.ammunition -= rounds
        
        self.status["deployment_progress"] = 100.0
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the gun."""
        status = super().get_status()
        status["ammunition_remaining"] = self.ammunition
        return status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using neuromorphic computing for gun aiming."""
        base_result = super().process_data(input_data)
        
        # Add gun-specific processing
        if input_data.get("computation") == "target_validation":
            # Enhanced target validation for guns
            base_result["lead_angle"] = 2.5  # Degrees
            base_result["recommended_burst"] = 50  # Rounds
            
        return base_result
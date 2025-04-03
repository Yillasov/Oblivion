"""
Factory module for creating landing gear specifications.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional
from src.landing_gear.base import LandingGearSpecs
from src.landing_gear.types import LandingGearType


def create_landing_gear_specs(
    gear_type: LandingGearType,
    weight: float,
    dimensions: Dict[str, float],
    power_requirements: float,
    max_load_capacity: float,
    deployment_time: float,
    retraction_time: float,
    materials: Optional[Dict[str, str]] = None,
    ground_clearance: float = 0.3,
    shock_absorption: float = 0.7,
    max_landing_speed: float = 100.0,
    type_specific_params: Optional[Dict[str, Any]] = None
) -> LandingGearSpecs:
    """
    Create landing gear specifications with simplified parameters.
    
    Args:
        gear_type: Type of landing gear
        weight: Weight in kg
        dimensions: Dimensions in meters (dict with keys like 'length', 'width', 'height')
        power_requirements: Power requirements in watts
        max_load_capacity: Maximum load capacity in kg
        deployment_time: Time to deploy in seconds
        retraction_time: Time to retract in seconds
        materials: Dictionary of materials used (component -> material)
        ground_clearance: Ground clearance in meters
        shock_absorption: Shock absorption rating (0-1 scale)
        max_landing_speed: Maximum safe landing speed in km/h
        type_specific_params: Additional parameters specific to the gear type
        
    Returns:
        LandingGearSpecs: Configured landing gear specifications
    """
    # Initialize materials dictionary if not provided
    if materials is None:
        materials = {
            "structure": "aluminum_alloy",
            "hydraulics": "steel",
            "bearings": "titanium",
            "shock_absorbers": "composite"
        }
    
    # Initialize type-specific parameters if not provided
    if type_specific_params is None:
        type_specific_params = {}
    
    # Create base specs
    specs = LandingGearSpecs(
        weight=weight,
        dimensions=dimensions,
        power_requirements=power_requirements,
        max_load_capacity=max_load_capacity,
        gear_type=gear_type,
        deployment_time=deployment_time,
        retraction_time=retraction_time,
        materials=materials,
        ground_clearance=ground_clearance,
        shock_absorption=shock_absorption,
        max_landing_speed=max_landing_speed
    )
    
    # Add type-specific parameters
    if gear_type == LandingGearType.RETRACTABLE_MORPHING:
        specs.morphing_capabilities = type_specific_params
    elif gear_type == LandingGearType.ELECTROMAGNETIC_CATAPULT:
        specs.electromagnetic_specs = type_specific_params
    elif gear_type == LandingGearType.VTOL_ROTORS:
        specs.vtol_specs = type_specific_params
    elif gear_type == LandingGearType.AIR_CUSHION:
        specs.cushion_specs = type_specific_params
    elif gear_type == LandingGearType.ADAPTIVE_SHOCK_ABSORBING:
        specs.adaptive_shock_specs = type_specific_params
    elif gear_type == LandingGearType.MAGNETIC_LEVITATION:
        specs.magnetic_specs = type_specific_params
    elif gear_type == LandingGearType.INFLATABLE:
        specs.inflatable_specs = type_specific_params
    elif gear_type == LandingGearType.PARACHUTE_ASSISTED:
        specs.parachute_specs = type_specific_params
    elif gear_type == LandingGearType.AUTONOMOUS_SKID:
        specs.skid_specs = type_specific_params
    elif gear_type == LandingGearType.ROCKET_ASSISTED:
        specs.rocket_specs = type_specific_params
    
    return specs


# Example specifications for different landing gear types

def create_retractable_morphing_specs() -> LandingGearSpecs:
    """Create specifications for retractable morphing landing gear."""
    return create_landing_gear_specs(
        gear_type=LandingGearType.RETRACTABLE_MORPHING,
        weight=120.5,
        dimensions={"length": 1.8, "width": 0.5, "height": 0.7},
        power_requirements=1200.0,
        max_load_capacity=5000.0,
        deployment_time=3.5,
        retraction_time=4.2,
        materials={
            "structure": "carbon_fiber_composite",
            "actuators": "titanium_alloy",
            "joints": "high_strength_steel",
            "morphing_elements": "shape_memory_alloy"
        },
        ground_clearance=0.35,
        shock_absorption=0.85,
        max_landing_speed=120.0,
        type_specific_params={
            "morphing_range_degrees": 45.0,
            "morphing_speed": 15.0,  # degrees per second
            "morphing_power": 800.0,  # watts during morphing
            "configurations": ["takeoff", "cruise", "landing", "storage"]
        }
    )


def create_vtol_rotors_specs() -> LandingGearSpecs:
    """Create specifications for VTOL rotors landing gear."""
    return create_landing_gear_specs(
        gear_type=LandingGearType.VTOL_ROTORS,
        weight=85.0,
        dimensions={"length": 0.8, "width": 0.8, "height": 0.4},
        power_requirements=3500.0,
        max_load_capacity=4200.0,
        deployment_time=2.0,
        retraction_time=2.5,
        materials={
            "structure": "aluminum_lithium_alloy",
            "rotors": "carbon_fiber",
            "motors": "neodymium_magnets",
            "controllers": "silicon_carbide"
        },
        ground_clearance=0.4,
        shock_absorption=0.6,
        max_landing_speed=80.0,
        type_specific_params={
            "rotor_count": 4,
            "rotor_diameter": 0.3,  # meters
            "max_rpm": 12000,
            "thrust_per_rotor": 1200.0,  # Newtons
            "control_response_time": 0.05  # seconds
        }
    )
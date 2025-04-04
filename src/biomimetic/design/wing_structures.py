#!/usr/bin/env python3
"""
Biological Wing Structures Library

This module provides detailed reference models for bird and bat wing structures
to support biomimetic design of flexible and articulated wing systems.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.biomimetic.design.principles import BiologicalReference, BiologicalInspiration

logger = get_logger("wing_structures")


class WingType(Enum):
    """Types of biological wing structures."""
    BIRD_ELLIPTICAL = "bird_elliptical"  # High maneuverability (forest birds)
    BIRD_HIGH_ASPECT = "bird_high_aspect"  # Soaring (albatross, gulls)
    BIRD_HIGH_SPEED = "bird_high_speed"  # Fast flight (falcons, swifts)
    BIRD_SLOTTED = "bird_slotted"  # Slow flight, high lift (eagles, vultures)
    BAT_MEMBRANE = "bat_membrane"  # Standard bat wing
    BAT_ARTICULATED = "bat_articulated"  # Highly articulated bat wing


@dataclass
class JointStructure:
    """Representation of a wing joint."""
    name: str
    position: Tuple[float, float, float]  # Normalized position (x, y, z)
    rotation_axes: List[Tuple[float, float, float]]  # Primary rotation axes
    range_of_motion: List[Tuple[float, float]]  # Min/max angles in degrees
    stiffness: float  # 0.0 = completely flexible, 1.0 = rigid
    actuation_type: str  # "passive", "active", "hybrid"
    biological_function: str  # Description of biological function


@dataclass
class WingStructure:
    """Detailed model of a biological wing structure."""
    wing_type: WingType
    species_examples: List[str]
    aspect_ratio: float
    planform_area_ratio: float  # Ratio of wing area to body size
    joints: List[JointStructure]
    membrane_properties: Optional[Dict[str, Any]] = None
    feather_properties: Optional[Dict[str, Any]] = None
    bone_structure: Optional[Dict[str, List[str]]] = None
    muscle_arrangement: Optional[Dict[str, List[str]]] = None
    aerodynamic_features: Dict[str, Any] = field(default_factory=dict)
    scaling_laws: Dict[str, str] = field(default_factory=dict)
    performance_characteristics: Dict[str, float] = field(default_factory=dict)


class WingStructureLibrary:
    """Library of biological wing structure reference models."""
    
    def __init__(self):
        """Initialize the wing structure library."""
        self.structures: Dict[str, WingStructure] = {}
        self._initialize_bird_wing_structures()
        self._initialize_bat_wing_structures()
        logger.info(f"Initialized wing structure library with {len(self.structures)} models")
    
    def _initialize_bird_wing_structures(self):
        """Initialize bird wing structure models."""
        # High-aspect ratio soaring wing (albatross)
        self.structures["albatross"] = WingStructure(
            wing_type=WingType.BIRD_HIGH_ASPECT,
            species_examples=["Wandering Albatross", "Royal Albatross"],
            aspect_ratio=15.0,
            planform_area_ratio=0.65,
            joints=[
                JointStructure(
                    name="shoulder",
                    position=(0.1, 0.0, 0.0),
                    rotation_axes=[(0, 0, 1), (1, 0, 0)],
                    range_of_motion=[(-20, 70), (-10, 20)],
                    stiffness=0.7,
                    actuation_type="active",
                    biological_function="Primary wing elevation/depression and protraction/retraction"
                ),
                JointStructure(
                    name="elbow",
                    position=(0.4, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0)],
                    range_of_motion=[(-5, 90)],
                    stiffness=0.6,
                    actuation_type="active",
                    biological_function="Wing folding and partial control of wing camber"
                ),
                JointStructure(
                    name="wrist",
                    position=(0.7, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0), (1, 0, 0)],
                    range_of_motion=[(-10, 80), (-5, 15)],
                    stiffness=0.5,
                    actuation_type="active",
                    biological_function="Fine control of wingtip and adaptation to gusts"
                )
            ],
            feather_properties={
                "primary_count": 10,
                "primary_aspect_ratio": 8.0,
                "secondary_count": 20,
                "secondary_aspect_ratio": 4.0,
                "feather_overlap": 0.3,  # 30% overlap
                "feather_flexibility": 0.4  # Moderately flexible
            },
            bone_structure={
                "humerus": ["shoulder", "elbow"],
                "radius_ulna": ["elbow", "wrist"],
                "carpometacarpus": ["wrist", "digit_joint"],
                "digit": ["digit_joint", "wingtip"]
            },
            aerodynamic_features={
                "wingtip_slots": False,
                "alula": False,
                "wing_camber": 0.04,  # 4% camber
                "twist_angle": 5.0  # degrees
            },
            scaling_laws={
                "wingspan": "mass^(1/3)",
                "wing_area": "mass^(2/3)",
                "wing_loading": "mass^(1/3)"
            },
            performance_characteristics={
                "glide_ratio": 25.0,
                "minimum_sink_rate": 0.6,  # m/s
                "turn_radius": 50.0,  # m
                "energy_efficiency": 0.9  # Relative scale 0-1
            }
        )
        
        # High-speed wing (peregrine falcon)
        self.structures["peregrine_falcon"] = WingStructure(
            wing_type=WingType.BIRD_HIGH_SPEED,
            species_examples=["Peregrine Falcon", "Common Swift"],
            aspect_ratio=7.5,
            planform_area_ratio=0.45,
            joints=[
                JointStructure(
                    name="shoulder",
                    position=(0.1, 0.0, 0.0),
                    rotation_axes=[(0, 0, 1), (1, 0, 0)],
                    range_of_motion=[(-10, 60), (-15, 15)],
                    stiffness=0.8,
                    actuation_type="active",
                    biological_function="Primary wing control for high-speed flight"
                ),
                JointStructure(
                    name="elbow",
                    position=(0.35, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0)],
                    range_of_motion=[(-5, 70)],
                    stiffness=0.7,
                    actuation_type="active",
                    biological_function="Wing folding and shape control"
                ),
                JointStructure(
                    name="wrist",
                    position=(0.65, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0), (1, 0, 0)],
                    range_of_motion=[(-5, 60), (-10, 10)],
                    stiffness=0.6,
                    actuation_type="active",
                    biological_function="Fine control for high-speed maneuvering"
                )
            ],
            feather_properties={
                "primary_count": 9,
                "primary_aspect_ratio": 6.0,
                "secondary_count": 12,
                "secondary_aspect_ratio": 3.0,
                "feather_overlap": 0.25,
                "feather_flexibility": 0.3  # Stiffer feathers for high-speed
            },
            aerodynamic_features={
                "wingtip_slots": True,
                "alula": True,
                "wing_camber": 0.03,
                "twist_angle": 3.0
            },
            performance_characteristics={
                "glide_ratio": 15.0,
                "minimum_sink_rate": 1.2,  # m/s
                "turn_radius": 20.0,  # m
                "energy_efficiency": 0.7,
                "max_speed": 389.0  # km/h (diving)
            }
        )
        
        # Highly maneuverable wing (sparrowhawk)
        self.structures["sparrowhawk"] = WingStructure(
            wing_type=WingType.BIRD_ELLIPTICAL,
            species_examples=["Eurasian Sparrowhawk", "Cooper's Hawk", "Goshawk"],
            aspect_ratio=5.2,
            planform_area_ratio=0.55,
            joints=[
                JointStructure(
                    name="shoulder",
                    position=(0.1, 0.0, 0.0),
                    rotation_axes=[(0, 0, 1), (1, 0, 0), (0, 1, 0)],
                    range_of_motion=[(-30, 80), (-20, 20), (-10, 10)],
                    stiffness=0.6,
                    actuation_type="active",
                    biological_function="Multi-axis control for forest maneuvering"
                ),
                JointStructure(
                    name="elbow",
                    position=(0.4, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0), (1, 0, 0)],
                    range_of_motion=[(-10, 100), (-5, 5)],
                    stiffness=0.5,
                    actuation_type="active",
                    biological_function="Wing shape control for tight turns"
                ),
                JointStructure(
                    name="wrist",
                    position=(0.7, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0), (1, 0, 0), (0, 0, 1)],
                    range_of_motion=[(-15, 90), (-10, 10), (-5, 5)],
                    stiffness=0.4,
                    actuation_type="active",
                    biological_function="Fine control for rapid direction changes"
                )
            ],
            feather_properties={
                "primary_count": 10,
                "primary_aspect_ratio": 4.5,
                "secondary_count": 12,
                "secondary_aspect_ratio": 2.5,
                "feather_overlap": 0.35,
                "feather_flexibility": 0.5  # More flexible for maneuverability
            },
            aerodynamic_features={
                "wingtip_slots": True,
                "alula": True,
                "wing_camber": 0.05,
                "twist_angle": 6.0
            },
            performance_characteristics={
                "glide_ratio": 12.0,
                "minimum_sink_rate": 0.9,  # m/s
                "turn_radius": 5.0,  # m
                "energy_efficiency": 0.6,
                "maneuverability": 0.9  # Relative scale 0-1
            }
        )
    
    def _initialize_bat_wing_structures(self):
        """Initialize bat wing structure models."""
        # Standard bat wing (common pipistrelle)
        self.structures["pipistrelle_bat"] = WingStructure(
            wing_type=WingType.BAT_MEMBRANE,
            species_examples=["Common Pipistrelle", "Soprano Pipistrelle"],
            aspect_ratio=6.8,
            planform_area_ratio=0.7,
            joints=[
                JointStructure(
                    name="shoulder",
                    position=(0.05, 0.0, 0.0),
                    rotation_axes=[(0, 0, 1), (1, 0, 0), (0, 1, 0)],
                    range_of_motion=[(-20, 90), (-30, 30), (-10, 10)],
                    stiffness=0.5,
                    actuation_type="active",
                    biological_function="Primary wing control"
                ),
                JointStructure(
                    name="elbow",
                    position=(0.25, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0)],
                    range_of_motion=[(-10, 160)],
                    stiffness=0.4,
                    actuation_type="active",
                    biological_function="Wing extension and folding"
                ),
                JointStructure(
                    name="wrist",
                    position=(0.45, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0), (1, 0, 0)],
                    range_of_motion=[(-20, 90), (-15, 15)],
                    stiffness=0.3,
                    actuation_type="active",
                    biological_function="Distal wing control"
                ),
                JointStructure(
                    name="metacarpal_joint_1",
                    position=(0.55, 0.1, 0.0),
                    rotation_axes=[(0, 1, 0)],
                    range_of_motion=[(-5, 40)],
                    stiffness=0.2,
                    actuation_type="active",
                    biological_function="Digit 1 control"
                ),
                JointStructure(
                    name="metacarpal_joint_2",
                    position=(0.6, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0)],
                    range_of_motion=[(-10, 60)],
                    stiffness=0.2,
                    actuation_type="active",
                    biological_function="Digit 2 control"
                ),
                JointStructure(
                    name="metacarpal_joint_3",
                    position=(0.65, -0.1, 0.0),
                    rotation_axes=[(0, 1, 0)],
                    range_of_motion=[(-15, 70)],
                    stiffness=0.2,
                    actuation_type="active",
                    biological_function="Digit 3 control"
                ),
                JointStructure(
                    name="phalangeal_joint_1",
                    position=(0.75, 0.15, 0.0),
                    rotation_axes=[(0, 1, 0)],
                    range_of_motion=[(-5, 30)],
                    stiffness=0.15,
                    actuation_type="active",
                    biological_function="Fine control of wing tip shape"
                ),
                JointStructure(
                    name="phalangeal_joint_2",
                    position=(0.85, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0)],
                    range_of_motion=[(-10, 40)],
                    stiffness=0.15,
                    actuation_type="active",
                    biological_function="Fine control of trailing edge"
                )
            ],
            membrane_properties={
                "elasticity": 0.6,  # 0-1 scale
                "thickness_mm": 0.1,
                "anisotropy": 0.7,  # Directional elasticity
                "fiber_orientation": "spanwise",
                "attachment_points": ["arm", "digits", "leg", "tail"]
            },
            bone_structure={
                "humerus": ["shoulder", "elbow"],
                "radius_ulna": ["elbow", "wrist"],
                "metacarpals": ["wrist", "metacarpal_joints"],
                "phalanges": ["metacarpal_joints", "phalangeal_joints", "wingtip"]
            },
            muscle_arrangement={
                "pectoralis": ["body", "humerus"],
                "serratus": ["body", "scapula"],
                "biceps": ["shoulder", "radius"],
                "triceps": ["humerus", "ulna"],
                "digital_flexors": ["forearm", "digits"]
            },
            aerodynamic_features={
                "camber_control": "active",
                "membrane_tension": "muscular",
                "wing_loading": 8.5,  # N/m²
                "trailing_edge_control": "individual_digits"
            },
            performance_characteristics={
                "glide_ratio": 8.0,
                "minimum_sink_rate": 0.5,  # m/s
                "turn_radius": 0.3,  # m
                "energy_efficiency": 0.7,
                "maneuverability": 0.95,
                "hover_capability": 0.8  # Relative scale 0-1
            }
        )
        
        # Highly articulated bat wing (vampire bat)
        self.structures["vampire_bat"] = WingStructure(
            wing_type=WingType.BAT_ARTICULATED,
            species_examples=["Common Vampire Bat", "Hairy-legged Vampire Bat"],
            aspect_ratio=5.2,
            planform_area_ratio=0.6,
            joints=[
                JointStructure(
                    name="shoulder",
                    position=(0.05, 0.0, 0.0),
                    rotation_axes=[(0, 0, 1), (1, 0, 0), (0, 1, 0)],
                    range_of_motion=[(-30, 100), (-40, 40), (-20, 20)],
                    stiffness=0.4,
                    actuation_type="active",
                    biological_function="Multi-axis control for quadrupedal locomotion"
                ),
                JointStructure(
                    name="elbow",
                    position=(0.2, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0), (0, 0, 1)],
                    range_of_motion=[(-10, 170), (-10, 10)],
                    stiffness=0.3,
                    actuation_type="active",
                    biological_function="Wing folding and terrestrial locomotion"
                ),
                JointStructure(
                    name="wrist",
                    position=(0.4, 0.0, 0.0),
                    rotation_axes=[(0, 1, 0), (1, 0, 0), (0, 0, 1)],
                    range_of_motion=[(-30, 100), (-20, 20), (-15, 15)],
                    stiffness=0.25,
                    actuation_type="active",
                    biological_function="Complex wing control and thumb positioning"
                ),
                # Additional 7 joints for digits with greater ranges of motion
                # than standard bat wing, enabling quadrupedal locomotion
            ],
            membrane_properties={
                "elasticity": 0.7,
                "thickness_mm": 0.08,
                "anisotropy": 0.8,
                "fiber_orientation": "radial",
                "attachment_points": ["arm", "digits", "leg", "minimal_tail"]
            },
            muscle_arrangement={
                "pectoralis": ["body", "humerus"],
                "serratus": ["body", "scapula"],
                "biceps": ["shoulder", "radius"],
                "triceps": ["humerus", "ulna"],
                "digital_flexors": ["forearm", "digits"],
                "digital_extensors": ["forearm", "digits"],
                "thumb_muscles": ["wrist", "thumb"]
            },
            aerodynamic_features={
                "camber_control": "active",
                "membrane_tension": "muscular",
                "wing_loading": 10.2,  # N/m²
                "trailing_edge_control": "individual_digits",
                "thumb_control": "independent"
            },
            performance_characteristics={
                "glide_ratio": 6.0,
                "minimum_sink_rate": 0.6,  # m/s
                "turn_radius": 0.2,  # m
                "energy_efficiency": 0.6,
                "maneuverability": 0.98,
                "hover_capability": 0.7,
                "terrestrial_mobility": 0.9  # Relative scale 0-1
            }
        )
    
    def get_structure(self, structure_id: str) -> Optional[WingStructure]:
        """
        Get a wing structure by ID.
        
        Args:
            structure_id: ID of the wing structure
            
        Returns:
            WingStructure if found, None otherwise
        """
        return self.structures.get(structure_id)
    
    def get_structures_by_type(self, wing_type: WingType) -> List[WingStructure]:
        """
        Get wing structures by type.
        
        Args:
            wing_type: Type of wing structure
            
        Returns:
            List of matching wing structures
        """
        return [s for s in self.structures.values() if s.wing_type == wing_type]
    
    def create_biological_reference(self, structure_id: str) -> Optional[BiologicalReference]:
        """
        Create a BiologicalReference from a wing structure.
        
        Args:
            structure_id: ID of the wing structure
            
        Returns:
            BiologicalReference if structure found, None otherwise
        """
        structure = self.get_structure(structure_id)
        if not structure:
            return None
            
        # Determine inspiration type
        if structure.wing_type in [WingType.BIRD_ELLIPTICAL, WingType.BIRD_HIGH_ASPECT, 
                                  WingType.BIRD_HIGH_SPEED, WingType.BIRD_SLOTTED]:
            inspiration = BiologicalInspiration.BIRD
        elif structure.wing_type in [WingType.BAT_MEMBRANE, WingType.BAT_ARTICULATED]:
            inspiration = BiologicalInspiration.BAT
        else:
            inspiration = BiologicalInspiration.BIRD
        
        # Extract key features
        key_features = []
        if structure.wing_type == WingType.BIRD_HIGH_ASPECT:
            key_features.extend(["efficient_gliding", "high_aspect_ratio", "low_energy_flight"])
        elif structure.wing_type == WingType.BIRD_HIGH_SPEED:
            key_features.extend(["high_speed_flight", "streamlined_body"])
        elif structure.wing_type == WingType.BIRD_ELLIPTICAL:
            key_features.extend(["high_maneuverability", "bird_wing"])
        elif structure.wing_type == WingType.BAT_MEMBRANE:
            key_features.extend(["bat_wing", "membrane_wing", "high_maneuverability"])
        elif structure.wing_type == WingType.BAT_ARTICULATED:
            key_features.extend(["bat_wing", "articulated_joints", "multi_functional"])
        
        # Add common features
        key_features.append("wing_morphing")
        
        # Create biological reference
        return BiologicalReference(
            name=structure_id.replace("_", " ").title(),
            species=structure.species_examples[0] if structure.species_examples else "Unknown",
            inspiration_type=inspiration,
            key_features=key_features,
            performance_metrics=structure.performance_characteristics,
            morphological_data={
                "aspect_ratio": structure.aspect_ratio,
                "planform_area_ratio": structure.planform_area_ratio,
                "joint_count": len(structure.joints),
                "wing_type": structure.wing_type.value
            },
            behavioral_data={
                "flight_style": structure.wing_type.value,
                "joint_mobility": {j.name: j.range_of_motion for j in structure.joints},
                "actuation_types": {j.name: j.actuation_type for j in structure.joints}
            }
        )


# Integration function to add wing structures to the reference database
def add_wing_structures_to_database(database):
    """
    Add wing structures to the biological reference database.
    
    Args:
        database: BiologicalReferenceDatabase instance
    """
    wing_library = WingStructureLibrary()
    
    # Convert wing structures to biological references and add to database
    for structure_id in wing_library.structures:
        bio_ref = wing_library.create_biological_reference(structure_id)
        if bio_ref:
            database.add_reference(structure_id, bio_ref)
            
    logger.info(f"Added {len(wing_library.structures)} wing structures to reference database")
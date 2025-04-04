#!/usr/bin/env python3
"""
Insect Flight Mechanics Models

This module provides detailed models of insect flight mechanics for biomimetic design
inspiration, focusing on wing kinematics, unsteady aerodynamics, and control strategies.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.biomimetic.design.principles import BiologicalReference, BiologicalInspiration
from src.biomimetic.design.wing_structures import JointStructure, WingStructure, WingType

logger = get_logger("insect_flight")


class InsectFlightType(Enum):
    """Types of insect flight mechanics."""
    DIRECT_FLIGHT = "direct_flight"  # Direct flight muscles (beetles, bees)
    INDIRECT_FLIGHT = "indirect_flight"  # Indirect flight muscles (flies, wasps)
    CLAP_FLING = "clap_fling"  # Clap and fling mechanism (small insects)
    DELAYED_STALL = "delayed_stall"  # Delayed stall mechanism (dragonflies)
    ROTATIONAL_CIRCULATION = "rotational_circulation"  # Rotational circulation (fruit flies)


@dataclass
class WingKinematics:
    """Detailed model of insect wing kinematics."""
    stroke_amplitude: float  # Degrees
    stroke_plane_angle: float  # Degrees from horizontal
    wing_beat_frequency: float  # Hz
    upstroke_ratio: float  # Ratio of upstroke to downstroke duration
    phase_shift: float  # Phase shift between forewings and hindwings (if applicable)
    rotation_timing: Dict[str, float]  # Timing of pronation/supination
    deviation_angle: float  # Out-of-plane motion
    wing_trajectory: str  # Description of wing path (figure-8, oval, etc.)


@dataclass
class UnsteadyAerodynamics:
    """Model of unsteady aerodynamic mechanisms in insect flight."""
    leading_edge_vortex: Dict[str, Any]  # LEV characteristics
    rotational_lift: Dict[str, float]  # Rotational lift coefficients
    wake_capture: Dict[str, Any]  # Wake capture effects
    clap_fling: Optional[Dict[str, Any]] = None  # Clap and fling parameters
    added_mass: Dict[str, float] = field(default_factory=dict)  # Added mass effects
    wing_wing_interaction: Optional[Dict[str, Any]] = None  # Interaction between wings


@dataclass
class InsectFlightModel:
    """Comprehensive model of insect flight mechanics."""
    species: str
    flight_type: InsectFlightType
    body_mass: float  # mg
    wing_span: float  # mm
    aspect_ratio: float
    wing_loading: float  # N/m²
    wing_kinematics: WingKinematics
    unsteady_aerodynamics: UnsteadyAerodynamics
    control_strategy: Dict[str, Any]
    performance_metrics: Dict[str, float]
    scaling_laws: Dict[str, str] = field(default_factory=dict)
    wing_structure: Optional[WingStructure] = None


class InsectFlightLibrary:
    """Library of insect flight mechanics models."""
    
    def __init__(self):
        """Initialize the insect flight library."""
        self.models: Dict[str, InsectFlightModel] = {}
        self._initialize_models()
        logger.info(f"Initialized insect flight library with {len(self.models)} models")
    
    def _initialize_models(self):
        """Initialize insect flight mechanics models."""
        # Fruit fly (Drosophila) model
        self.models["fruit_fly"] = InsectFlightModel(
            species="Drosophila melanogaster",
            flight_type=InsectFlightType.INDIRECT_FLIGHT,
            body_mass=1.1,  # mg
            wing_span=6.0,  # mm
            aspect_ratio=2.8,
            wing_loading=1.2,  # N/m²
            wing_kinematics=WingKinematics(
                stroke_amplitude=150.0,  # degrees
                stroke_plane_angle=45.0,  # degrees from horizontal
                wing_beat_frequency=200.0,  # Hz
                upstroke_ratio=0.45,  # Slightly faster downstroke
                phase_shift=0.0,  # Single pair of wings
                rotation_timing={
                    "pronation_start": 0.8,  # Fraction of cycle
                    "pronation_duration": 0.15,  # Fraction of cycle
                    "supination_start": 0.3,  # Fraction of cycle
                    "supination_duration": 0.15  # Fraction of cycle
                },
                deviation_angle=10.0,  # degrees
                wing_trajectory="figure-8"
            ),
            unsteady_aerodynamics=UnsteadyAerodynamics(
                leading_edge_vortex={
                    "strength": 0.8,  # Relative strength
                    "stability": "stable",
                    "attachment_point": 0.7  # Fraction of chord
                },
                rotational_lift={
                    "coefficient": 1.5,
                    "timing_sensitivity": 0.8  # Sensitivity to rotation timing
                },
                wake_capture={
                    "strength": 0.6,  # Relative strength
                    "duration": 0.1  # Fraction of cycle
                },
                added_mass={
                    "coefficient": 0.3
                }
            ),
            control_strategy={
                "roll": "asymmetric_stroke_amplitude",
                "pitch": "stroke_plane_adjustment",
                "yaw": "wing_rotation_timing",
                "hover_stability": "passive_damping"
            },
            performance_metrics={
                "hover_efficiency": 0.7,  # Relative scale
                "maneuverability": 0.9,  # Relative scale
                "max_acceleration": 10.0,  # m/s²
                "turn_rate": 2000.0,  # deg/s
                "max_lift_coefficient": 1.8
            },
            scaling_laws={
                "wing_beat_frequency": "mass^(-0.33)",
                "power_requirement": "mass^(0.67)"
            }
        )
        
        # Hawkmoth model
        self.models["hawkmoth"] = InsectFlightModel(
            species="Manduca sexta",
            flight_type=InsectFlightType.INDIRECT_FLIGHT,
            body_mass=1500.0,  # mg
            wing_span=100.0,  # mm
            aspect_ratio=5.0,
            wing_loading=5.8,  # N/m²
            wing_kinematics=WingKinematics(
                stroke_amplitude=110.0,  # degrees
                stroke_plane_angle=30.0,  # degrees from horizontal
                wing_beat_frequency=25.0,  # Hz
                upstroke_ratio=0.5,  # Equal upstroke and downstroke
                phase_shift=0.0,  # Single pair of wings
                rotation_timing={
                    "pronation_start": 0.85,  # Fraction of cycle
                    "pronation_duration": 0.1,  # Fraction of cycle
                    "supination_start": 0.35,  # Fraction of cycle
                    "supination_duration": 0.1  # Fraction of cycle
                },
                deviation_angle=5.0,  # degrees
                wing_trajectory="oval"
            ),
            unsteady_aerodynamics=UnsteadyAerodynamics(
                leading_edge_vortex={
                    "strength": 0.9,  # Relative strength
                    "stability": "stable",
                    "attachment_point": 0.6  # Fraction of chord
                },
                rotational_lift={
                    "coefficient": 1.2,
                    "timing_sensitivity": 0.6  # Sensitivity to rotation timing
                },
                wake_capture={
                    "strength": 0.4,  # Relative strength
                    "duration": 0.08  # Fraction of cycle
                },
                added_mass={
                    "coefficient": 0.2
                }
            ),
            control_strategy={
                "roll": "asymmetric_stroke_amplitude",
                "pitch": "stroke_plane_adjustment",
                "yaw": "asymmetric_angle_of_attack",
                "hover_stability": "active_control"
            },
            performance_metrics={
                "hover_efficiency": 0.6,  # Relative scale
                "maneuverability": 0.7,  # Relative scale
                "max_acceleration": 5.0,  # m/s²
                "turn_rate": 800.0,  # deg/s
                "max_lift_coefficient": 1.5,
                "forward_speed": 5.0  # m/s
            },
            scaling_laws={
                "wing_beat_frequency": "mass^(-0.33)",
                "power_requirement": "mass^(0.67)"
            }
        )
        
        # Dragonfly model
        self.models["dragonfly"] = InsectFlightModel(
            species="Anax junius",
            flight_type=InsectFlightType.DELAYED_STALL,
            body_mass=750.0,  # mg
            wing_span=110.0,  # mm (combined wingspan)
            aspect_ratio=12.0,  # High aspect ratio wings
            wing_loading=3.2,  # N/m²
            wing_kinematics=WingKinematics(
                stroke_amplitude=60.0,  # degrees (smaller amplitude)
                stroke_plane_angle=20.0,  # degrees from horizontal
                wing_beat_frequency=30.0,  # Hz
                upstroke_ratio=0.5,  # Equal upstroke and downstroke
                phase_shift=0.5,  # 180 degree phase shift between fore and hind wings
                rotation_timing={
                    "pronation_start": 0.9,  # Fraction of cycle
                    "pronation_duration": 0.1,  # Fraction of cycle
                    "supination_start": 0.4,  # Fraction of cycle
                    "supination_duration": 0.1  # Fraction of cycle
                },
                deviation_angle=3.0,  # degrees
                wing_trajectory="figure-8"
            ),
            unsteady_aerodynamics=UnsteadyAerodynamics(
                leading_edge_vortex={
                    "strength": 0.7,  # Relative strength
                    "stability": "stable",
                    "attachment_point": 0.5  # Fraction of chord
                },
                rotational_lift={
                    "coefficient": 1.0,
                    "timing_sensitivity": 0.5  # Sensitivity to rotation timing
                },
                wake_capture={
                    "strength": 0.8,  # Relative strength
                    "duration": 0.15  # Fraction of cycle
                },
                wing_wing_interaction={
                    "forewing_hindwing_distance": 0.3,  # Chord lengths
                    "interaction_strength": 0.7,  # Relative strength
                    "optimal_phase": 0.5  # Optimal phase difference
                },
                added_mass={
                    "coefficient": 0.15
                }
            ),
            control_strategy={
                "roll": "asymmetric_stroke_amplitude",
                "pitch": "forewing_hindwing_phase",
                "yaw": "wing_couple_modulation",
                "hover_stability": "active_control",
                "forward_flight": "wing_phasing"
            },
            performance_metrics={
                "hover_efficiency": 0.5,  # Relative scale
                "maneuverability": 0.95,  # Relative scale
                "max_acceleration": 15.0,  # m/s²
                "turn_rate": 1000.0,  # deg/s
                "max_lift_coefficient": 1.4,
                "forward_speed": 10.0,  # m/s
                "glide_ratio": 4.0
            },
            scaling_laws={
                "wing_beat_frequency": "mass^(-0.33)",
                "power_requirement": "mass^(0.67)"
            }
        )
        
        # Bumblebee model
        self.models["bumblebee"] = InsectFlightModel(
            species="Bombus terrestris",
            flight_type=InsectFlightType.INDIRECT_FLIGHT,
            body_mass=200.0,  # mg
            wing_span=25.0,  # mm
            aspect_ratio=6.3,
            wing_loading=6.5,  # N/m²
            wing_kinematics=WingKinematics(
                stroke_amplitude=130.0,  # degrees
                stroke_plane_angle=40.0,  # degrees from horizontal
                wing_beat_frequency=130.0,  # Hz
                upstroke_ratio=0.48,  # Slightly faster downstroke
                phase_shift=0.0,  # Single pair of wings
                rotation_timing={
                    "pronation_start": 0.82,  # Fraction of cycle
                    "pronation_duration": 0.12,  # Fraction of cycle
                    "supination_start": 0.32,  # Fraction of cycle
                    "supination_duration": 0.12  # Fraction of cycle
                },
                deviation_angle=8.0,  # degrees
                wing_trajectory="figure-8"
            ),
            unsteady_aerodynamics=UnsteadyAerodynamics(
                leading_edge_vortex={
                    "strength": 0.95,  # Relative strength
                    "stability": "stable",
                    "attachment_point": 0.65  # Fraction of chord
                },
                rotational_lift={
                    "coefficient": 1.4,
                    "timing_sensitivity": 0.7  # Sensitivity to rotation timing
                },
                wake_capture={
                    "strength": 0.5,  # Relative strength
                    "duration": 0.09  # Fraction of cycle
                },
                added_mass={
                    "coefficient": 0.25
                }
            ),
            control_strategy={
                "roll": "asymmetric_stroke_amplitude",
                "pitch": "stroke_plane_adjustment",
                "yaw": "wing_rotation_timing",
                "hover_stability": "active_control",
                "load_carrying": "increased_stroke_amplitude"
            },
            performance_metrics={
                "hover_efficiency": 0.65,  # Relative scale
                "maneuverability": 0.8,  # Relative scale
                "max_acceleration": 8.0,  # m/s²
                "turn_rate": 900.0,  # deg/s
                "max_lift_coefficient": 1.9,
                "forward_speed": 7.0,  # m/s
                "payload_capacity": 0.8  # Relative to body mass
            },
            scaling_laws={
                "wing_beat_frequency": "mass^(-0.33)",
                "power_requirement": "mass^(0.67)"
            }
        )
    
    def get_model(self, model_id: str) -> Optional[InsectFlightModel]:
        """
        Get an insect flight model by ID.
        
        Args:
            model_id: ID of the insect flight model
            
        Returns:
            InsectFlightModel if found, None otherwise
        """
        return self.models.get(model_id)
    
    def get_models_by_type(self, flight_type: InsectFlightType) -> List[InsectFlightModel]:
        """
        Get insect flight models by type.
        
        Args:
            flight_type: Type of insect flight mechanics
            
        Returns:
            List of matching insect flight models
        """
        return [m for m in self.models.values() if m.flight_type == flight_type]
    
    def create_biological_reference(self, model_id: str) -> Optional[BiologicalReference]:
        """
        Create a BiologicalReference from an insect flight model.
        
        Args:
            model_id: ID of the insect flight model
            
        Returns:
            BiologicalReference if model found, None otherwise
        """
        model = self.get_model(model_id)
        if not model:
            return None
        
        # Extract key features
        key_features = ["insect_flight", "high_maneuverability"]
        
        if model.flight_type == InsectFlightType.CLAP_FLING:
            key_features.append("clap_fling_mechanism")
        elif model.flight_type == InsectFlightType.DELAYED_STALL:
            key_features.append("delayed_stall")
        elif model.flight_type == InsectFlightType.ROTATIONAL_CIRCULATION:
            key_features.append("rotational_circulation")
        
        if model.wing_kinematics.wing_beat_frequency > 100:
            key_features.append("high_frequency_flapping")
        
        if model.performance_metrics.get("hover_efficiency", 0) > 0.6:
            key_features.append("efficient_hovering")
            
        # Create biological reference
        return BiologicalReference(
            name=model_id.replace("_", " ").title(),
            species=model.species,
            inspiration_type=BiologicalInspiration.INSECT,
            key_features=key_features,
            performance_metrics={
                "maneuverability": model.performance_metrics.get("maneuverability", 0),
                "hover_efficiency": model.performance_metrics.get("hover_efficiency", 0),
                "max_acceleration": model.performance_metrics.get("max_acceleration", 0),
                "turn_rate": model.performance_metrics.get("turn_rate", 0),
                "max_lift_coefficient": model.performance_metrics.get("max_lift_coefficient", 0)
            },
            morphological_data={
                "wing_span": model.wing_span,
                "aspect_ratio": model.aspect_ratio,
                "wing_loading": model.wing_loading,
                "wing_beat_frequency": model.wing_kinematics.wing_beat_frequency,
                "stroke_amplitude": model.wing_kinematics.stroke_amplitude
            },
            behavioral_data={
                "flight_type": model.flight_type.value,
                "control_strategy": model.control_strategy,
                "wing_trajectory": model.wing_kinematics.wing_trajectory
            }
        )


def create_insect_wing_structure(model_id: str, library: InsectFlightLibrary) -> Optional[WingStructure]:
    """
    Create a WingStructure from an insect flight model for integration with wing_structures.py.
    
    Args:
        model_id: ID of the insect flight model
        library: InsectFlightLibrary instance
        
    Returns:
        WingStructure if model found, None otherwise
    """
    model = library.get_model(model_id)
    if not model:
        return None
    
    # Define a new wing type for insects
    wing_type = WingType.BIRD_ELLIPTICAL  # Default fallback
    
    # Create joint structures based on insect wing articulation
    joints = []
    
    # Wing base joint (connects to thorax)
    joints.append(JointStructure(
        name="wing_base",
        position=(0.0, 0.0, 0.0),
        rotation_axes=[(0, 0, 1), (1, 0, 0), (0, 1, 0)],
        range_of_motion=[
            (-model.wing_kinematics.stroke_amplitude/2, model.wing_kinematics.stroke_amplitude/2),
            (-20, 20),
            (-model.wing_kinematics.deviation_angle, model.wing_kinematics.deviation_angle)
        ],
        stiffness=0.3,
        actuation_type="active",
        biological_function="Primary wing actuation"
    ))
    
    # Wing flexion joint (passive deformation during flapping)
    joints.append(JointStructure(
        name="wing_flexion",
        position=(0.6, 0.0, 0.0),
        rotation_axes=[(0, 1, 0)],
        range_of_motion=[(-30, 30)],
        stiffness=0.2,
        actuation_type="passive",
        biological_function="Passive wing deformation during flapping"
    ))
    
    # Create wing structure
    return WingStructure(
        wing_type=wing_type,
        species_examples=[model.species],
        aspect_ratio=model.aspect_ratio,
        planform_area_ratio=0.2,  # Typical for insects
        joints=joints,
        membrane_properties={
            "elasticity": 0.8,
            "thickness_mm": 0.01,
            "anisotropy": 0.9,
            "venation_pattern": "radial" if "dragonfly" in model_id else "branching"
        },
        aerodynamic_features={
            "leading_edge_vortex": True,
            "wing_rotation": True,
            "clap_fling": model.flight_type == InsectFlightType.CLAP_FLING,
            "wing_flexibility": 0.8
        },
        performance_characteristics=model.performance_metrics
    )


# Integration function to add insect flight models to the reference database
def add_insect_models_to_database(database):
    """
    Add insect flight models to the biological reference database.
    
    Args:
        database: BiologicalReferenceDatabase instance
    """
    insect_library = InsectFlightLibrary()
    
    # Convert insect flight models to biological references and add to database
    for model_id in insect_library.models:
        bio_ref = insect_library.create_biological_reference(model_id)
        if bio_ref:
            database.add_reference(f"insect_{model_id}", bio_ref)
            
    logger.info(f"Added {len(insect_library.models)} insect flight models to reference database")
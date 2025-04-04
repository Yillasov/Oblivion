#!/usr/bin/env python3
"""
Biomimetic Design Principles Module

This module provides fundamental principles and abstractions for biomimetic design
of UCAVs, inspired by biological flying organisms and their adaptations.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("biomimetic_design")


class BiologicalInspiration(Enum):
    """Sources of biological inspiration for UCAV design."""
    BIRD = "bird"  # Bird-inspired designs (e.g., eagles, falcons)
    BAT = "bat"    # Bat-inspired designs (flexible membranes)
    INSECT = "insect"  # Insect-inspired designs (e.g., dragonflies)
    FLYING_FISH = "flying_fish"  # Flying fish (water-to-air transition)
    SEED = "seed"  # Seed-based designs (e.g., maple seeds, autorotation)


class BiomimeticPrinciple(Enum):
    """Core biomimetic design principles."""
    FORM_FOLLOWS_FUNCTION = "form_follows_function"
    MULTI_FUNCTIONALITY = "multi_functionality"
    MATERIAL_EFFICIENCY = "material_efficiency"
    ADAPTIVE_MORPHOLOGY = "adaptive_morphology"
    SENSORY_INTEGRATION = "sensory_integration"
    ENERGY_EFFICIENCY = "energy_efficiency"
    SELF_ORGANIZATION = "self_organization"


@dataclass
class BiologicalReference:
    """Reference model for a biological organism or system."""
    name: str
    species: str
    inspiration_type: BiologicalInspiration
    key_features: List[str]
    performance_metrics: Dict[str, float]
    morphological_data: Dict[str, Any]
    behavioral_data: Optional[Dict[str, Any]] = None
    references: Optional[List[str]] = None


class BiomimeticDesignFramework:
    """
    Core framework for biomimetic design principles and methodologies.
    
    This class provides the foundation for applying biomimetic design principles
    to UCAV development, including biological reference models and parameter mapping.
    """
    
    def __init__(self):
        """Initialize the biomimetic design framework."""
        self.biological_references: Dict[str, BiologicalReference] = {}
        self.design_principles: Dict[BiomimeticPrinciple, Dict[str, Any]] = {}
        self.parameter_mappings: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with core design principles
        self._initialize_design_principles()
        # Load initial biological references
        self._load_reference_models()
        
        logger.info("Biomimetic design framework initialized")
    
    def _initialize_design_principles(self) -> None:
        """Initialize core biomimetic design principles."""
        self.design_principles = {
            BiomimeticPrinciple.FORM_FOLLOWS_FUNCTION: {
                "description": "Design shape and structure based on functional requirements",
                "application_areas": ["wing_design", "fuselage_shape", "control_surfaces"],
                "metrics": ["lift_to_drag_ratio", "maneuverability", "stability"]
            },
            BiomimeticPrinciple.MULTI_FUNCTIONALITY: {
                "description": "Components serve multiple functions simultaneously",
                "application_areas": ["wing_structures", "sensor_integration", "propulsion"],
                "metrics": ["weight_efficiency", "system_complexity", "redundancy"]
            },
            BiomimeticPrinciple.ADAPTIVE_MORPHOLOGY: {
                "description": "Ability to change shape or properties in response to conditions",
                "application_areas": ["morphing_wings", "adaptive_control_surfaces", "reconfigurable_structures"],
                "metrics": ["adaptation_range", "response_time", "energy_requirements"]
            },
            BiomimeticPrinciple.MATERIAL_EFFICIENCY: {
                "description": "Optimized material usage and distribution",
                "application_areas": ["structural_design", "material_selection", "manufacturing"],
                "metrics": ["strength_to_weight_ratio", "material_utilization", "durability"]
            },
            BiomimeticPrinciple.SENSORY_INTEGRATION: {
                "description": "Seamless integration of sensors into structural components",
                "application_areas": ["distributed_sensing", "structural_health_monitoring", "situational_awareness"],
                "metrics": ["sensor_coverage", "data_integration", "detection_capability"]
            },
            BiomimeticPrinciple.ENERGY_EFFICIENCY: {
                "description": "Minimizing energy consumption through optimized design",
                "application_areas": ["propulsion", "aerodynamics", "power_management"],
                "metrics": ["energy_consumption", "range", "endurance"]
            },
            BiomimeticPrinciple.SELF_ORGANIZATION: {
                "description": "Ability to maintain organization without external control",
                "application_areas": ["autonomous_operation", "fault_tolerance", "swarm_behavior"],
                "metrics": ["autonomy_level", "resilience", "adaptability"]
            }
        }
    
    def _load_reference_models(self) -> None:
        """Load initial biological reference models."""
        # Example reference model: Peregrine Falcon
        self.biological_references["peregrine_falcon"] = BiologicalReference(
            name="Peregrine Falcon",
            species="Falco peregrinus",
            inspiration_type=BiologicalInspiration.BIRD,
            key_features=[
                "high_speed_dive",
                "wing_morphing",
                "streamlined_body",
                "efficient_respiratory_system"
            ],
            performance_metrics={
                "max_speed_kph": 389.0,
                "wing_loading_n_per_sqm": 140.0,
                "aspect_ratio": 2.5,
                "glide_ratio": 10.0
            },
            morphological_data={
                "wingspan_m": 1.1,
                "wing_area_sqm": 0.175,
                "mass_kg": 0.8,
                "body_length_m": 0.45
            },
            references=[
                "Tucker, V. A. (1998). Gliding flight: speed and acceleration of ideal falcons during diving and pull out.",
                "Ponitz, B., et al. (2014). Diving-flight aerodynamics of a peregrine falcon."
            ]
        )
        
        # Example reference model: Common Swift
        self.biological_references["common_swift"] = BiologicalReference(
            name="Common Swift",
            species="Apus apus",
            inspiration_type=BiologicalInspiration.BIRD,
            key_features=[
                "high_endurance",
                "efficient_gliding",
                "wing_morphing",
                "low_energy_flight"
            ],
            performance_metrics={
                "max_speed_kph": 110.0,
                "wing_loading_n_per_sqm": 40.0,
                "aspect_ratio": 8.5,
                "glide_ratio": 15.0
            },
            morphological_data={
                "wingspan_m": 0.42,
                "wing_area_sqm": 0.02,
                "mass_kg": 0.04,
                "body_length_m": 0.18
            },
            references=[
                "Henningsson, P., et al. (2011). Vortex wake and flight kinematics of a swift in cruising flight.",
                "Lentink, D., et al. (2007). How swifts control their glide performance."
            ]
        )
    
    def get_principle(self, principle: BiomimeticPrinciple) -> Dict[str, Any]:
        """
        Get information about a specific biomimetic design principle.
        
        Args:
            principle: The biomimetic principle to retrieve
            
        Returns:
            Dict containing principle information
        """
        return self.design_principles.get(principle, {})
    
    def get_biological_reference(self, reference_id: str) -> Optional[BiologicalReference]:
        """
        Get a biological reference model by ID.
        
        Args:
            reference_id: ID of the reference model
            
        Returns:
            BiologicalReference if found, None otherwise
        """
        return self.biological_references.get(reference_id)
    
    def map_biological_to_engineering(self, 
                                     biological_feature: str, 
                                     target_system: str) -> Dict[str, Any]:
        """
        Map biological features to engineering parameters.
        
        Args:
            biological_feature: The biological feature to map
            target_system: The target engineering system
            
        Returns:
            Dictionary of mapped engineering parameters
        """
        # This would contain more sophisticated mapping logic in a full implementation
        mapping_key = f"{biological_feature}_{target_system}"
        
        if mapping_key in self.parameter_mappings:
            return self.parameter_mappings[mapping_key]
        
        # Default mappings for common features
        if biological_feature == "wing_morphing":
            return {
                "parameters": {
                    "aspect_ratio_range": [2.0, 8.0],
                    "sweep_angle_range_deg": [15.0, 45.0],
                    "camber_range": [0.02, 0.08],
                    "flexibility_factor": 0.7
                },
                "implementation_notes": "Requires flexible materials and actuators",
                "performance_impact": {
                    "maneuverability": 0.8,
                    "efficiency": 0.6,
                    "speed": -0.2
                }
            }
        
        # Return empty mapping if no match
        return {}
    
    def add_biological_reference(self, reference_id: str, reference: BiologicalReference) -> bool:
        """
        Add a new biological reference model.
        
        Args:
            reference_id: ID for the reference model
            reference: The biological reference model
            
        Returns:
            True if added successfully, False otherwise
        """
        if reference_id in self.biological_references:
            logger.warning(f"Reference ID {reference_id} already exists")
            return False
            
        self.biological_references[reference_id] = reference
        return True
    
    def get_applicable_principles(self, design_area: str) -> List[BiomimeticPrinciple]:
        """
        Get biomimetic principles applicable to a specific design area.
        
        Args:
            design_area: The design area to check
            
        Returns:
            List of applicable biomimetic principles
        """
        applicable_principles = []
        
        for principle, data in self.design_principles.items():
            if design_area in data.get("application_areas", []):
                applicable_principles.append(principle)
                
        return applicable_principles
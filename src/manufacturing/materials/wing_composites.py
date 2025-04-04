#!/usr/bin/env python3
"""
Specialized composite layup techniques for biomimetic wing structures.
Provides optimized layup patterns based on biological wing structures.
"""

import os
import sys
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.manufacturing.materials.biomimetic_materials import BiomimeticMaterial, BiomimeticStructure
from src.biomimetic.design.wing_structures import WingStructure, WingType

logger = get_logger("wing_composites")


class FiberOrientation(Enum):
    """Fiber orientation patterns for composite layups."""
    UNIDIRECTIONAL = "unidirectional"
    BIDIRECTIONAL = "bidirectional"
    QUASI_ISOTROPIC = "quasi_isotropic"
    TAILORED = "tailored"
    BIOMIMETIC = "biomimetic"


class LayupRegion(Enum):
    """Regions of a wing for specialized layup patterns."""
    LEADING_EDGE = "leading_edge"
    TRAILING_EDGE = "trailing_edge"
    MAIN_SPAR = "main_spar"
    SECONDARY_SPAR = "secondary_spar"
    SKIN_UPPER = "skin_upper"
    SKIN_LOWER = "skin_lower"
    WINGTIP = "wingtip"
    ROOT = "root"
    CONTROL_SURFACE = "control_surface"


@dataclass
class LayupPattern:
    """Definition of a composite layup pattern."""
    name: str
    region: LayupRegion
    orientation: FiberOrientation
    ply_count: int
    ply_angles: List[float]
    material: str
    thickness_mm: float
    weight_factor: float
    biological_inspiration: Optional[str] = None
    special_properties: Dict[str, Any] = None


class WingCompositeTechniques:
    """Specialized composite layup techniques for biomimetic wings."""
    
    def __init__(self, biomimetic_materials=None):
        """
        Initialize wing composite techniques.
        
        Args:
            biomimetic_materials: Optional reference to biomimetic materials
        """
        self.biomimetic_materials = biomimetic_materials
        self.layup_patterns = self._initialize_layup_patterns()
        logger.info(f"Initialized wing composite techniques with {len(self.layup_patterns)} patterns")
    
    def _initialize_layup_patterns(self) -> Dict[str, LayupPattern]:
        """Initialize library of layup patterns."""
        patterns = {}
        
        # Bird wing leading edge pattern (inspired by bird wing bones)
        patterns["bird_leading_edge"] = LayupPattern(
            name="bird_leading_edge",
            region=LayupRegion.LEADING_EDGE,
            orientation=FiberOrientation.BIOMIMETIC,
            ply_count=6,
            ply_angles=[0, 45, 90, 90, 45, 0],
            material="bird_bone_titanium",
            thickness_mm=2.5,
            weight_factor=0.8,
            biological_inspiration="bird_wing_bone",
            special_properties={
                "impact_resistance": 1.4,
                "fatigue_resistance": 1.6
            }
        )
        
        # Bat wing membrane pattern
        patterns["bat_membrane"] = LayupPattern(
            name="bat_membrane",
            region=LayupRegion.SKIN_UPPER,
            orientation=FiberOrientation.TAILORED,
            ply_count=3,
            ply_angles=[30, -30, 30],
            material="spider_fiber_composite",
            thickness_mm=0.8,
            weight_factor=0.6,
            biological_inspiration="bat_wing_membrane",
            special_properties={
                "flexibility": 1.8,
                "tear_resistance": 1.5
            }
        )
        
        # Main spar pattern (inspired by bird wing bones)
        patterns["wing_main_spar"] = LayupPattern(
            name="wing_main_spar",
            region=LayupRegion.MAIN_SPAR,
            orientation=FiberOrientation.UNIDIRECTIONAL,
            ply_count=8,
            ply_angles=[0, 0, 45, -45, -45, 45, 0, 0],
            material="honeycomb_composite",
            thickness_mm=4.0,
            weight_factor=0.75,
            biological_inspiration="bird_wing_bone",
            special_properties={
                "bending_stiffness": 1.7,
                "torsional_stiffness": 1.3
            }
        )
        
        # Wingtip pattern (inspired by bird primary feathers)
        patterns["bird_wingtip"] = LayupPattern(
            name="bird_wingtip",
            region=LayupRegion.WINGTIP,
            orientation=FiberOrientation.BIOMIMETIC,
            ply_count=4,
            ply_angles=[15, -15, -15, 15],
            material="nacre_composite",
            thickness_mm=1.2,
            weight_factor=0.7,
            biological_inspiration="bird_primary_feathers",
            special_properties={
                "aeroelastic_tailoring": 1.6,
                "flutter_resistance": 1.4
            }
        )
        
        return patterns
    
    def get_layup_pattern(self, pattern_name: str) -> Optional[LayupPattern]:
        """Get a specific layup pattern by name."""
        return self.layup_patterns.get(pattern_name)
    
    def generate_wing_layup_schedule(self, 
                                   wing_structure: WingStructure,
                                   wing_span: float,
                                   wing_chord: float) -> Dict[str, Any]:
        """
        Generate a complete layup schedule for a biomimetic wing.
        
        Args:
            wing_structure: Wing structure definition
            wing_span: Wing span in meters
            wing_chord: Wing chord in meters
            
        Returns:
            Complete layup schedule with patterns for each region
        """
        # Select appropriate patterns based on wing type
        if wing_structure.wing_type in [WingType.BIRD_HIGH_ASPECT, 
                                      WingType.BIRD_ELLIPTICAL,
                                      WingType.BIRD_HIGH_SPEED,
                                      WingType.BIRD_SLOTTED]:
            base_patterns = self._get_bird_wing_patterns(wing_structure.wing_type)
        elif wing_structure.wing_type in [WingType.BAT_MEMBRANE, WingType.BAT_ARTICULATED]:
            base_patterns = self._get_bat_wing_patterns(wing_structure.wing_type)
        else:
            base_patterns = self._get_default_patterns()
        
        # Scale patterns based on wing dimensions
        scaled_patterns = self._scale_patterns_to_wing(base_patterns, wing_span, wing_chord)
        
        # Calculate material quantities
        material_quantities = self._calculate_material_quantities(scaled_patterns, wing_span, wing_chord)
        
        # Generate manufacturing instructions
        manufacturing_steps = self._generate_manufacturing_steps(scaled_patterns)
        
        return {
            "wing_type": wing_structure.wing_type.value,
            "wing_span": wing_span,
            "wing_chord": wing_chord,
            "layup_patterns": scaled_patterns,
            "material_quantities": material_quantities,
            "manufacturing_steps": manufacturing_steps,
            "estimated_weight": sum(material_quantities.values()),
            "biological_inspirations": self._get_biological_inspirations(scaled_patterns)
        }
    
    def _get_bird_wing_patterns(self, wing_type: WingType) -> Dict[LayupRegion, LayupPattern]:
        """Get appropriate patterns for bird wings."""
        patterns = {
            LayupRegion.LEADING_EDGE: self.layup_patterns["bird_leading_edge"],
            LayupRegion.MAIN_SPAR: self.layup_patterns["wing_main_spar"],
            LayupRegion.WINGTIP: self.layup_patterns["bird_wingtip"]
        }
        
        # Customize based on specific bird wing type
        if wing_type == WingType.BIRD_HIGH_ASPECT:
            # Modify for high aspect ratio (soaring birds)
            patterns[LayupRegion.MAIN_SPAR].ply_count += 2
            patterns[LayupRegion.MAIN_SPAR].ply_angles = [0, 0, 0, 45, -45, -45, 45, 0, 0, 0]
        elif wing_type == WingType.BIRD_HIGH_SPEED:
            # Modify for high speed (falcons, swifts)
            patterns[LayupRegion.LEADING_EDGE].thickness_mm += 0.5
        
        return patterns
    
    def _get_bat_wing_patterns(self, wing_type: WingType) -> Dict[LayupRegion, LayupPattern]:
        """Get appropriate patterns for bat wings."""
        patterns = {
            LayupRegion.LEADING_EDGE: self.layup_patterns["bird_leading_edge"],  # Reuse with modifications
            LayupRegion.SKIN_UPPER: self.layup_patterns["bat_membrane"],
            LayupRegion.SKIN_LOWER: self.layup_patterns["bat_membrane"]
        }
        
        # Customize for highly articulated bat wings
        if wing_type == WingType.BAT_ARTICULATED:
            patterns[LayupRegion.SKIN_UPPER].thickness_mm -= 0.2
            patterns[LayupRegion.SKIN_LOWER].thickness_mm -= 0.2
        
        return patterns
    
    def _get_default_patterns(self) -> Dict[LayupRegion, LayupPattern]:
        """Get default patterns for generic wings."""
        return {
            LayupRegion.LEADING_EDGE: self.layup_patterns["bird_leading_edge"],
            LayupRegion.MAIN_SPAR: self.layup_patterns["wing_main_spar"],
            LayupRegion.SKIN_UPPER: self.layup_patterns["bat_membrane"]
        }
    
    def _scale_patterns_to_wing(self, 
                              patterns: Dict[LayupRegion, LayupPattern],
                              wing_span: float,
                              wing_chord: float) -> Dict[str, Dict[str, Any]]:
        """Scale layup patterns to match wing dimensions."""
        scaled = {}
        
        for region, pattern in patterns.items():
            # Create a copy of the pattern with scaled dimensions
            scaled[region.value] = {
                "name": pattern.name,
                "material": pattern.material,
                "ply_count": pattern.ply_count,
                "ply_angles": pattern.ply_angles.copy(),
                "thickness_mm": pattern.thickness_mm,
                "area_m2": self._calculate_region_area(region, wing_span, wing_chord),
                "weight_kg": 0.0,  # Will be calculated later
                "biological_inspiration": pattern.biological_inspiration
            }
        
        return scaled
    
    def _calculate_region_area(self, 
                             region: LayupRegion, 
                             wing_span: float, 
                             wing_chord: float) -> float:
        """Calculate area of a wing region in square meters."""
        if region == LayupRegion.LEADING_EDGE:
            return 0.3 * wing_span * 0.2 * wing_chord
        elif region == LayupRegion.TRAILING_EDGE:
            return 0.7 * wing_span * 0.15 * wing_chord
        elif region == LayupRegion.MAIN_SPAR:
            return 0.8 * wing_span * 0.1 * wing_chord
        elif region == LayupRegion.SECONDARY_SPAR:
            return 0.6 * wing_span * 0.08 * wing_chord
        elif region == LayupRegion.SKIN_UPPER or region == LayupRegion.SKIN_LOWER:
            return 0.9 * wing_span * 0.9 * wing_chord
        elif region == LayupRegion.WINGTIP:
            return 0.2 * wing_span * 0.5 * wing_chord
        elif region == LayupRegion.ROOT:
            return 0.15 * wing_span * wing_chord
        elif region == LayupRegion.CONTROL_SURFACE:
            return 0.3 * wing_span * 0.25 * wing_chord
        else:
            return 0.1 * wing_span * wing_chord
    
    def _calculate_material_quantities(self, 
                                     scaled_patterns: Dict[str, Dict[str, Any]],
                                     wing_span: float,
                                     wing_chord: float) -> Dict[str, float]:
        """Calculate material quantities in kg."""
        material_quantities = {}
        
        for region, pattern in scaled_patterns.items():
            material = pattern["material"]
            area_m2 = pattern["area_m2"]
            thickness_m = pattern["thickness_mm"] / 1000.0
            
            # Get material density if available
            density = 1600.0  # Default density (kg/m³) for carbon fiber composites
            if self.biomimetic_materials and material in self.biomimetic_materials.biomimetic_materials:
                material_dict = self.biomimetic_materials._material_to_dict(
                    self.biomimetic_materials.biomimetic_materials[material]
                )
                density = material_dict.get("density", density)
            
            # Calculate weight
            weight_kg = area_m2 * thickness_m * density
            pattern["weight_kg"] = weight_kg
            
            # Add to total for this material
            if material in material_quantities:
                material_quantities[material] += weight_kg
            else:
                material_quantities[material] = weight_kg
        
        return material_quantities
    
    def _generate_manufacturing_steps(self, 
                                    scaled_patterns: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate manufacturing steps for the layup process."""
        steps = []
        
        # Step 1: Prepare molds
        steps.append({
            "step": 1,
            "description": "Prepare and clean molds",
            "details": "Apply release agent to all mold surfaces"
        })
        
        # Step 2: Cut materials
        material_list = []
        for region, pattern in scaled_patterns.items():
            for i in range(pattern["ply_count"]):
                material_list.append({
                    "material": pattern["material"],
                    "region": region,
                    "angle": pattern["ply_angles"][i],
                    "area_m2": pattern["area_m2"]
                })
        
        steps.append({
            "step": 2,
            "description": "Cut composite materials",
            "details": f"Cut {len(material_list)} plies according to templates",
            "materials": material_list
        })
        
        # Step 3: Layup process
        steps.append({
            "step": 3,
            "description": "Perform layup by region",
            "details": "Follow region-specific layup sequence",
            "regions": list(scaled_patterns.keys())
        })
        
        # Step 4: Vacuum bagging
        steps.append({
            "step": 4,
            "description": "Apply vacuum bagging",
            "details": "Ensure proper sealing and apply vacuum"
        })
        
        # Step 5: Curing
        steps.append({
            "step": 5,
            "description": "Cure composite",
            "details": "Cure according to material specifications"
        })
        
        # Step 6: Post-processing
        steps.append({
            "step": 6,
            "description": "Post-processing",
            "details": "Trim edges, sand surfaces, and inspect quality"
        })
        
        return steps
    
    def _get_biological_inspirations(self, 
                                   scaled_patterns: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Extract biological inspirations from patterns."""
        inspirations = {}
        
        for region, pattern in scaled_patterns.items():
            if pattern.get("biological_inspiration"):
                inspirations[region] = pattern["biological_inspiration"]
        
        return inspirations


def create_wing_composite_layup(wing_structure: WingStructure, 
                              wing_span: float, 
                              wing_chord: float) -> Dict[str, Any]:
    """
    Convenience function to create a composite layup for a wing.
    
    Args:
        wing_structure: Wing structure definition
        wing_span: Wing span in meters
        wing_chord: Wing chord in meters
        
    Returns:
        Complete layup schedule
    """
    from src.manufacturing.materials.biomimetic_materials import BiomimeticMaterialSelector
    
    # Create material selector to access biomimetic materials
    material_selector = BiomimeticMaterialSelector()
    
    # Create composite techniques with material reference
    composite_techniques = WingCompositeTechniques(material_selector)
    
    # Generate layup schedule
    return composite_techniques.generate_wing_layup_schedule(
        wing_structure, wing_span, wing_chord
    )
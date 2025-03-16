from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from enum import Enum

class MaterialProperty(Enum):
    DENSITY = "density"
    TENSILE_STRENGTH = "tensile_strength"
    THERMAL_RESISTANCE = "thermal_resistance"
    RADAR_ABSORPTION = "radar_absorption"
    COST = "cost"
    MANUFACTURABILITY = "manufacturability"

class Material:
    """Base class for materials used in UCAV airframes."""
    
    def __init__(self, name: str, properties: Dict[MaterialProperty, float]):
        self.name = name
        self.properties = properties
    
    def get_property(self, prop: MaterialProperty) -> float:
        """Get a specific material property."""
        return self.properties.get(prop, 0.0)

class MaterialOptimizer:
    """Optimizes material selection for UCAV airframes."""
    
    def __init__(self, available_materials: List[Material]):
        self.materials = available_materials
    
    def optimize(self, 
                requirements: Dict[MaterialProperty, Tuple[float, float]], 
                weights: Dict[MaterialProperty, float]) -> List[Tuple[Material, float]]:
        """
        Optimize material selection based on requirements and weights.
        
        Args:
            requirements: Min/max values for each property
            weights: Importance weights for each property
            
        Returns:
            List of (material, score) tuples sorted by score
        """
        scores = []
        
        for material in self.materials:
            score = 0.0
            for prop, weight in weights.items():
                if prop in requirements:
                    min_val, max_val = requirements[prop]
                    prop_value = material.get_property(prop)
                    
                    # Calculate normalized score for this property
                    if min_val <= prop_value <= max_val:
                        normalized_score = (prop_value - min_val) / (max_val - min_val)
                        score += weight * normalized_score
            
            scores.append((material, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

class MaterialLibrary:
    """Library of materials for UCAV construction."""
    
    def __init__(self):
        self.materials = {}
        self._initialize_default_materials()
    
    def _initialize_default_materials(self):
        """Initialize default materials library."""
        # Add some common aerospace materials
        self.add_material(Material(
            "Carbon Fiber Composite",
            {
                MaterialProperty.DENSITY: 1.6,
                MaterialProperty.TENSILE_STRENGTH: 3500,
                MaterialProperty.THERMAL_RESISTANCE: 500,
                MaterialProperty.RADAR_ABSORPTION: 0.8,
                MaterialProperty.COST: 100,
                MaterialProperty.MANUFACTURABILITY: 0.7
            }
        ))
        
        self.add_material(Material(
            "Titanium Alloy",
            {
                MaterialProperty.DENSITY: 4.5,
                MaterialProperty.TENSILE_STRENGTH: 1200,
                MaterialProperty.THERMAL_RESISTANCE: 1200,
                MaterialProperty.RADAR_ABSORPTION: 0.2,
                MaterialProperty.COST: 150,
                MaterialProperty.MANUFACTURABILITY: 0.5
            }
        ))
    
    def add_material(self, material: Material):
        """Add a material to the library."""
        self.materials[material.name] = material
    
    def get_material(self, name: str) -> Optional[Material]:
        """Get a material by name."""
        return self.materials.get(name)
    
    def get_all_materials(self) -> List[Material]:
        """Get all materials in the library."""
        return list(self.materials.values())
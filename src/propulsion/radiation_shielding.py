"""
Radiation Shielding Calculator for Nuclear Propulsion Systems.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.propulsion.nuclear_thermal_safety import ShieldingStatus


@dataclass
class ShieldingMaterial:
    """Properties of radiation shielding materials."""
    name: str
    density: float  # g/cm³
    half_value_thickness: Dict[str, float]  # cm for different radiation types
    max_temperature: float  # K
    thermal_conductivity: float  # W/(m·K)
    cost_factor: float  # Relative cost factor (1.0 = baseline)


class RadiationType(Enum):
    """Types of radiation to shield against."""
    GAMMA = "gamma"
    NEUTRON = "neutron"
    ALPHA = "alpha"
    BETA = "beta"


class RadiationShieldingCalculator:
    """Simple radiation shielding calculator for nuclear propulsion."""
    
    def __init__(self):
        """Initialize radiation shielding calculator."""
        self.materials = self._initialize_materials()
        
    def _initialize_materials(self) -> Dict[str, ShieldingMaterial]:
        """Initialize database of shielding materials."""
        return {
            "lead": ShieldingMaterial(
                name="Lead",
                density=11.34,
                half_value_thickness={
                    "gamma": 1.2,
                    "neutron": 10.0,
                    "alpha": 0.001,
                    "beta": 0.1
                },
                max_temperature=600.0,
                thermal_conductivity=35.3,
                cost_factor=1.0
            ),
            "water": ShieldingMaterial(
                name="Water",
                density=1.0,
                half_value_thickness={
                    "gamma": 10.0,
                    "neutron": 2.7,
                    "alpha": 0.001,
                    "beta": 0.5
                },
                max_temperature=373.0,
                thermal_conductivity=0.6,
                cost_factor=0.1
            ),
            "borated_polyethylene": ShieldingMaterial(
                name="Borated Polyethylene",
                density=1.19,
                half_value_thickness={
                    "gamma": 15.0,
                    "neutron": 2.0,
                    "alpha": 0.001,
                    "beta": 0.3
                },
                max_temperature=400.0,
                thermal_conductivity=0.25,
                cost_factor=2.0
            ),
            "tungsten": ShieldingMaterial(
                name="Tungsten",
                density=19.3,
                half_value_thickness={
                    "gamma": 0.8,
                    "neutron": 6.5,
                    "alpha": 0.001,
                    "beta": 0.05
                },
                max_temperature=3400.0,
                thermal_conductivity=173.0,
                cost_factor=5.0
            ),
        }
    
    def calculate_attenuation(self, 
                             material_name: str, 
                             thickness: float, 
                             radiation_type: RadiationType) -> float:
        """
        Calculate radiation attenuation for given material and thickness.
        
        Args:
            material_name: Name of shielding material
            thickness: Thickness in cm
            radiation_type: Type of radiation
            
        Returns:
            Attenuation factor (0-1, where 0 is complete attenuation)
        """
        if material_name not in self.materials:
            return 1.0  # No attenuation
            
        material = self.materials[material_name]
        radiation = radiation_type.value
        
        if radiation not in material.half_value_thickness:
            return 1.0  # No attenuation data
            
        # Calculate attenuation using half-value thickness
        # I = I₀ × 0.5^(thickness/half_value_thickness)
        half_value = material.half_value_thickness[radiation]
        attenuation = 0.5 ** (thickness / half_value)
        
        return attenuation
    
    def calculate_multi_layer_attenuation(self, 
                                        layers: List[Tuple[str, float]], 
                                        radiation_type: RadiationType) -> float:
        """
        Calculate attenuation for multiple layers of different materials.
        
        Args:
            layers: List of (material_name, thickness) tuples
            radiation_type: Type of radiation
            
        Returns:
            Combined attenuation factor
        """
        total_attenuation = 1.0
        
        for material_name, thickness in layers:
            layer_attenuation = self.calculate_attenuation(
                material_name, thickness, radiation_type)
            total_attenuation *= layer_attenuation
            
        return total_attenuation
    
    def recommend_shielding(self, 
                          radiation_levels: Dict[str, float],
                          target_attenuation: float,
                          max_weight: Optional[float] = None,
                          max_thickness: Optional[float] = None) -> Dict[str, Any]:
        """
        Recommend shielding configuration based on requirements.
        
        Args:
            radiation_levels: Current radiation levels by type
            target_attenuation: Target attenuation factor (0-1)
            max_weight: Maximum weight constraint in kg/m²
            max_thickness: Maximum thickness constraint in cm
            
        Returns:
            Recommended shielding configuration
        """
        best_config = None
        best_score = float('inf')
        
        # Try different material combinations
        for primary_material in self.materials.keys():
            for secondary_material in self.materials.keys():
                # Try different thickness ratios
                for ratio in [0.2, 0.5, 0.8]:
                    # Start with a minimal thickness and increase
                    for total_thickness in np.arange(0.5, 20.0, 0.5):
                        if max_thickness and total_thickness > max_thickness:
                            continue
                            
                        primary_thickness = total_thickness * ratio
                        secondary_thickness = total_thickness * (1 - ratio)
                        
                        # Calculate weight
                        primary_weight = primary_thickness * self.materials[primary_material].density
                        secondary_weight = secondary_thickness * self.materials[secondary_material].density
                        total_weight = primary_weight + secondary_weight
                        
                        if max_weight and total_weight > max_weight:
                            continue
                        
                        # Check attenuation for all radiation types
                        meets_requirements = True
                        for rad_type_str, level in radiation_levels.items():
                            try:
                                rad_type = RadiationType(rad_type_str)
                                layers = [(primary_material, primary_thickness),
                                         (secondary_material, secondary_thickness)]
                                attenuation = self.calculate_multi_layer_attenuation(layers, rad_type)
                                
                                if attenuation > target_attenuation:
                                    meets_requirements = False
                                    break
                            except ValueError:
                                # Skip invalid radiation types
                                continue
                        
                        if meets_requirements:
                            # Calculate score (lower is better)
                            cost_factor = (self.materials[primary_material].cost_factor * primary_thickness +
                                         self.materials[secondary_material].cost_factor * secondary_thickness)
                            score = total_weight * cost_factor
                            
                            if score < best_score:
                                best_score = score
                                best_config = {
                                    "layers": [
                                        {"material": primary_material, "thickness": primary_thickness},
                                        {"material": secondary_material, "thickness": secondary_thickness}
                                    ],
                                    "total_thickness": total_thickness,
                                    "total_weight": total_weight,
                                    "cost_factor": cost_factor
                                }
        
        if best_config:
            return {
                "success": True,
                "configuration": best_config,
                "attenuation_by_type": {
                    rad_type_str: self.calculate_multi_layer_attenuation(
                        [(layer["material"], layer["thickness"]) for layer in best_config["layers"]],
                        RadiationType(rad_type_str)
                    ) for rad_type_str in radiation_levels.keys() if rad_type_str in [rt.value for rt in RadiationType]
                }
            }
        else:
            return {
                "success": False,
                "message": "Could not find configuration meeting requirements"
            }
    
    def evaluate_shielding_status(self, 
                                thickness: float,
                                original_thickness: float,
                                radiation_level: float,
                                max_safe_level: float) -> ShieldingStatus:
        """
        Evaluate current shielding status based on thickness and radiation levels.
        
        Args:
            thickness: Current shielding thickness in cm
            original_thickness: Original design thickness in cm
            radiation_level: Current radiation level
            max_safe_level: Maximum safe radiation level
            
        Returns:
            Current shielding status
        """
        # Calculate degradation percentage
        degradation = (original_thickness - thickness) / original_thickness
        
        # Check radiation level relative to max safe level
        radiation_ratio = radiation_level / max_safe_level
        
        if degradation < 0.1 and radiation_ratio < 0.8:
            return ShieldingStatus.OPTIMAL
        elif degradation < 0.3 and radiation_ratio < 0.95:
            return ShieldingStatus.DEGRADED
        elif degradation < 0.5 and radiation_ratio < 1.2:
            return ShieldingStatus.COMPROMISED
        else:
            return ShieldingStatus.FAILED
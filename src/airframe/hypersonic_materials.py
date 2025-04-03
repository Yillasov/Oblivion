#!/usr/bin/env python3
"""
Advanced material modeling for hypersonic UCAV applications.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np
from dataclasses import dataclass
# Change back to absolute import
from src.airframe.materials import Material, MaterialProperty

class HypersonicProperty(Enum):
    """Properties specific to hypersonic flight materials."""
    HEAT_SHIELD_RATING = "heat_shield_rating"
    ABLATION_RESISTANCE = "ablation_resistance"
    THERMAL_EXPANSION = "thermal_expansion"
    OXIDATION_RESISTANCE = "oxidation_resistance"
    ACOUSTIC_DAMPING = "acoustic_damping"
    SHOCK_RESISTANCE = "shock_resistance"

@dataclass
class TemperatureProfile:
    """Temperature-dependent material properties."""
    temperature_points: List[float]  # Temperature points in Kelvin
    property_values: Dict[str, List[float]]  # Property values at each temperature
    
    def get_property_at_temperature(self, property_name: str, temperature: float) -> float:
        """Interpolate property value at a specific temperature."""
        if property_name not in self.property_values:
            return 0.0
            
        # Simple linear interpolation
        values = self.property_values[property_name]
        temps = self.temperature_points
        
        # Find temperature bounds
        if temperature <= temps[0]:
            return values[0]
        if temperature >= temps[-1]:
            return values[-1]
            
        # Interpolate
        for i in range(len(temps) - 1):
            if temps[i] <= temperature <= temps[i+1]:
                t_ratio = (temperature - temps[i]) / (temps[i+1] - temps[i])
                return values[i] + t_ratio * (values[i+1] - values[i])
        
        return 0.0

class HypersonicMaterial(Material):
    """Material specialized for hypersonic flight conditions."""
    
    def __init__(self, name: str, 
                 properties: Dict[MaterialProperty, float],
                 hypersonic_properties: Dict[HypersonicProperty, float],
                 temperature_profile: Optional[TemperatureProfile] = None,
                 max_service_temperature: float = 1500.0):
        super().__init__(name, properties)
        self.hypersonic_properties = hypersonic_properties
        self.temperature_profile = temperature_profile
        self.max_service_temperature = max_service_temperature
        
    def get_hypersonic_property(self, prop: HypersonicProperty) -> float:
        """Get a specific hypersonic material property."""
        return self.hypersonic_properties.get(prop, 0.0)
        
    def get_property_at_temperature(self, prop: str, temperature: float) -> float:
        """Get material property at specific temperature."""
        if self.temperature_profile:
            return self.temperature_profile.get_property_at_temperature(prop, temperature)
        return 0.0
        
    def calculate_thermal_stress(self, temperature_gradient: float) -> float:
        """Calculate thermal stress based on temperature gradient."""
        thermal_expansion = self.get_hypersonic_property(HypersonicProperty.THERMAL_EXPANSION)
        elastic_modulus = self.get_property(MaterialProperty.TENSILE_STRENGTH) / 100.0  # Approximation
        return thermal_expansion * elastic_modulus * temperature_gradient

class HypersonicMaterialLibrary:
    """Library of advanced materials for hypersonic UCAV applications."""
    
    def __init__(self):
        self.materials: Dict[str, HypersonicMaterial] = {}
        self._initialize_hypersonic_materials()
        
    def _initialize_hypersonic_materials(self):
        """Initialize library with advanced hypersonic materials."""
        # Ultra-High Temperature Ceramic (UHTC)
        uhtc_temp_profile = TemperatureProfile(
            temperature_points=[300, 1000, 1500, 2000, 2500],
            property_values={
                "thermal_conductivity": [60, 40, 30, 25, 20],
                "strength_retention": [1.0, 0.95, 0.85, 0.7, 0.5]
            }
        )
        
        self.add_material(HypersonicMaterial(
            "Zirconium Diboride UHTC",
            {
                MaterialProperty.DENSITY: 6.1,
                MaterialProperty.TENSILE_STRENGTH: 500,
                MaterialProperty.THERMAL_RESISTANCE: 2800,
                MaterialProperty.RADAR_ABSORPTION: 0.3,
                MaterialProperty.COST: 800,
                MaterialProperty.MANUFACTURABILITY: 0.4
            },
            {
                HypersonicProperty.HEAT_SHIELD_RATING: 0.9,
                HypersonicProperty.ABLATION_RESISTANCE: 0.85,
                HypersonicProperty.THERMAL_EXPANSION: 6.7e-6,
                HypersonicProperty.OXIDATION_RESISTANCE: 0.8,
                HypersonicProperty.ACOUSTIC_DAMPING: 0.5,
                HypersonicProperty.SHOCK_RESISTANCE: 0.75
            },
            uhtc_temp_profile,
            max_service_temperature=2700.0
        ))
        
        # Carbon-Carbon Composite
        cc_temp_profile = TemperatureProfile(
            temperature_points=[300, 1000, 1500, 2000],
            property_values={
                "thermal_conductivity": [120, 100, 80, 60],
                "strength_retention": [1.0, 0.9, 0.7, 0.5]
            }
        )
        
        self.add_material(HypersonicMaterial(
            "Carbon-Carbon Composite",
            {
                MaterialProperty.DENSITY: 1.8,
                MaterialProperty.TENSILE_STRENGTH: 700,
                MaterialProperty.THERMAL_RESISTANCE: 2200,
                MaterialProperty.RADAR_ABSORPTION: 0.9,
                MaterialProperty.COST: 1200,
                MaterialProperty.MANUFACTURABILITY: 0.5
            },
            {
                HypersonicProperty.HEAT_SHIELD_RATING: 0.85,
                HypersonicProperty.ABLATION_RESISTANCE: 0.7,
                HypersonicProperty.THERMAL_EXPANSION: 1.5e-6,
                HypersonicProperty.OXIDATION_RESISTANCE: 0.4,
                HypersonicProperty.ACOUSTIC_DAMPING: 0.8,
                HypersonicProperty.SHOCK_RESISTANCE: 0.9
            },
            cc_temp_profile,
            max_service_temperature=2400.0
        ))
        
        # Ceramic Matrix Composite (CMC)
        self.add_material(HypersonicMaterial(
            "SiC/SiC Ceramic Matrix Composite",
            {
                MaterialProperty.DENSITY: 2.7,
                MaterialProperty.TENSILE_STRENGTH: 800,
                MaterialProperty.THERMAL_RESISTANCE: 1800,
                MaterialProperty.RADAR_ABSORPTION: 0.6,
                MaterialProperty.COST: 950,
                MaterialProperty.MANUFACTURABILITY: 0.6
            },
            {
                HypersonicProperty.HEAT_SHIELD_RATING: 0.8,
                HypersonicProperty.ABLATION_RESISTANCE: 0.75,
                HypersonicProperty.THERMAL_EXPANSION: 4.0e-6,
                HypersonicProperty.OXIDATION_RESISTANCE: 0.85,
                HypersonicProperty.ACOUSTIC_DAMPING: 0.7,
                HypersonicProperty.SHOCK_RESISTANCE: 0.8
            },
            max_service_temperature=1800.0
        ))
        
    def add_material(self, material: HypersonicMaterial) -> None:
        """Add a material to the library."""
        self.materials[material.name] = material
        
    def get_material(self, name: str) -> Optional[HypersonicMaterial]:
        """Get a material by name."""
        return self.materials.get(name)
        
    def get_all_materials(self) -> List[HypersonicMaterial]:
        """Get all materials in the library."""
        return list(self.materials.values())
        
    def find_suitable_materials(self, max_temp: float, 
                              min_ablation_resistance: float = 0.7) -> List[HypersonicMaterial]:
        """Find materials suitable for given temperature and ablation requirements."""
        return [
            m for m in self.materials.values() 
            if m.max_service_temperature >= max_temp and 
            m.get_hypersonic_property(HypersonicProperty.ABLATION_RESISTANCE) >= min_ablation_resistance
        ]
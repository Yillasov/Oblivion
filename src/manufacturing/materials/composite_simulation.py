#!/usr/bin/env python3
"""
Flexible/Composite Material Simulation Module

Provides simulation capabilities for advanced composite materials used in
manufacturing, with focus on flexible composites and their mechanical properties.
"""

import numpy as np
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger

logger = get_logger("composite_simulation")


class CompositeType(Enum):
    """Types of composite materials."""
    LAMINATE = "laminate"
    SANDWICH = "sandwich"
    BRAIDED = "braided"
    WOVEN = "woven"
    FLEXIBLE = "flexible"
    HYBRID = "hybrid"


@dataclass
class LayerProperties:
    """Properties of a single composite layer."""
    material: str
    thickness: float  # mm
    orientation: float  # degrees
    youngs_modulus: Tuple[float, float]  # MPa in primary and secondary directions
    poisson_ratio: float
    shear_modulus: float  # MPa
    density: float  # kg/m³
    max_strain: Optional[float] = None  # Maximum allowable strain


@dataclass
class CompositeProperties:
    """Properties of a composite material."""
    composite_type: CompositeType
    layers: List[LayerProperties]
    total_thickness: float  # mm
    density: float  # kg/m³
    flexible: bool = False
    max_temperature: float = 120.0  # °C
    thermal_expansion: Optional[Tuple[float, float]] = None  # Coefficient in x,y directions


class CompositeSimulator:
    """Simulator for composite material behavior."""
    
    def __init__(self, time_step: float = 0.01):
        """
        Initialize composite simulator.
        
        Args:
            time_step: Simulation time step in seconds
        """
        self.time_step = time_step
        self.materials_db = {}  # Material properties database
        self.current_composite = None
        self.simulation_results = {}
        
        # Initialize default materials
        self._initialize_materials()
        
        logger.info("Composite simulator initialized")
    
    def _initialize_materials(self) -> None:
        """Initialize database with common composite materials."""
        # Carbon fiber properties
        self.materials_db["carbon_fiber"] = {
            "youngs_modulus": (230000, 15000),  # MPa in primary and secondary directions
            "poisson_ratio": 0.3,
            "shear_modulus": 50000,  # MPa
            "density": 1800,  # kg/m³
            "max_strain": 0.018,  # 1.8%
            "thermal_expansion": (0.1e-6, 10e-6)  # 1/°C
        }
        
        # Glass fiber properties
        self.materials_db["glass_fiber"] = {
            "youngs_modulus": (85000, 8500),  # MPa
            "poisson_ratio": 0.2,
            "shear_modulus": 4500,  # MPa
            "density": 2500,  # kg/m³
            "max_strain": 0.025,  # 2.5%
            "thermal_expansion": (6e-6, 30e-6)  # 1/°C
        }
        
        # Flexible composite properties
        self.materials_db["flexible_composite"] = {
            "youngs_modulus": (50000, 40000),  # MPa
            "poisson_ratio": 0.35,
            "shear_modulus": 3000,  # MPa
            "density": 1500,  # kg/m³
            "max_strain": 0.05,  # 5%
            "thermal_expansion": (8e-6, 8e-6)  # 1/°C
        }
        
        # Aramid fiber properties
        self.materials_db["aramid_fiber"] = {
            "youngs_modulus": (120000, 5000),  # MPa
            "poisson_ratio": 0.4,
            "shear_modulus": 2500,  # MPa
            "density": 1440,  # kg/m³
            "max_strain": 0.035,  # 3.5%
            "thermal_expansion": (-2e-6, 60e-6)  # 1/°C
        }
    
    def create_composite(self, layup_sequence: List[Dict[str, Any]], 
                        composite_type: CompositeType = CompositeType.LAMINATE) -> CompositeProperties:
        """
        Create a composite material from a layup sequence.
        
        Args:
            layup_sequence: List of layer specifications
            composite_type: Type of composite
            
        Returns:
            Composite properties
        """
        layers = []
        total_thickness = 0.0
        total_mass = 0.0
        
        for layer_spec in layup_sequence:
            material_name = layer_spec.get("material", "carbon_fiber")
            material = self.materials_db.get(material_name, self.materials_db["carbon_fiber"])
            
            thickness = layer_spec.get("thickness", 0.125)  # mm
            orientation = layer_spec.get("orientation", 0.0)  # degrees
            
            layer = LayerProperties(
                material=material_name,
                thickness=thickness,
                orientation=orientation,
                youngs_modulus=material["youngs_modulus"],
                poisson_ratio=material["poisson_ratio"],
                shear_modulus=material["shear_modulus"],
                density=material["density"],
                max_strain=material.get("max_strain")
            )
            
            layers.append(layer)
            total_thickness += thickness
            total_mass += thickness * material["density"]
        
        # Calculate average density
        avg_density = total_mass / total_thickness if total_thickness > 0 else 0
        
        # Determine if composite is flexible
        is_flexible = composite_type == CompositeType.FLEXIBLE or any(
            layer.material == "flexible_composite" for layer in layers
        )
        
        composite = CompositeProperties(
            composite_type=composite_type,
            layers=layers,
            total_thickness=total_thickness,
            density=avg_density,
            flexible=is_flexible,
            thermal_expansion=self._calculate_thermal_expansion(layers)
        )
        
        self.current_composite = composite
        return composite
    
    def _calculate_thermal_expansion(self, layers: List[LayerProperties]) -> Tuple[float, float]:
        """Calculate effective thermal expansion coefficients."""
        if not layers:
            return (0.0, 0.0)
        
        # Simple weighted average for demonstration
        total_thickness = sum(layer.thickness for layer in layers)
        alpha_x = sum(
            layer.thickness * self.materials_db[layer.material]["thermal_expansion"][0]
            for layer in layers
        ) / total_thickness
        
        alpha_y = sum(
            layer.thickness * self.materials_db[layer.material]["thermal_expansion"][1]
            for layer in layers
        ) / total_thickness
        
        return (alpha_x, alpha_y)
    
    def simulate_mechanical_response(self, 
                                   load_conditions: Dict[str, float],
                                   temperature: float = 25.0) -> Dict[str, Any]:
        """
        Simulate mechanical response of the composite under load.
        
        Args:
            load_conditions: Applied loads and boundary conditions
            temperature: Operating temperature in °C
            
        Returns:
            Simulation results
        """
        if not self.current_composite:
            return {"error": "No composite defined"}
        
        # Extract load conditions
        tension_x = load_conditions.get("tension_x", 0.0)  # N/mm
        tension_y = load_conditions.get("tension_y", 0.0)  # N/mm
        shear_xy = load_conditions.get("shear_xy", 0.0)  # N/mm
        bending_moment = load_conditions.get("bending_moment", 0.0)  # N·mm/mm
        
        # Calculate thermal effects
        delta_t = temperature - 25.0  # Temperature difference from reference
        
        # Calculate strains (simplified model)
        # For a more accurate model, would use Classical Lamination Theory
        thickness = self.current_composite.total_thickness
        
        # Simplified calculation of effective moduli
        e_x, e_y, g_xy = self._calculate_effective_moduli()
        
        # Calculate strains
        strain_x = tension_x / (e_x * thickness) + delta_t * self.current_composite.thermal_expansion[0]
        strain_y = tension_y / (e_y * thickness) + delta_t * self.current_composite.thermal_expansion[1]
        strain_xy = shear_xy / (g_xy * thickness)
        
        # Calculate curvature from bending (simplified)
        curvature = 12 * bending_moment / (e_x * thickness**3)
        
        # Check for failure
        max_strain = min([layer.max_strain for layer in self.current_composite.layers 
                         if layer.max_strain is not None], default=0.02)
        
        failure_index = max(abs(strain_x), abs(strain_y)) / max_strain
        
        # Store results
        results = {
            "strain_x": strain_x,
            "strain_y": strain_y,
            "strain_xy": strain_xy,
            "curvature": curvature,
            "failure_index": failure_index,
            "failure_predicted": failure_index > 1.0,
            "max_deflection": self._calculate_max_deflection(curvature, load_conditions)
        }
        
        self.simulation_results = results
        return results
    
    def _calculate_effective_moduli(self) -> Tuple[float, float, float]:
        """Calculate effective moduli of the composite."""
        if not self.current_composite or not self.current_composite.layers:
            return (0.0, 0.0, 0.0)
        
        # Simple rule of mixtures for demonstration
        # For accurate results, would use lamination theory
        total_thickness = self.current_composite.total_thickness
        
        # Initialize weighted sums
        e_x_sum = 0.0
        e_y_sum = 0.0
        g_xy_sum = 0.0
        
        for layer in self.current_composite.layers:
            # Account for fiber orientation
            theta_rad = np.radians(layer.orientation)
            c = np.cos(theta_rad)
            s = np.sin(theta_rad)
            
            e1, e2 = layer.youngs_modulus
            
            # Simplified transformation for demonstration
            # Would use full transformation matrices in production
            e_x_effective = e1 * c**4 + e2 * s**4
            e_y_effective = e1 * s**4 + e2 * c**4
            
            # Weight by thickness
            weight = layer.thickness / total_thickness
            e_x_sum += e_x_effective * weight
            e_y_sum += e_y_effective * weight
            g_xy_sum += layer.shear_modulus * weight
        
        return (e_x_sum, e_y_sum, g_xy_sum)
    
    def _calculate_max_deflection(self, curvature: float, 
                                load_conditions: Dict[str, float]) -> float:
        """Calculate maximum deflection based on curvature and span."""
        # Get span from load conditions or use default
        span = load_conditions.get("span", 100.0)  # mm
        
        # For a simply supported beam with uniform curvature
        max_deflection = curvature * span**2 / 8.0
        
        return max_deflection
    
    def simulate_impact(self, impact_energy: float, 
                      impact_location: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, Any]:
        """
        Simulate impact response of the composite.
        
        Args:
            impact_energy: Impact energy in Joules
            impact_location: Location of impact (x, y) in mm
            
        Returns:
            Impact simulation results
        """
        if not self.current_composite:
            return {"error": "No composite defined"}
        
        # Simplified impact model
        thickness = self.current_composite.total_thickness
        is_flexible = self.current_composite.flexible
        
        # Calculate energy absorption capacity (simplified)
        energy_capacity = thickness * 5.0  # J/mm
        
        # Flexible composites absorb more energy
        if is_flexible:
            energy_capacity *= 1.5
        
        # Calculate damage
        damage_ratio = min(1.0, impact_energy / energy_capacity)
        
        # Calculate deformation
        deformation = impact_energy / (thickness * 10.0)
        if is_flexible:
            deformation *= 2.0
        
        # Calculate if penetration occurs
        penetration = damage_ratio > 0.8
        
        results = {
            "damage_ratio": damage_ratio,
            "deformation": deformation,
            "penetration": penetration,
            "energy_absorbed": min(impact_energy, energy_capacity)
        }
        
        return results


def create_flexible_composite() -> CompositeProperties:
    """Create a default flexible composite for quick testing."""
    simulator = CompositeSimulator()
    
    layup = [
        {"material": "flexible_composite", "thickness": 0.5, "orientation": 0},
        {"material": "aramid_fiber", "thickness": 0.2, "orientation": 45},
        {"material": "flexible_composite", "thickness": 0.5, "orientation": 90}
    ]
    
    return simulator.create_composite(layup, CompositeType.FLEXIBLE)
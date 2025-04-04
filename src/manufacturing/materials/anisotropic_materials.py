#!/usr/bin/env python3
"""
Anisotropic material properties modeling for advanced manufacturing.
Provides tools for simulating direction-dependent material behavior.
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

logger = get_logger("anisotropic_materials")


class AnisotropyType(Enum):
    """Types of material anisotropy."""
    ISOTROPIC = "isotropic"
    ORTHOTROPIC = "orthotropic"
    TRANSVERSELY_ISOTROPIC = "transversely_isotropic"
    MONOCLINIC = "monoclinic"
    FULLY_ANISOTROPIC = "fully_anisotropic"


@dataclass
class AnisotropicProperties:
    """Anisotropic material properties."""
    anisotropy_type: AnisotropyType
    
    # Elastic moduli in different directions (MPa)
    elastic_moduli: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Shear moduli (MPa)
    shear_moduli: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Poisson's ratios
    poisson_ratios: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Thermal expansion coefficients (1/°C)
    thermal_expansion: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Thermal conductivity (W/m·K)
    thermal_conductivity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Principal material directions (unit vectors)
    material_directions: Optional[List[np.ndarray]] = None


class AnisotropicMaterialModel:
    """Model for anisotropic material behavior."""
    
    def __init__(self):
        """Initialize anisotropic material model."""
        self.materials_db = {}
        self._initialize_materials()
        logger.info("Anisotropic material model initialized")
    
    def _initialize_materials(self) -> None:
        """Initialize database with common anisotropic materials."""
        # Carbon fiber composite (transversely isotropic)
        self.materials_db["carbon_fiber_composite"] = AnisotropicProperties(
            anisotropy_type=AnisotropyType.TRANSVERSELY_ISOTROPIC,
            elastic_moduli=(230000.0, 15000.0, 15000.0),  # E1, E2, E3 (MPa)
            shear_moduli=(5000.0, 5000.0, 7000.0),        # G12, G13, G23 (MPa)
            poisson_ratios=(0.25, 0.25, 0.3),             # v12, v13, v23
            thermal_expansion=(0.1e-6, 25e-6, 25e-6),     # α1, α2, α3 (1/°C)
            thermal_conductivity=(7.0, 0.8, 0.8),         # k1, k2, k3 (W/m·K)
            material_directions=[
                np.array([1.0, 0.0, 0.0]),  # Fiber direction
                np.array([0.0, 1.0, 0.0]),  # Transverse direction 1
                np.array([0.0, 0.0, 1.0])   # Transverse direction 2
            ]
        )
        
        # Titanium alloy (isotropic)
        self.materials_db["titanium_alloy"] = AnisotropicProperties(
            anisotropy_type=AnisotropyType.ISOTROPIC,
            elastic_moduli=(110000.0, 110000.0, 110000.0),
            shear_moduli=(42000.0, 42000.0, 42000.0),
            poisson_ratios=(0.33, 0.33, 0.33),
            thermal_expansion=(8.6e-6, 8.6e-6, 8.6e-6),
            thermal_conductivity=(6.7, 6.7, 6.7)
        )
        
        # Wood (orthotropic)
        self.materials_db["wood"] = AnisotropicProperties(
            anisotropy_type=AnisotropyType.ORTHOTROPIC,
            elastic_moduli=(11000.0, 1000.0, 500.0),
            shear_moduli=(900.0, 700.0, 400.0),
            poisson_ratios=(0.35, 0.4, 0.3),
            thermal_expansion=(3.0e-6, 30e-6, 40e-6),
            thermal_conductivity=(0.4, 0.2, 0.1)
        )
    
    def get_material(self, material_name: str) -> Optional[AnisotropicProperties]:
        """Get anisotropic properties for a specific material."""
        return self.materials_db.get(material_name)
    
    def add_material(self, material_name: str, properties: AnisotropicProperties) -> None:
        """Add a new material to the database."""
        self.materials_db[material_name] = properties
        logger.info(f"Added material '{material_name}' to anisotropic database")
    
    def calculate_stiffness_matrix(self, material_name: str) -> np.ndarray:
        """Calculate the stiffness matrix for a material."""
        material = self.get_material(material_name)
        if not material:
            logger.warning(f"Material '{material_name}' not found in database")
            return np.zeros((6, 6))
        
        # Extract properties
        E1, E2, E3 = material.elastic_moduli
        G12, G13, G23 = material.shear_moduli
        v12, v13, v23 = material.poisson_ratios
        
        # Calculate derived Poisson's ratios using symmetry relations
        v21 = v12 * (E2 / E1)
        v31 = v13 * (E3 / E1)
        v32 = v23 * (E3 / E2)
        
        # Initialize stiffness matrix
        C = np.zeros((6, 6))
        
        if material.anisotropy_type == AnisotropyType.ISOTROPIC:
            # For isotropic materials
            E = E1
            G = G12
            v = v12
            
            lam = (E * v) / ((1 + v) * (1 - 2 * v))
            mu = G
            
            # Fill the stiffness matrix
            for i in range(3):
                for j in range(3):
                    if i == j:
                        C[i, j] = lam + 2 * mu
                    else:
                        C[i, j] = lam
            
            C[3, 3] = C[4, 4] = C[5, 5] = mu
            
        elif material.anisotropy_type == AnisotropyType.ORTHOTROPIC:
            # For orthotropic materials
            denominator = (1 - v12*v21 - v23*v32 - v31*v13 - 2*v21*v32*v13)
            
            C[0, 0] = E1 * (1 - v23*v32) / denominator
            C[1, 1] = E2 * (1 - v13*v31) / denominator
            C[2, 2] = E3 * (1 - v12*v21) / denominator
            
            C[0, 1] = E1 * (v21 + v31*v23) / denominator
            C[1, 0] = C[0, 1]
            
            C[0, 2] = E1 * (v31 + v21*v32) / denominator
            C[2, 0] = C[0, 2]
            
            C[1, 2] = E2 * (v32 + v12*v31) / denominator
            C[2, 1] = C[1, 2]
            
            C[3, 3] = G12
            C[4, 4] = G13
            C[5, 5] = G23
            
        elif material.anisotropy_type == AnisotropyType.TRANSVERSELY_ISOTROPIC:
            # For transversely isotropic materials (simplified)
            E_L = E1  # Longitudinal modulus
            E_T = E2  # Transverse modulus
            v_LT = v12  # Major Poisson's ratio
            v_TT = v23  # Transverse Poisson's ratio
            G_LT = G12  # Longitudinal-transverse shear modulus
            
            v_TL = v_LT * (E_T / E_L)
            
            # Fill the stiffness matrix
            C[0, 0] = E_L * (1 - v_TT**2) / (1 - v_TT**2 - 2*v_LT*v_TL)
            C[1, 1] = C[2, 2] = E_T * (1 - v_LT*v_TL) / (1 - v_TT**2 - 2*v_LT*v_TL)
            
            C[0, 1] = C[0, 2] = E_L * (v_TL + v_LT*v_TT) / (1 - v_TT**2 - 2*v_LT*v_TL)
            C[1, 0] = C[2, 0] = C[0, 1]
            
            C[1, 2] = C[2, 1] = E_T * (v_TT + v_LT*v_TL) / (1 - v_TT**2 - 2*v_LT*v_TL)
            
            C[3, 3] = C[4, 4] = G_LT
            C[5, 5] = E_T / (2 * (1 + v_TT))
        
        return C
    
    def calculate_thermal_strain(self, material_name: str, 
                               temperature_change: float) -> np.ndarray:
        """Calculate thermal strain for a temperature change."""
        material = self.get_material(material_name)
        if not material:
            return np.zeros(6)
        
        # Extract thermal expansion coefficients
        alpha = material.thermal_expansion
        
        # Calculate thermal strain vector [εx, εy, εz, γyz, γxz, γxy]
        thermal_strain = np.array([
            alpha[0] * temperature_change,
            alpha[1] * temperature_change,
            alpha[2] * temperature_change,
            0.0,  # No shear thermal strain
            0.0,
            0.0
        ])
        
        return thermal_strain
    
    def calculate_stress(self, material_name: str, strain: np.ndarray, 
                       temperature_change: float = 0.0) -> np.ndarray:
        """
        Calculate stress from strain and temperature change.
        
        Args:
            material_name: Name of the material
            strain: Strain vector [εx, εy, εz, γyz, γxz, γxy]
            temperature_change: Temperature change from reference (°C)
            
        Returns:
            Stress vector [σx, σy, σz, τyz, τxz, τxy]
        """
        # Get stiffness matrix
        C = self.calculate_stiffness_matrix(material_name)
        
        # Calculate thermal strain
        thermal_strain = self.calculate_thermal_strain(material_name, temperature_change)
        
        # Calculate mechanical strain (total strain - thermal strain)
        mechanical_strain = strain - thermal_strain
        
        # Calculate stress using Hooke's law: σ = C·ε
        stress = C @ mechanical_strain
        
        return stress
    
    def rotate_properties(self, material_name: str, 
                        rotation_matrix: np.ndarray) -> AnisotropicProperties:
        """
        Rotate material properties to a new coordinate system.
        
        Args:
            material_name: Name of the material
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Rotated material properties
        """
        material = self.get_material(material_name)
        if not material:
            logger.warning(f"Material '{material_name}' not found in database")
            return None
        
        # Create a copy of the material properties
        rotated = AnisotropicProperties(
            anisotropy_type=material.anisotropy_type,
            elastic_moduli=material.elastic_moduli,
            shear_moduli=material.shear_moduli,
            poisson_ratios=material.poisson_ratios,
            thermal_expansion=material.thermal_expansion,
            thermal_conductivity=material.thermal_conductivity
        )
        
        # For fully anisotropic materials, we would transform the stiffness matrix
        # This is a simplified implementation for demonstration
        
        # Rotate material directions if they exist
        if material.material_directions:
            rotated.material_directions = [
                rotation_matrix @ direction 
                for direction in material.material_directions
            ]
        
        return rotated


def get_anisotropic_model() -> AnisotropicMaterialModel:
    """Get a singleton instance of the anisotropic material model."""
    if not hasattr(get_anisotropic_model, "instance"):
        get_anisotropic_model.instance = AnisotropicMaterialModel()
    return get_anisotropic_model.instance
"""
Radar Cross-Section (RCS) Simulation Module

Provides capabilities to simulate and analyze radar cross-section 
characteristics of UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.stealth.materials.ram.ram_system import RAMSystem
from src.propulsion.stealth_integration import PropulsionStealthIntegrator


class RCSFrequencyBand(Enum):
    """Radar frequency bands."""
    L_BAND = "L"  # 1-2 GHz
    S_BAND = "S"  # 2-4 GHz
    C_BAND = "C"  # 4-8 GHz
    X_BAND = "X"  # 8-12 GHz
    KU_BAND = "Ku"  # 12-18 GHz


@dataclass
class RCSSimulationConfig:
    """Configuration for RCS simulation."""
    frequency_band: RCSFrequencyBand = RCSFrequencyBand.X_BAND
    angle_resolution: float = 5.0  # Degrees
    include_propulsion_effects: bool = True
    include_material_effects: bool = True
    include_shape_effects: bool = True


class RCSSimulator:
    """Simulates radar cross-section characteristics of UCAV platforms."""
    
    def __init__(self, config: RCSSimulationConfig):
        """Initialize RCS simulator."""
        self.config = config
        self.shape_rcs_data: Dict[str, np.ndarray] = {}
        self.material_systems: Dict[str, RAMSystem] = {}
        self.propulsion_integrator: Optional[PropulsionStealthIntegrator] = None
        
    def register_material_system(self, system_id: str, system: RAMSystem) -> None:
        """Register a RAM system for RCS calculation."""
        self.material_systems[system_id] = system
        
    def register_propulsion_integrator(self, integrator: PropulsionStealthIntegrator) -> None:
        """Register propulsion-stealth integrator."""
        self.propulsion_integrator = integrator
        
    def load_shape_data(self, shape_id: str, rcs_data: np.ndarray) -> None:
        """
        Load RCS data for a specific shape.
        
        Args:
            shape_id: Identifier for the shape
            rcs_data: Array of RCS values by angle (azimuth, elevation)
        """
        self.shape_rcs_data[shape_id] = rcs_data
        
    def calculate_rcs(self, 
                     shape_id: str, 
                     azimuth: float, 
                     elevation: float,
                     propulsion_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate RCS for given parameters.
        
        Args:
            shape_id: Shape identifier
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            propulsion_state: Current propulsion system state
            
        Returns:
            Dictionary with RCS results
        """
        if shape_id not in self.shape_rcs_data:
            return {"error": f"Shape {shape_id} not found"}
            
        # Get base RCS from shape
        base_rcs = self._get_shape_rcs(shape_id, azimuth, elevation)
        
        # Apply material effects
        material_factor = 1.0
        if self.config.include_material_effects and self.material_systems:
            material_factor = self._calculate_material_factor()
            
        # Apply propulsion effects
        propulsion_factor = 1.0
        if self.config.include_propulsion_effects and self.propulsion_integrator and propulsion_state:
            propulsion_factor = self._calculate_propulsion_factor(propulsion_state)
            
        # Calculate final RCS
        final_rcs = base_rcs * material_factor * propulsion_factor
        
        return {
            "base_rcs": base_rcs,
            "material_factor": material_factor,
            "propulsion_factor": propulsion_factor,
            "final_rcs": final_rcs,
            "frequency_band": self.config.frequency_band.value,
            "azimuth": azimuth,
            "elevation": elevation
        }
        
    def generate_rcs_profile(self, 
                           shape_id: str,
                           propulsion_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate complete RCS profile for all angles.
        
        Args:
            shape_id: Shape identifier
            propulsion_state: Current propulsion system state
            
        Returns:
            Dictionary with RCS profile data
        """
        if shape_id not in self.shape_rcs_data:
            return {"error": f"Shape {shape_id} not found"}
            
        # Generate angles based on resolution
        resolution = self.config.angle_resolution
        azimuth_angles = np.arange(0, 360, resolution)
        elevation_angles = np.arange(-90, 91, resolution)
        
        # Calculate RCS for each angle
        rcs_data = {}
        for az in azimuth_angles:
            rcs_data[float(az)] = {}
            for el in elevation_angles:
                rcs_data[float(az)][float(el)] = self.calculate_rcs(
                    shape_id, float(az), float(el), propulsion_state
                )["final_rcs"]
                
        # Calculate statistics
        rcs_values = [rcs_data[az][el] for az in rcs_data for el in rcs_data[az]]
        avg_rcs = np.mean(rcs_values)
        max_rcs = np.max(rcs_values)
        min_rcs = np.min(rcs_values)
        
        return {
            "profile": rcs_data,
            "statistics": {
                "average_rcs": avg_rcs,
                "max_rcs": max_rcs,
                "min_rcs": min_rcs,
                "frequency_band": self.config.frequency_band.value
            }
        }
        
    def _get_shape_rcs(self, shape_id: str, azimuth: float, elevation: float) -> float:
        """Get RCS value for a specific shape and angle."""
        rcs_data = self.shape_rcs_data[shape_id]
        
        # Convert angles to indices
        resolution = self.config.angle_resolution
        az_idx = int(azimuth / resolution) % (360 // int(resolution))
        el_idx = int((elevation + 90) / resolution) % (181 // int(resolution))
        
        # Get RCS value
        return float(rcs_data[az_idx, el_idx])
        
    def _calculate_material_factor(self) -> float:
        """Calculate RCS reduction factor from material systems."""
        if not self.material_systems:
            return 1.0
            
        # Average reduction from all material systems
        total_reduction = 0.0
        for system_id, system in self.material_systems.items():
            # Get RCS from material specs
            rcs = system.specs.radar_cross_section
            total_reduction += (1.0 - rcs)
            
        avg_reduction = total_reduction / len(self.material_systems)
        return 1.0 - avg_reduction
        
    def _calculate_propulsion_factor(self, propulsion_state: Dict[str, Any]) -> float:
        """Calculate RCS impact from propulsion systems."""
        if not self.propulsion_integrator:
            return 1.0
            
        # Get propulsion systems with stealth integration
        systems = propulsion_state.get("systems", {})
        
        total_factor = 0.0
        count = 0
        
        for system_id, state in systems.items():
            # Check if system has stealth integration
            if self.propulsion_integrator.get_propulsion_impact(system_id).get("has_stealth_integration", False):
                # Higher power levels generally increase RCS
                power_level = state.get("power_level", 0.0)
                
                # Calculate factor (higher power = higher RCS)
                factor = 1.0 + (power_level * 0.5)  # 50% increase at max power
                
                total_factor += factor
                count += 1
                
        if count == 0:
            return 1.0
            
        return total_factor / count
import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import math
from src.core.geometry.complex_geometry import (
    ComplexGeometry, PrimitiveGeometry, MeshGeometry, 
    PrimitiveType, Transform, Material
)

@dataclass
class UCAVGeometry:
    """Enhanced UCAV geometry model with complex geometry support."""
    
    # Basic parameters
    length: float
    wingspan: float
    mean_chord: float
    sweep_angle: float  # in degrees
    taper_ratio: float
    
    # Optional features
    has_vertical_tail: bool = False
    has_canards: bool = False
    
    # Advanced geometry
    complex_geometry: Optional[ComplexGeometry] = None
    
    def __post_init__(self):
        """Initialize complex geometry if not provided."""
        if self.complex_geometry is None:
            self.complex_geometry = self._generate_complex_geometry()
    
    def _generate_complex_geometry(self) -> ComplexGeometry:
        """Generate complex geometry from basic parameters."""
        ucav = ComplexGeometry("ucav")
        
        # Calculate derived dimensions
        root_chord = 2 * self.mean_chord / (1 + self.taper_ratio)
        tip_chord = root_chord * self.taper_ratio
        
        # Convert sweep angle to radians
        sweep_rad = math.radians(self.sweep_angle)
        
        # Create wing geometry
        wing = ComplexGeometry("wing")
        
        # Left wing
        left_wing = PrimitiveGeometry(
            type=PrimitiveType.WEDGE,
            dimensions={
                "length": self.wingspan / 2,
                "width_start": root_chord,
                "width_end": tip_chord,
                "height": root_chord * 0.1  # 10% thickness
            },
            transform=Transform(
                position=np.array([0, 0, 0]),
                rotation=np.array([0, 0, -sweep_rad]),
                scale=np.array([1, 1, 1])
            ),
            material=Material(
                name="composite",
                radar_reflectivity=0.3,
                ir_emissivity=0.7
            )
        )
        
        # Right wing (mirror of left wing)
        right_wing = PrimitiveGeometry(
            type=PrimitiveType.WEDGE,
            dimensions={
                "length": self.wingspan / 2,
                "width_start": root_chord,
                "width_end": tip_chord,
                "height": root_chord * 0.1  # 10% thickness
            },
            transform=Transform(
                position=np.array([0, 0, 0]),
                rotation=np.array([0, 0, sweep_rad]),
                scale=np.array([1, 1, 1])
            ),
            material=Material(
                name="composite",
                radar_reflectivity=0.3,
                ir_emissivity=0.7
            )
        )
        
        wing.add_component(left_wing)
        wing.add_component(right_wing)
        
        # Create fuselage
        fuselage = PrimitiveGeometry(
            type=PrimitiveType.ELLIPSOID,
            dimensions={
                "length": self.length,
                "width": root_chord * 0.3,
                "height": root_chord * 0.2
            },
            transform=Transform(
                position=np.array([self.length/2, 0, 0]),
                rotation=np.array([0, 0, 0]),
                scale=np.array([1, 1, 1])
            ),
            material=Material(
                name="fuselage_material",
                radar_reflectivity=0.4,
                ir_emissivity=0.6
            )
        )
        
        # Add components to UCAV
        ucav.add_component(wing)
        ucav.add_component(fuselage)
        
        # Add vertical tail if needed
        if self.has_vertical_tail:
            vertical_tail = PrimitiveGeometry(
                type=PrimitiveType.WEDGE,
                dimensions={
                    "length": self.wingspan * 0.15,
                    "width_start": root_chord * 0.5,
                    "width_end": root_chord * 0.3,
                    "height": root_chord * 0.05
                },
                transform=Transform(
                    position=np.array([self.length * 0.8, 0, 0]),
                    rotation=np.array([0, math.pi/2, 0]),  # Rotate to vertical
                    scale=np.array([1, 1, 1])
                ),
                material=Material(
                    name="tail_material",
                    radar_reflectivity=0.35,
                    ir_emissivity=0.65
                )
            )
            ucav.add_component(vertical_tail)
        
        # Add canards if needed
        if self.has_canards:
            # Left canard
            left_canard = PrimitiveGeometry(
                type=PrimitiveType.WEDGE,
                dimensions={
                    "length": self.wingspan * 0.15,
                    "width_start": root_chord * 0.4,
                    "width_end": root_chord * 0.2,
                    "height": root_chord * 0.05
                },
                transform=Transform(
                    position=np.array([self.length * 0.2, 0, 0]),
                    rotation=np.array([0, 0, -sweep_rad * 0.8]),
                    scale=np.array([1, 1, 1])
                ),
                material=Material(
                    name="canard_material",
                    radar_reflectivity=0.3,
                    ir_emissivity=0.7
                )
            )
            
            # Right canard (mirror of left)
            right_canard = PrimitiveGeometry(
                type=PrimitiveType.WEDGE,
                dimensions={
                    "length": self.wingspan * 0.15,
                    "width_start": root_chord * 0.4,
                    "width_end": root_chord * 0.2,
                    "height": root_chord * 0.05
                },
                transform=Transform(
                    position=np.array([self.length * 0.2, 0, 0]),
                    rotation=np.array([0, 0, sweep_rad * 0.8]),
                    scale=np.array([1, 1, 1])
                ),
                material=Material(
                    name="canard_material",
                    radar_reflectivity=0.3,
                    ir_emissivity=0.7
                )
            )
            
            ucav.add_component(left_canard)
            ucav.add_component(right_canard)
        
        return ucav
    
    def get_geometry_data(self) -> Dict[str, Any]:
        """Get geometry data for simulation and visualization."""
        if self.complex_geometry is None:
            return {}
        return self.complex_geometry.to_dict()
    
    def get_cross_section(self, angle: float) -> float:
        """
        Calculate cross-section area from a specific viewing angle.
        
        Args:
            angle: Viewing angle in radians (0 = front, pi/2 = side)
            
        Returns:
            Cross-section area in square meters
        """
        # Simplified cross-section calculation
        # For a more accurate calculation, we would project all geometry onto a plane
        
        # Front view (angle near 0)
        if abs(angle) < 0.2:
            height = self.mean_chord * 0.2  # Approximate height
            width = self.wingspan
            return height * width * 0.7  # Scaling factor for realistic area
            
        # Side view (angle near pi/2)
        elif abs(angle - math.pi/2) < 0.2:
            height = self.mean_chord * 0.2  # Approximate height
            width = self.length
            return height * width * 0.8  # Scaling factor for realistic area
            
        # Top view (angle near pi)
        elif abs(angle - math.pi) < 0.2:
            # Wing area calculation
            wing_area = self.wingspan * self.mean_chord
            return wing_area
            
        # Interpolate for other angles
        else:
            front_area = self.mean_chord * 0.2 * self.wingspan * 0.7
            side_area = self.mean_chord * 0.2 * self.length * 0.8
            
            # Simple interpolation
            if angle < math.pi/2:
                factor = angle / (math.pi/2)
                return front_area * (1 - factor) + side_area * factor
            else:
                factor = (angle - math.pi/2) / (math.pi/2)
                return side_area * (1 - factor) + (self.wingspan * self.mean_chord) * factor
    
    def get_radar_cross_section(self, azimuth: float, elevation: float) -> float:
        """
        Calculate radar cross-section (RCS) for given viewing angles.
        
        Args:
            azimuth: Horizontal viewing angle in radians
            elevation: Vertical viewing angle in radians
            
        Returns:
            RCS in square meters
        """
        # Get base cross-section
        base_cross_section = self.get_cross_section(azimuth)
        
        # Apply stealth characteristics
        stealth_factor = 0.1  # Base stealth factor
        
        # Front aspect has lowest RCS due to design
        if abs(azimuth) < 0.3:
            stealth_factor *= 0.2
        
        # Side aspect has higher RCS
        elif abs(azimuth - math.pi/2) < 0.3:
            stealth_factor *= 1.5
            
        # Apply elevation effects (looking from above/below)
        if abs(elevation) > 0.3:
            stealth_factor *= 1.2
            
        # Calculate final RCS
        return base_cross_section * stealth_factor
    
    def get_ir_signature(self, power_level: float = 0.5) -> Dict[str, float]:
        """
        Calculate infrared signature based on power level.
        
        Args:
            power_level: Engine power level (0.0 to 1.0)
            
        Returns:
            Dictionary with IR signature data
        """
        # Base heat signature
        base_heat = 20.0 + 80.0 * power_level  # Temperature in degrees C above ambient
        
        # Calculate signature for different aspects
        return {
            "rear": base_heat * 1.0,  # Full signature from rear
            "side": base_heat * 0.6,  # Reduced from side
            "front": base_heat * 0.3,  # Minimal from front
            "top": base_heat * 0.7,   # Significant from top
            "bottom": base_heat * 0.5  # Moderate from bottom
        }
    
    def export_for_cfd(self) -> Dict[str, Any]:
        """Export geometry in a format suitable for CFD analysis."""
        return {
            "type": "ucav",
            "dimensions": {
                "length": self.length,
                "wingspan": self.wingspan,
                "mean_chord": self.mean_chord,
                "sweep_angle": self.sweep_angle,
                "taper_ratio": self.taper_ratio
            },
            "features": {
                "has_vertical_tail": self.has_vertical_tail,
                "has_canards": self.has_canards
            },
            "geometry": self.complex_geometry.to_dict() if self.complex_geometry else {}
        }
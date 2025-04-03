#!/usr/bin/env python3
"""
UCAV Aerodynamics Simulation Model

A simplified aerodynamics model for Unmanned Combat Aerial Vehicles.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import logging
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger
from src.simulation.aerodynamics.cfd_engine import CFDEngine

logger = get_logger("ucav_aerodynamics")


@dataclass
class UCAVGeometry:
    """Geometric parameters of a UCAV."""
    
    # Main dimensions
    length: float  # Aircraft length (m)
    wingspan: float  # Wing span (m)
    wing_area: float  # Wing reference area (m²)
    
    # Wing parameters
    aspect_ratio: float  # Wing aspect ratio
    taper_ratio: float  # Wing taper ratio
    sweep_angle: float  # Wing sweep angle (degrees)
    
    # Control surfaces
    has_vertical_tail: bool = False
    has_canards: bool = False
    
    def __post_init__(self):
        """Compute derived parameters."""
        # Mean aerodynamic chord
        self.mean_chord = self.wing_area / self.wingspan
        
        # Aspect ratio validation
        if self.aspect_ratio <= 0:
            self.aspect_ratio = (self.wingspan ** 2) / self.wing_area


class UCAVAerodynamicsModel:
    """
    Simplified aerodynamics model for UCAV simulation.
    """
    
    def __init__(self, geometry: UCAVGeometry):
        """
        Initialize the UCAV aerodynamics model.
        
        Args:
            geometry: UCAV geometric parameters
        """
        self.geometry = geometry
        
        # Default aerodynamic coefficients
        self.cd0 = 0.015  # Zero-lift drag coefficient
        self.cl_alpha = 4.0  # Lift curve slope (per radian)
        self.cm0 = -0.02  # Zero-lift pitching moment
        self.cm_alpha = -0.5  # Pitching moment slope (per radian)
        
        # Oswald efficiency factor
        self.e = 0.85
        
        # Initialize CFD engine for detailed simulations
        self.cfd = None
        
        logger.info(f"UCAV aerodynamics model initialized with wingspan {geometry.wingspan}m")
    
    def setup_cfd(self, grid_size: Tuple[int, int, int] = (60, 40, 40),
             domain_size: Tuple[float, float, float] = (30.0, 20.0, 20.0)):
        """
        Set up CFD engine for detailed simulations.
        
        Args:
            grid_size: CFD grid size
            domain_size: CFD domain size
        """
        self.cfd = CFDEngine(grid_size, domain_size)
        
        # Position UCAV in the domain
        ucav_position = (5.0, domain_size[1]/2 - self.geometry.wingspan/2, domain_size[2]/2)
        
        # Create detailed UCAV geometry for CFD
        ucav_object = self._create_detailed_geometry(ucav_position)
        
        # Set boundary conditions with more realistic parameters
        inflow_velocity = (30.0, 0.0, 0.0)  # 30 m/s in x direction
        self.cfd.set_boundary_conditions(inflow_velocity=inflow_velocity, objects=[ucav_object])
        
        logger.info("CFD engine set up for UCAV simulation with detailed geometry")

    def _create_detailed_geometry(self, position: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Create detailed UCAV geometry for CFD simulation.
        
        Args:
            position: Position of the UCAV in the domain
            
        Returns:
            Dict containing detailed geometry description
        """
        # Extract geometric parameters
        length = self.geometry.length
        wingspan = self.geometry.wingspan
        mean_chord = self.geometry.mean_chord
        sweep_angle = self.geometry.sweep_angle
        taper_ratio = self.geometry.taper_ratio
        
        # Calculate wing root and tip chord
        root_chord = 2 * mean_chord / (1 + taper_ratio)
        tip_chord = root_chord * taper_ratio
        
        # Create wing geometry
        wing = {
            'type': 'swept_wing',
            'position': position,
            'root_chord': root_chord,
            'tip_chord': tip_chord,
            'wingspan': wingspan,
            'sweep_angle': sweep_angle,
            'airfoil': 'NACA64A010'  # Default airfoil for UCAV
        }
        
        # Create fuselage geometry
        fuselage = {
            'type': 'fuselage',
            'position': position,
            'length': length,
            'max_width': root_chord * 0.3,
            'max_height': root_chord * 0.2
        }
        
        # Add control surfaces if present
        components = [wing, fuselage]
        
        if self.geometry.has_vertical_tail:
            vertical_tail = {
                'type': 'vertical_tail',
                'position': (position[0] + length * 0.8, position[1], position[2]),
                'height': wingspan * 0.15,
                'root_chord': root_chord * 0.5,
                'tip_chord': root_chord * 0.3
            }
            components.append(vertical_tail)
        
        if self.geometry.has_canards:
            canards = {
                'type': 'canard',
                'position': (position[0] + length * 0.2, position[1], position[2]),
                'span': wingspan * 0.3,
                'root_chord': root_chord * 0.4,
                'tip_chord': root_chord * 0.2,
                'sweep_angle': sweep_angle * 0.8
            }
            components.append(canards)
        
        # Return complete UCAV geometry
        return {
            'type': 'composite',
            'components': components
        }
    
    def calculate_coefficients(self, alpha: float, beta: float, mach: float) -> Dict[str, float]:
        """
        Calculate aerodynamic coefficients using simplified models.
        
        Args:
            alpha: Angle of attack (degrees)
            beta: Sideslip angle (degrees)
            mach: Mach number
            
        Returns:
            Dict[str, float]: Dictionary of aerodynamic coefficients
        """
        # Convert angles to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        
        # Basic lift coefficient (thin airfoil theory with corrections)
        cl = self.cl_alpha * alpha_rad
        
        # Limit CL for stall (simplified)
        cl_max = 1.2
        if cl > cl_max:
            cl = cl_max * np.sin(np.pi * cl / (2 * cl_max))
        
        # Induced drag coefficient
        cdi = (cl ** 2) / (np.pi * self.geometry.aspect_ratio * self.e)
        
        # Wave drag (simplified for transonic/supersonic)
        cdw = 0.0
        if mach > 0.8:
            cdw = 0.1 * (mach - 0.8) ** 2
        
        # Total drag
        cd = self.cd0 + cdi + cdw
        
        # Side force coefficient (simplified)
        cy = -0.5 * beta_rad
        
        # Pitching moment
        cm = self.cm0 + self.cm_alpha * alpha_rad
        
        # Rolling moment (simplified)
        cl_roll = -0.1 * beta_rad
        
        # Yawing moment (simplified)
        cn = 0.1 * beta_rad
        
        # Return all coefficients
        return {
            'CL': cl,
            'CD': cd,
            'CY': cy,
            'Cm': cm,
            'Cl': cl_roll,
            'Cn': cn
        }
    
    def calculate_forces(self, velocity: float, altitude: float, 
                        alpha: float, beta: float) -> Dict[str, float]:
        """
        Calculate aerodynamic forces and moments.
        
        Args:
            velocity: Airspeed (m/s)
            altitude: Altitude (m)
            alpha: Angle of attack (degrees)
            beta: Sideslip angle (degrees)
            
        Returns:
            Dict[str, float]: Dictionary of forces and moments
        """
        # Calculate air density based on altitude (simplified model)
        rho = 1.225 * np.exp(-altitude / 10000)
        
        # Calculate dynamic pressure
        q = 0.5 * rho * velocity ** 2
        
        # Calculate Mach number (simplified)
        temperature = 288.15 - 0.0065 * altitude  # Standard atmosphere temperature (K)
        speed_of_sound = np.sqrt(1.4 * 287 * temperature)
        mach = velocity / speed_of_sound
        
        # Get aerodynamic coefficients
        coeffs = self.calculate_coefficients(alpha, beta, mach)
        
        # Calculate forces and moments
        forces = {
            'lift': coeffs['CL'] * q * self.geometry.wing_area,
            'drag': coeffs['CD'] * q * self.geometry.wing_area,
            'side_force': coeffs['CY'] * q * self.geometry.wing_area,
            'pitch_moment': coeffs['Cm'] * q * self.geometry.wing_area * self.geometry.mean_chord,
            'roll_moment': coeffs['Cl'] * q * self.geometry.wing_area * self.geometry.wingspan,
            'yaw_moment': coeffs['Cn'] * q * self.geometry.wing_area * self.geometry.wingspan
        }
        
        return forces
    
    def run_cfd_simulation(self, steps: int = 100) -> Dict[str, Any]:
        """
        Run CFD simulation for more detailed aerodynamic analysis.
        
        Args:
            steps: Number of simulation steps
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        if self.cfd is None:
            logger.error("CFD engine not initialized. Call setup_cfd() first.")
            return {}
        
        # Run simulation
        for _ in range(steps):
            self.cfd.step()
        
        # Get surface indices (simplified)
        # In a real implementation, this would be derived from the UCAV geometry
        surface_indices = []
        grid_size = self.cfd.grid_size
        
        # Create a simple approximation of the UCAV surface
        center_x = int(grid_size[0] * 0.2)  # 20% into the domain
        center_y = int(grid_size[1] * 0.5)  # Center of domain
        center_z = int(grid_size[2] * 0.5)  # Center of domain
        
        # Wing surface points (very simplified)
        wing_span_cells = int(self.geometry.wingspan / self.cfd.dy)
        wing_length_cells = int(self.geometry.length / self.cfd.dx)
        
        for i in range(center_x, center_x + wing_length_cells):
            for j in range(center_y - wing_span_cells//2, center_y + wing_span_cells//2):
                surface_indices.append((i, j, center_z))
        
        # Compute forces
        forces = self.cfd.compute_forces(surface_indices)
        
        # Get flow field data
        flow_field = self.cfd.get_flow_field()
        
        return {
            'forces': forces,
            'flow_field': flow_field,
            'simulation_time': self.cfd.time
        }


def create_default_ucav_model() -> UCAVAerodynamicsModel:
    """
    Create a default UCAV aerodynamics model with typical parameters.
    
    Returns:
        UCAVAerodynamicsModel: Default UCAV model
    """
    # Create geometry for a typical UCAV (similar to X-47B)
    geometry = UCAVGeometry(
        length=11.6,           # meters
        wingspan=18.9,         # meters
        wing_area=50.0,        # square meters
        aspect_ratio=7.14,     # wingspan^2 / area
        taper_ratio=0.3,       # tip chord / root chord
        sweep_angle=35.0,      # degrees
        has_vertical_tail=False,
        has_canards=False
    )
    
    # Create and return model
    return UCAVAerodynamicsModel(geometry)
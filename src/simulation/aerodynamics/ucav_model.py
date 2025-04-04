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
from dataclasses import dataclass, field

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
    
    # Morphing capabilities
    has_morphing_surfaces: bool = False
    morphing_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute derived parameters."""
        # Mean aerodynamic chord
        self.mean_chord = self.wing_area / self.wingspan
        
        # Aspect ratio validation
        if self.aspect_ratio <= 0:
            self.aspect_ratio = (self.wingspan ** 2) / self.wing_area
            
        # Initialize default morphing configuration if needed
        if self.has_morphing_surfaces and not self.morphing_config:
            self.morphing_config = {
                "wing_twist": 0.0,  # degrees
                "camber_morphing": 0.0,  # 0-1 scale
                "span_extension": 0.0,  # % of wingspan
                "sweep_morphing": 0.0,  # degrees
                "wingtip_morphing": 0.0,  # 0-1 scale
                "flexible_skin": {
                    "elasticity": 0.5,  # 0-1 scale
                    "thickness_variation": 0.0  # mm
                }
            }


@dataclass
class FlexibleSurfaceProperties:
    """Properties of flexible/morphing surfaces."""
    
    # Material properties
    youngs_modulus: float  # MPa
    poisson_ratio: float
    density: float  # kg/m³
    thickness: float  # mm
    
    # Actuation properties
    max_deformation: float  # mm
    response_time: float  # seconds
    energy_consumption: float  # W/cm²
    
    # Aerodynamic effects
    drag_reduction: float  # % reduction at optimal configuration
    lift_enhancement: float  # % enhancement at optimal configuration
    
    # Operational limits
    max_speed: float  # m/s
    max_temperature: float  # °C
    fatigue_cycles: int  # estimated cycles before replacement


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
        
        # Flexible surface properties (if applicable)
        self.flexible_properties = None
        if geometry.has_morphing_surfaces:
            self.flexible_properties = FlexibleSurfaceProperties(
                youngs_modulus=200.0,  # MPa (typical for composite materials)
                poisson_ratio=0.3,
                density=1600.0,  # kg/m³
                thickness=0.5,  # mm
                max_deformation=50.0,  # mm
                response_time=0.5,  # seconds
                energy_consumption=0.2,  # W/cm²
                drag_reduction=8.0,  # %
                lift_enhancement=12.0,  # %
                max_speed=300.0,  # m/s
                max_temperature=120.0,  # °C
                fatigue_cycles=100000
            )
            logger.info("Initialized with flexible/morphing surface capabilities")
        
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
        
        # Apply morphing effects if enabled
        if self.geometry.has_morphing_surfaces:
            # Apply span extension
            span_extension = self.geometry.morphing_config.get("span_extension", 0.0)
            wingspan *= (1.0 + span_extension / 100.0)
            
            # Apply sweep morphing
            sweep_morphing = self.geometry.morphing_config.get("sweep_morphing", 0.0)
            sweep_angle += sweep_morphing
            
            # Apply camber morphing (will affect airfoil shape)
            camber_morphing = self.geometry.morphing_config.get("camber_morphing", 0.0)
        
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
        
        # Add flexible surface properties if applicable
        if self.geometry.has_morphing_surfaces:
            wing['flexible_surface'] = {
                'enabled': True,
                'properties': {
                    'thickness': self.flexible_properties.thickness,
                    'youngs_modulus': self.flexible_properties.youngs_modulus,
                    'poisson_ratio': self.flexible_properties.poisson_ratio,
                    'max_deformation': self.flexible_properties.max_deformation
                },
                'morphing_state': {
                    'wing_twist': self.geometry.morphing_config.get("wing_twist", 0.0),
                    'camber_morphing': self.geometry.morphing_config.get("camber_morphing", 0.0),
                    'wingtip_morphing': self.geometry.morphing_config.get("wingtip_morphing", 0.0)
                }
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
        
        # Apply morphing surface effects if enabled
        if self.geometry.has_morphing_surfaces:
            # Get morphing configuration
            camber = self.geometry.morphing_config.get("camber_morphing", 0.0)
            wing_twist = self.geometry.morphing_config.get("wing_twist", 0.0)
            wingtip = self.geometry.morphing_config.get("wingtip_morphing", 0.0)
            
            # Adjust lift coefficient based on camber morphing
            cl_morph = cl * (1.0 + camber * self.flexible_properties.lift_enhancement / 100.0)
            
            # Adjust drag coefficient based on morphing
            # More complex model would use CFD for accurate results
            cd_reduction = (camber + wingtip) * self.flexible_properties.drag_reduction / 200.0
            cd_morph = cd * (1.0 - cd_reduction)
            
            # Update coefficients
            cl = cl_morph
            cd = cd_morph
            
            # Wing twist affects pitching moment and roll moment
            cm_twist = 0.02 * wing_twist / 5.0  # Simplified effect
        else:
            cm_twist = 0.0
        
        # Side force coefficient (simplified)
        cy = -0.5 * beta_rad
        
        # Pitching moment
        cm = self.cm0 + self.cm_alpha * alpha_rad + cm_twist
        
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
    
    def set_morphing_configuration(self, config: Dict[str, Any]) -> None:
        """
        Set the morphing surface configuration.
        
        Args:
            config: Dictionary with morphing parameters
        """
        if not self.geometry.has_morphing_surfaces:
            logger.warning("Morphing surfaces not enabled for this UCAV")
            return
            
        # Update morphing configuration
        for key, value in config.items():
            if key in self.geometry.morphing_config:
                self.geometry.morphing_config[key] = value
            elif key == "flexible_skin" and isinstance(value, dict):
                for skin_key, skin_value in value.items():
                    if skin_key in self.geometry.morphing_config.get("flexible_skin", {}):
                        self.geometry.morphing_config["flexible_skin"][skin_key] = skin_value
        
        logger.info(f"Updated morphing configuration: {config}")
    
    def optimize_morphing(self, flight_condition: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize morphing configuration for given flight conditions.
        
        Args:
            flight_condition: Dictionary with flight parameters
                (velocity, altitude, alpha, beta)
                
        Returns:
            Dict[str, Any]: Optimized morphing configuration
        """
        if not self.geometry.has_morphing_surfaces:
            logger.warning("Morphing surfaces not enabled for this UCAV")
            return {}
            
        # Extract flight conditions
        velocity = flight_condition.get("velocity", 100.0)
        altitude = flight_condition.get("altitude", 5000.0)
        alpha = flight_condition.get("alpha", 2.0)
        beta = flight_condition.get("beta", 0.0)
        
        # Calculate Mach number
        temperature = 288.15 - 0.0065 * altitude
        speed_of_sound = np.sqrt(1.4 * 287 * temperature)
        mach = velocity / speed_of_sound
        
        # Simple optimization strategy based on flight regime
        # In a real implementation, this would use more sophisticated algorithms
        
        # High speed - reduce drag
        if mach > 0.7:
            optimal_config = {
                "wing_twist": 1.0,
                "camber_morphing": 0.2,
                "span_extension": 0.0,
                "sweep_morphing": 5.0,  # Increase sweep for high speed
                "wingtip_morphing": 0.8  # Reduce vortex drag
            }
        # Low speed, high alpha - maximize lift
        elif alpha > 5.0:
            optimal_config = {
                "wing_twist": 2.0,
                "camber_morphing": 0.8,  # Increase camber for lift
                "span_extension": 5.0,  # Extend span for efficiency
                "sweep_morphing": -2.0,  # Reduce sweep for better low-speed performance
                "wingtip_morphing": 0.5
            }
        # Cruise - balance efficiency
        else:
            optimal_config = {
                "wing_twist": 0.5,
                "camber_morphing": 0.4,
                "span_extension": 2.0,
                "sweep_morphing": 0.0,
                "wingtip_morphing": 0.6
            }
        
        # Apply the optimized configuration
        self.set_morphing_configuration(optimal_config)
        
        return optimal_config
    
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
        
        # Additional analysis for flexible surfaces
        flexible_analysis = {}
        if self.geometry.has_morphing_surfaces:
            # In a real implementation, this would analyze surface deformation
            # and fluid-structure interaction
            flexible_analysis = {
                "max_deformation": 2.5,  # mm
                "surface_pressure_distribution": "uniform",  # simplified
                "energy_consumption": 0.15 * self.geometry.wing_area  # W
            }
        
        return {
            'forces': forces,
            'flow_field': flow_field,
            'simulation_time': self.cfd.time,
            'flexible_surface_analysis': flexible_analysis
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


def create_morphing_ucav_model() -> UCAVAerodynamicsModel:
    """
    Create a UCAV aerodynamics model with morphing capabilities.
    
    Returns:
        UCAVAerodynamicsModel: UCAV model with morphing surfaces
    """
    # Create geometry for a morphing UCAV
    geometry = UCAVGeometry(
        length=10.8,           # meters
        wingspan=16.5,         # meters
        wing_area=45.0,        # square meters
        aspect_ratio=6.05,     # wingspan^2 / area
        taper_ratio=0.25,      # tip chord / root chord
        sweep_angle=30.0,      # degrees
        has_vertical_tail=False,
        has_canards=True,
        has_morphing_surfaces=True,
        morphing_config={
            "wing_twist": 0.0,  # degrees
            "camber_morphing": 0.0,  # 0-1 scale
            "span_extension": 0.0,  # % of wingspan
            "sweep_morphing": 0.0,  # degrees
            "wingtip_morphing": 0.0,  # 0-1 scale
            "flexible_skin": {
                "elasticity": 0.7,  # 0-1 scale
                "thickness_variation": 0.2  # mm
            }
        }
    )
    
    # Create and return model
    return UCAVAerodynamicsModel(geometry)
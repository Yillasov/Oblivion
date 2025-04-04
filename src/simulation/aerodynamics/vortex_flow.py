#!/usr/bin/env python3
"""
Vortex-Based Flow Simulation for Biomimetic Wings
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import matplotlib.pyplot as plt

from src.core.utils.logging_framework import get_logger
from src.biomimetic.design.wing_structures import WingStructure, WingType

logger = get_logger("vortex_flow")

class VortexLatticeMethod:
    """Vortex Lattice Method for biomimetic wing analysis."""
    
    def __init__(self, num_chordwise=8, num_spanwise=16):
        self.num_chordwise = num_chordwise
        self.num_spanwise = num_spanwise
        self.panels = []
        self.vortices = []
        self.collocation_points = []
        self.normals = []
        self.circulation = None
        logger.info(f"Initialized VLM with {num_chordwise}x{num_spanwise} panels")
    
    def setup_biomimetic_wing(self, wing_structure: WingStructure):
        """Set up a biomimetic wing based on WingStructure."""
        # Extract wing parameters
        aspect_ratio = wing_structure.aspect_ratio
        span = 1.0  # Default span
        
        # Configure based on wing type
        if wing_structure.wing_type == WingType.BIRD_HIGH_ASPECT:
            taper = 0.4
            sweep = 15.0
            twist = -2.0
            camber = 0.04
        elif wing_structure.wing_type == WingType.BIRD_ELLIPTICAL:
            taper = 0.01
            sweep = 5.0
            twist = -3.0
            camber = 0.05
        elif wing_structure.wing_type == WingType.BAT_WING:
            taper = 0.3
            sweep = 20.0
            twist = -5.0
            camber = 0.06
        elif wing_structure.wing_type == WingType.INSECT_WING:
            taper = 0.6
            sweep = 10.0
            twist = 0.0
            camber = 0.02
        else:
            taper = 0.6
            sweep = 10.0
            twist = -2.0
            camber = 0.03
        
        # Generate wing geometry
        self._generate_panels(span, aspect_ratio, taper, sweep, twist, camber)
        logger.info(f"Set up {wing_structure.wing_type} wing with {len(self.panels)} panels")
    
    def _generate_panels(self, span, aspect_ratio, taper, sweep, twist, camber):
        """Generate wing panels based on geometric parameters."""
        # Clear existing geometry
        self.panels = []
        self.collocation_points = []
        self.normals = []
        self.vortices = []
        
        # Calculate chord lengths
        area = span**2 / aspect_ratio
        root_chord = 2 * area / span
        tip_chord = root_chord * taper
        
        # Generate panel grid
        for j in range(self.num_spanwise):
            y_frac = j / self.num_spanwise
            y_next_frac = (j + 1) / self.num_spanwise
            
            y = -span/2 + span * y_frac
            y_next = -span/2 + span * y_next_frac
            
            chord = root_chord + (tip_chord - root_chord) * (2 * abs(y) / span)
            chord_next = root_chord + (tip_chord - root_chord) * (2 * abs(y_next) / span)
            
            # Apply sweep
            sweep_rad = np.radians(sweep)
            x_offset = abs(y) * np.tan(sweep_rad)
            x_offset_next = abs(y_next) * np.tan(sweep_rad)
            
            # Apply twist
            twist_angle = twist * (2 * abs(y) / span)
            twist_next = twist * (2 * abs(y_next) / span)
            
            for i in range(self.num_chordwise):
                x_frac = i / self.num_chordwise
                x_next_frac = (i + 1) / self.num_chordwise
                
                # Create panel vertices
                p1 = np.array([x_offset + x_frac * chord, y, 0])
                p2 = np.array([x_offset + x_next_frac * chord, y, 0])
                p3 = np.array([x_offset_next + x_next_frac * chord_next, y_next, 0])
                p4 = np.array([x_offset_next + x_frac * chord_next, y_next, 0])
                
                # Apply camber
                if camber > 0:
                    p1[2] += 4 * camber * x_frac * (1 - x_frac) * chord
                    p2[2] += 4 * camber * x_next_frac * (1 - x_next_frac) * chord
                    p3[2] += 4 * camber * x_next_frac * (1 - x_next_frac) * chord_next
                    p4[2] += 4 * camber * x_frac * (1 - x_frac) * chord_next
                
                # Add panel
                self.panels.append([p1, p2, p3, p4])
                
                # Calculate panel normal
                v1 = p2 - p1
                v2 = p4 - p1
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                self.normals.append(normal)
                
                # Calculate collocation point (3/4 chord, center span)
                collocation = 0.25 * (p1 + p2 + p3 + p4)
                self.collocation_points.append(collocation)
                
                # Create horseshoe vortex (bound vortex at 1/4 chord)
                vortex_p1 = p1 + 0.25 * (p2 - p1)
                vortex_p2 = p4 + 0.25 * (p3 - p4)
                self.vortices.append([vortex_p1, vortex_p2])
    
    def solve(self, freestream_velocity):
        """Solve the VLM for a given freestream velocity."""
        num_panels = len(self.panels)
        
        # Initialize influence coefficient matrix
        A = np.zeros((num_panels, num_panels))
        
        # Initialize RHS vector
        b = np.zeros(num_panels)
        
        # Compute influence coefficients
        for i in range(num_panels):
            coll_point = self.collocation_points[i]
            normal = self.normals[i]
            
            # Compute influence of each vortex
            for j in range(num_panels):
                vortex = self.vortices[j]
                v_induced = self._vortex_induced_velocity(vortex, coll_point)
                A[i, j] = np.dot(v_induced, normal)
            
            # RHS is negative of freestream velocity dotted with normal
            b[i] = -np.dot(freestream_velocity, normal)
        
        # Solve linear system for circulation strengths
        try:
            self.circulation = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.circulation = np.linalg.lstsq(A, b, rcond=None)[0]
            logger.warning("Used least squares solver due to singular matrix")
        
        # Calculate forces and coefficients
        forces = self._calculate_forces(freestream_velocity)
        return forces
    
    def _vortex_induced_velocity(self, vortex, point):
        """Calculate velocity induced by a horseshoe vortex at a point."""
        # Simplified Biot-Savart calculation for horseshoe vortex
        p1, p2 = vortex
        
        # Create trailing vortices extending to infinity
        p3 = p2 + np.array([100.0, 0.0, 0.0])
        p4 = p1 + np.array([100.0, 0.0, 0.0])
        
        # Calculate induced velocity from each segment
        segments = [(p1, p2), (p2, p3), (p4, p1)]
        v_induced = np.zeros(3)
        
        for start, end in segments:
            r1 = point - start
            r2 = point - end
            
            # Vector from start to end
            r0 = end - start
            r0_mag = np.linalg.norm(r0)
            
            if r0_mag < 1e-10:
                continue
            
            # Cross product
            cross = np.cross(r1, r2)
            cross_mag = np.linalg.norm(cross)
            
            if cross_mag < 1e-10:
                continue
            
            # Dot products
            dot1 = np.dot(r1, r0 / r0_mag)
            dot2 = np.dot(r2, r0 / r0_mag)
            
            # Biot-Savart law
            v = cross * (dot1 - dot2) / (4.0 * np.pi * cross_mag**2)
            v_induced += v
        
        return v_induced
    
    def _calculate_forces(self, freestream_velocity):
        """Calculate forces using Kutta-Joukowski theorem."""
        if self.circulation is None:
            return {}
            
        num_panels = len(self.panels)
        forces = np.zeros((num_panels, 3))
        rho = 1.225  # Air density (kg/m³)
        
        for i in range(num_panels):
            vortex = self.vortices[i]
            gamma = self.circulation[i]
            
            # Calculate bound vortex vector
            bound_vortex = vortex[1] - vortex[0]
            
            # Calculate force using Kutta-Joukowski theorem: F = rho * V × Γ
            forces[i] = rho * np.cross(freestream_velocity, gamma * bound_vortex)
        
        # Calculate total forces
        total_force = np.sum(forces, axis=0)
        
        # Calculate lift and drag
        v_mag = np.linalg.norm(freestream_velocity)
        v_dir = freestream_velocity / v_mag
        
        # Wind axes
        x_wind = v_dir
        z_wind = np.array([0, 0, 1])
        y_wind = np.cross(z_wind, x_wind)
        y_wind = y_wind / np.linalg.norm(y_wind)
        z_wind = np.cross(x_wind, y_wind)
        
        # Transform force to wind axes
        force_wind = np.array([
            np.dot(total_force, x_wind),
            np.dot(total_force, y_wind),
            np.dot(total_force, z_wind)
        ])
        
        # Lift is force in z_wind direction, drag is force in -x_wind direction
        lift = force_wind[2]
        drag = -force_wind[0]
        
        # Calculate wing area
        wing_area = 0.0
        for panel in self.panels:
            v1 = panel[1] - panel[0]
            v2 = panel[3] - panel[0]
            panel_area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            wing_area += panel_area
        
        # Calculate coefficients
        q = 0.5 * rho * v_mag**2
        cl = lift / (q * wing_area) if q * wing_area > 0 else 0
        cd = drag / (q * wing_area) if q * wing_area > 0 else 0
        
        return {
            "lift": lift,
            "drag": drag,
            "cl": cl,
            "cd": cd,
            "wing_area": wing_area
        }
    
    def visualize(self):
        """Visualize the VLM geometry and results."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot panels
        for panel in self.panels:
            x = [p[0] for p in panel] + [panel[0][0]]
            y = [p[1] for p in panel] + [panel[0][1]]
            z = [p[2] for p in panel] + [panel[0][2]]
            ax.plot(x, y, z, 'k-', alpha=0.3)
        
        # Plot vortices
        for vortex in self.vortices:
            x = [vortex[0][0], vortex[1][0]]
            y = [vortex[0][1], vortex[1][1]]
            z = [vortex[0][2], vortex[1][2]]
            ax.plot(x, y, z, 'r-', alpha=0.7)
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()
        return fig, ax


class LeadingEdgeVortexModel:
    """Model for leading edge vortex simulation in biomimetic wings."""
    
    def __init__(self, wing_span, chord, angle_of_attack):
        self.wing_span = wing_span
        self.chord = chord
        self.angle_of_attack = angle_of_attack
        self.vortex_strength = 0.0
        self.vortex_position = np.zeros(3)
        self.vortex_radius = 0.1 * chord
        logger.info(f"Initialized LEV model with span={wing_span}, chord={chord}")
    
    def update(self, velocity, dt):
        """Update LEV based on current flow conditions."""
        # Calculate vortex strength based on angle of attack
        alpha_rad = np.radians(self.angle_of_attack)
        if alpha_rad > 0.1:  # LEV forms at higher angles of attack
            # Simplified model for LEV strength
            self.vortex_strength = 0.5 * velocity * self.chord * np.sin(alpha_rad)
            
            # Update vortex position (simplified)
            self.vortex_position = np.array([
                0.25 * self.chord,  # 25% chord
                0.0,                # midspan
                0.1 * self.chord    # slightly above surface
            ])
            
            # Update vortex radius based on strength
            self.vortex_radius = 0.05 * self.chord * (1 + np.sin(alpha_rad))
        else:
            # No significant LEV at low angles of attack
            self.vortex_strength = 0.0
        
        return {
            "strength": self.vortex_strength,
            "position": self.vortex_position,
            "radius": self.vortex_radius
        }
    
    def calculate_lift_contribution(self, velocity):
        """Calculate lift contribution from LEV."""
        if self.vortex_strength <= 0:
            return 0.0
            
        # Simplified model based on vortex strength and span
        rho = 1.225  # Air density
        lev_lift = rho * velocity * self.vortex_strength * self.wing_span
        
        return lev_lift


# Adding Unsteady Aerodynamics Modeling Capabilities

class VortexParticle:
    """Representation of a discrete vortex particle in the flow field."""
    position: np.ndarray  # 3D position vector
    strength: np.ndarray  # 3D vorticity vector
    core_radius: float = 0.05  # Core radius for regularization
    age: float = 0.0  # Age of the particle in seconds
    
    def induced_velocity(self, point: np.ndarray) -> np.ndarray:
        """Calculate velocity induced by this vortex particle at a point."""
        r = point - self.position
        r_mag = np.linalg.norm(r)
        
        if r_mag < 1e-10:
            return np.zeros(3)
        
        # Regularized Biot-Savart kernel
        factor = 1.0 - np.exp(-(r_mag/self.core_radius)**2)
        factor *= 1.0 / (4.0 * np.pi * r_mag**3)
        
        return factor * np.cross(self.strength, r)


class VortexParticleMethod:
    """Vortex Particle Method for unsteady aerodynamic analysis."""
    
    def __init__(self, 
                 time_step: float = 0.01,
                 diffusion_coefficient: float = 0.01,
                 max_particles: int = 1000):
        """
        Initialize the Vortex Particle Method.
        
        Args:
            time_step: Simulation time step
            diffusion_coefficient: Vorticity diffusion coefficient
            max_particles: Maximum number of particles
        """
        self.time_step = time_step
        self.diffusion_coefficient = diffusion_coefficient
        self.max_particles = max_particles
        
        # Simulation state
        self.particles = []  # List of vortex particles
        self.time = 0.0
        self.surface_panels = []  # Surface panels for boundary conditions
        self.surface_normals = []  # Surface panel normals
        
        # Simulation parameters
        self.freestream_velocity = np.array([1.0, 0.0, 0.0])
        self.rho = 1.225  # Air density (kg/m³)
        
        logger.info(f"Initialized VPM with dt={time_step}, max particles={max_particles}")
    
    def add_particle(self, position: np.ndarray, strength: np.ndarray, 
                    core_radius: float = 0.05) -> None:
        """Add a vortex particle to the simulation."""
        if len(self.particles) < self.max_particles:
            self.particles.append(VortexParticle(
                position=position.copy(),
                strength=strength.copy(),
                core_radius=core_radius
            ))
    
    def setup_wing(self, vlm: VortexLatticeMethod) -> None:
        """Set up wing geometry from a VLM model."""
        self.surface_panels = vlm.panels.copy()
        self.surface_normals = vlm.normals.copy()
        
        # Seed initial vortex particles along trailing edge
        for i, vortex in enumerate(vlm.vortices):
            if vlm.circulation is not None:
                gamma = vlm.circulation[i]
            else:
                gamma = 1.0  # Default circulation if VLM not solved
                
            # Trailing edge midpoint
            te_mid = 0.5 * (vortex[0] + vortex[1])
            
            # Trailing edge vector (spanwise)
            te_vec = vortex[1] - vortex[0]
            te_length = np.linalg.norm(te_vec)
            
            if te_length > 0:
                # Create vortex particle with strength proportional to circulation
                strength = gamma * te_vec / te_length
                
                # Add particle slightly behind trailing edge
                offset = 0.05 * np.array([1.0, 0.0, 0.0])  # Small offset behind TE
                self.add_particle(
                    position=te_mid + offset,
                    strength=strength,
                    core_radius=0.05 * te_length
                )
        
        logger.info(f"Set up wing with {len(self.surface_panels)} panels and seeded {len(self.particles)} particles")
    
    def calculate_induced_velocity(self, point: np.ndarray) -> np.ndarray:
        """Calculate velocity induced by all vortex particles at a point."""
        v_induced = np.zeros(3)
        
        for particle in self.particles:
            v_induced += particle.induced_velocity(point)
        
        return v_induced
    
    def step(self) -> None:
        """Advance the simulation by one time step."""
        # Update time
        self.time += self.time_step
        
        # Update particle positions
        new_particles = []
        for particle in self.particles:
            # Calculate total velocity at particle position
            velocity = self.freestream_velocity + self.calculate_induced_velocity(particle.position)
            
            # Update position using Euler integration
            new_position = particle.position + velocity * self.time_step
            
            # Update age
            new_age = particle.age + self.time_step
            
            # Apply vortex stretching and diffusion (simplified)
            new_strength = particle.strength * (1.0 - self.diffusion_coefficient * self.time_step)
            
            # Create new particle if strength is still significant
            if np.linalg.norm(new_strength) > 0.01:
                new_particles.append(VortexParticle(
                    position=new_position,
                    strength=new_strength,
                    core_radius=particle.core_radius,
                    age=new_age
                ))
        
        # Replace old particles with new ones
        self.particles = new_particles
        
        # Shed new particles from trailing edge
        self._shed_new_particles()
        
        logger.debug(f"Time step {self.time:.3f}s: {len(self.particles)} particles")
    
    def _shed_new_particles(self) -> None:
        """Shed new vortex particles from the trailing edge."""
        # This is a simplified shedding model
        # In a full implementation, this would be based on the current circulation
        # distribution and the no-penetration boundary condition
        
        # For now, just add a few particles at the trailing edge with random strength
        if len(self.particles) < self.max_particles and len(self.surface_panels) > 0:
            # Find trailing edge panels
            for i in range(min(5, len(self.surface_panels))):
                panel = self.surface_panels[i]
                
                # Approximate trailing edge position
                te_pos = 0.5 * (panel[1] + panel[2])
                
                # Add small random perturbation
                perturbation = np.random.normal(0, 0.01, 3)
                
                # Create vortex strength (simplified)
                strength = np.array([0.0, 0.0, 0.1]) + np.random.normal(0, 0.05, 3)
                
                # Add new particle
                self.add_particle(
                    position=te_pos + np.array([0.05, 0.0, 0.0]) + perturbation,
                    strength=strength,
                    core_radius=0.05
                )


class DynamicStallModel:
    """Dynamic stall model for unsteady aerodynamics."""
    
    def __init__(self, 
                 airfoil_type: str = "NACA0012",
                 reynolds_number: float = 1e6):
        """
        Initialize the dynamic stall model.
        
        Args:
            airfoil_type: Type of airfoil
            reynolds_number: Reynolds number
        """
        self.airfoil_type = airfoil_type
        self.reynolds_number = reynolds_number
        
        # Static stall angle (degrees)
        self.static_stall_angle = 12.0
        
        # State variables
        self.current_alpha = 0.0  # Current angle of attack (degrees)
        self.previous_alpha = 0.0  # Previous angle of attack
        self.alpha_dot = 0.0  # Rate of change of angle of attack
        self.vortex_strength = 0.0  # LEV strength
        self.vortex_position = 0.25  # Chordwise position (x/c)
        self.separation_point = 1.0  # Trailing edge separation point (x/c)
        self.stall_state = "attached"  # Current stall state
        
        logger.info(f"Initialized dynamic stall model for {airfoil_type} at Re={reynolds_number}")
    
    def update(self, 
              alpha: float, 
              dt: float,
              velocity: float) -> Dict[str, float]:
        """
        Update the dynamic stall model.
        
        Args:
            alpha: Current angle of attack (degrees)
            dt: Time step
            velocity: Airspeed
            
        Returns:
            Dictionary with updated aerodynamic coefficients
        """
        # Store previous values
        self.previous_alpha = self.current_alpha
        self.current_alpha = alpha
        
        # Calculate rate of change of angle of attack
        self.alpha_dot = (self.current_alpha - self.previous_alpha) / dt
        
        # Reduced frequency
        reduced_freq = 0.5 * abs(self.alpha_dot) * 0.25 / velocity
        
        # Update stall state
        self._update_stall_state(reduced_freq)
        
        # Calculate dynamic lift and drag coefficients
        cl_static = self._static_cl(alpha)
        cd_static = self._static_cd(alpha)
        
        # Apply dynamic effects
        cl_dynamic = cl_static
        cd_dynamic = cd_static
        cm_dynamic = -0.05  # Default value
        
        if self.stall_state == "attached":
            # Attached flow - small dynamic effects
            cl_dynamic += 0.1 * reduced_freq * np.sign(self.alpha_dot)
            cm_dynamic -= 0.02 * reduced_freq * np.sign(self.alpha_dot)
        
        elif self.stall_state == "developing_vortex":
            # Developing LEV - increased lift
            vortex_effect = 0.5 * self.vortex_strength
            cl_dynamic += vortex_effect
            cd_dynamic += 0.1 * vortex_effect
            cm_dynamic -= 0.1 * vortex_effect * (self.vortex_position - 0.25)
        
        elif self.stall_state == "vortex_shedding":
            # Vortex shedding - fluctuating forces
            cl_dynamic *= 0.8  # Reduced lift
            cd_dynamic *= 1.5  # Increased drag
            cm_dynamic -= 0.15  # Nose-down pitching moment
        
        elif self.stall_state == "fully_stalled":
            # Fully stalled - reduced lift, increased drag
            cl_dynamic *= 0.6
            cd_dynamic *= 2.0
            cm_dynamic -= 0.2
        
        return {
            "cl": cl_dynamic,
            "cd": cd_dynamic,
            "cm": cm_dynamic,
            "stall_state": self.stall_state,
            "vortex_strength": self.vortex_strength,
            "vortex_position": self.vortex_position,
            "separation_point": self.separation_point
        }
    
    def _update_stall_state(self, reduced_freq: float) -> None:
        """Update the stall state based on current conditions."""
        alpha_abs = abs(self.current_alpha)
        alpha_dot_abs = abs(self.alpha_dot)
        
        # Dynamic stall angle increases with pitch rate
        dynamic_stall_angle = self.static_stall_angle + 5.0 * reduced_freq
        
        if alpha_abs < self.static_stall_angle:
            # Attached flow
            self.stall_state = "attached"
            self.vortex_strength = 0.0
            self.separation_point = 1.0
        
        elif alpha_abs < dynamic_stall_angle:
            # Developing vortex
            self.stall_state = "developing_vortex"
            self.vortex_strength = (alpha_abs - self.static_stall_angle) / (dynamic_stall_angle - self.static_stall_angle)
            self.vortex_position = 0.25 + 0.3 * self.vortex_strength
            self.separation_point = 1.0 - 0.4 * self.vortex_strength
        
        elif alpha_dot_abs > 10.0:
            # Vortex shedding during rapid pitch
            self.stall_state = "vortex_shedding"
            self.vortex_strength = 0.8
            self.vortex_position = 0.6
            self.separation_point = 0.5
        
        else:
            # Fully stalled
            self.stall_state = "fully_stalled"
            self.vortex_strength = 0.2
            self.vortex_position = 0.8
            self.separation_point = 0.3
    
    def _static_cl(self, alpha: float) -> float:
        """Calculate static lift coefficient."""
        alpha_rad = np.radians(alpha)
        if abs(alpha) < self.static_stall_angle:
            return 2 * np.pi * np.sin(alpha_rad) * np.cos(alpha_rad)
        else:
            # Post-stall behavior
            sign = np.sign(alpha)
            return sign * (0.8 + 0.2 * np.exp(-(abs(alpha) - self.static_stall_angle) / 5.0))
    
    def _static_cd(self, alpha: float) -> float:
        """Calculate static drag coefficient."""
        alpha_rad = np.radians(alpha)
        cd_min = 0.01
        return cd_min + 0.02 * alpha_rad**2 + 0.1 * (1 - np.cos(4 * alpha_rad))


class UnsteadyAerodynamicsModel:
    """Combined model for unsteady aerodynamics simulation."""
    
    def __init__(self, wing_structure: WingStructure):
        """
        Initialize the unsteady aerodynamics model.
        
        Args:
            wing_structure: Wing structure parameters
        """
        self.wing_structure = wing_structure
        
        # Initialize VLM for steady-state solution
        self.vlm = VortexLatticeMethod()
        self.vlm.setup_biomimetic_wing(wing_structure)
        
        # Initialize vortex particle method for wake evolution
        self.vpm = VortexParticleMethod()
        
        # Initialize dynamic stall model
        self.dynamic_stall = DynamicStallModel()
        
        # Initialize LEV model
        self.lev = LeadingEdgeVortexModel(
            wing_span=wing_structure.aspect_ratio,
            chord=1.0,
            angle_of_attack=5.0
        )
        
        # Simulation state
        self.time = 0.0
        self.dt = 0.01
        self.initialized = False
        
        logger.info(f"Initialized unsteady aerodynamics model for {wing_structure.wing_type}")
    
    def initialize(self, freestream_velocity: np.ndarray) -> None:
        """Initialize the simulation with a steady-state solution."""
        # Solve VLM for initial circulation distribution
        self.vlm.solve(freestream_velocity)
        
        # Set up VPM with wing geometry and initial vortex particles
        self.vpm.freestream_velocity = freestream_velocity
        self.vpm.setup_wing(self.vlm)
        
        self.initialized = True
        logger.info("Initialized unsteady simulation with steady-state solution")
    
    def step(self, 
            freestream_velocity: np.ndarray,
            angle_of_attack: float) -> Dict[str, Any]:
        """
        Advance the simulation by one time step.
        
        Args:
            freestream_velocity: Current freestream velocity
            angle_of_attack: Current angle of attack (degrees)
            
        Returns:
            Dictionary with aerodynamic forces and coefficients
        """
        if not self.initialized:
            self.initialize(freestream_velocity)
        
        # Update time
        self.time += self.dt
        
        # Update vortex particle simulation
        self.vpm.freestream_velocity = freestream_velocity
        self.vpm.step()
        
        # Update dynamic stall model
        airspeed = np.linalg.norm(freestream_velocity)
        stall_results = self.dynamic_stall.update(angle_of_attack, self.dt, airspeed)
        
        # Update LEV model
        lev_results = self.lev.update(airspeed, self.dt)
        
        # Calculate total forces and coefficients
        # This is a simplified approach - a full implementation would
        # integrate forces over the entire wing surface
        
        # Get baseline forces from VLM
        vlm_forces = self.vlm.solve(freestream_velocity)
        
        # Apply unsteady corrections
        cl = vlm_forces["cl"] * (1.0 + 0.2 * stall_results["vortex_strength"])
        cd = vlm_forces["cd"] * (1.0 + 0.5 * (1.0 - stall_results["separation_point"]))
        
        # Add LEV contribution
        lev_lift = self.lev.calculate_lift_contribution(airspeed)
        
        # Calculate forces
        q = 0.5 * self.vpm.rho * airspeed**2
        lift = cl * q * vlm_forces["wing_area"] + lev_lift
        drag = cd * q * vlm_forces["wing_area"]
        
        return {
            "lift": lift,
            "drag": drag,
            "cl": cl,
            "cd": cd,
            "cm": stall_results["cm"],
            "stall_state": stall_results["stall_state"],
            "vortex_strength": stall_results["vortex_strength"],
            "time": self.time,
            "num_particles": len(self.vpm.particles)
        }
    
    def visualize(self, ax=None):
        """Visualize the unsteady flow field."""
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Plot wing geometry
        for panel in self.vlm.panels:
            x = [p[0] for p in panel] + [panel[0][0]]
            y = [p[1] for p in panel] + [panel[0][1]]
            z = [p[2] for p in panel] + [panel[0][2]]
            ax.plot(x, y, z, 'k-', alpha=0.3)
        
        # Plot vortex particles
        if self.initialized:
            for particle in self.vpm.particles:
                # Scale marker size by vortex strength
                size = 30 * np.linalg.norm(particle.strength)
                ax.scatter(
                    particle.position[0], 
                    particle.position[1], 
                    particle.position[2],
                    c='r', s=size, alpha=0.6
                )
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = 0
        for panel in self.vlm.panels:
            for p in panel:
                max_range = max(max_range, np.linalg.norm(p))
        
        ax.set_xlim([-0.2*max_range, 1.2*max_range])
        ax.set_ylim([-0.7*max_range, 0.7*max_range])
        ax.set_zlim([-0.7*max_range, 0.7*max_range])
        
        plt.tight_layout()
        return fig, ax


# Updated integration function for UCAV model
def integrate_with_ucav_model(ucav_model, wing_structure):
    """Integrate vortex flow simulation with UCAV model."""
    # Create unsteady aerodynamics model
    unsteady_model = UnsteadyAerodynamicsModel(wing_structure)
    
    # Return integrated models
    return {
        "vlm": unsteady_model.vlm,
        "lev": unsteady_model.lev,
        "unsteady": unsteady_model
    }
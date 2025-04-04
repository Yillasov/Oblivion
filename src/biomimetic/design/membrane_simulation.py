#!/usr/bin/env python3
"""
Flexible Membrane Simulation

This module provides simulation capabilities for flexible membranes used in
biomimetic designs, particularly for insect and bat wing structures.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger

logger = get_logger("membrane_simulation")


@dataclass
class MembraneProperties:
    """Physical properties of a flexible membrane."""
    thickness: float  # mm
    youngs_modulus: float  # MPa
    poisson_ratio: float
    density: float  # kg/m³
    damping_coefficient: float
    anisotropic_ratio: float = 1.0  # Ratio of stiffness in primary vs secondary direction
    fiber_orientation: Optional[float] = None  # Degrees from x-axis
    pretension: Optional[Tuple[float, float]] = None  # N/m in x and y directions


@dataclass
class MembraneSimConfig:
    """Configuration for membrane simulation."""
    time_step: float = 0.001  # seconds
    gravity: float = 9.81  # m/s²
    air_density: float = 1.225  # kg/m³
    convergence_threshold: float = 1e-6
    max_iterations: int = 100
    enable_aero_effects: bool = True
    enable_contact: bool = True


class MembraneSimulation:
    """Finite element simulation for flexible membranes."""
    
    def __init__(self, 
                properties: MembraneProperties, 
                config: MembraneSimConfig,
                mesh_resolution: Tuple[int, int] = (20, 20),
                membrane_dimensions: Tuple[float, float] = (0.1, 0.1)):  # meters
        """
        Initialize membrane simulation.
        
        Args:
            properties: Physical properties of the membrane
            config: Simulation configuration
            mesh_resolution: Number of elements in x and y directions
            membrane_dimensions: Physical dimensions in x and y directions (meters)
        """
        self.properties = properties
        self.config = config
        self.mesh_resolution = mesh_resolution
        self.dimensions = membrane_dimensions
        
        # Initialize mesh and state variables
        self._initialize_mesh()
        self._initialize_state()
        
        logger.info(f"Initialized membrane simulation with {self.num_nodes} nodes")
    
    def _initialize_mesh(self) -> None:
        """Initialize the finite element mesh."""
        nx, ny = self.mesh_resolution
        self.num_nodes = (nx + 1) * (ny + 1)
        self.num_elements = nx * ny
        
        # Node positions (x, y, z)
        self.nodes = np.zeros((self.num_nodes, 3))
        
        # Element connectivity (node indices for each element)
        self.elements = np.zeros((self.num_elements, 4), dtype=int)
        
        # Create node positions
        dx = self.dimensions[0] / nx
        dy = self.dimensions[1] / ny
        
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_idx = j * (nx + 1) + i
                self.nodes[node_idx, 0] = i * dx
                self.nodes[node_idx, 1] = j * dy
                self.nodes[node_idx, 2] = 0.0  # Initial z position
        
        # Create element connectivity
        for j in range(ny):
            for i in range(nx):
                elem_idx = j * nx + i
                self.elements[elem_idx, 0] = j * (nx + 1) + i
                self.elements[elem_idx, 1] = j * (nx + 1) + i + 1
                self.elements[elem_idx, 2] = (j + 1) * (nx + 1) + i + 1
                self.elements[elem_idx, 3] = (j + 1) * (nx + 1) + i
    
    def _initialize_state(self) -> None:
        """Initialize simulation state variables."""
        # Displacement, velocity, and acceleration vectors
        self.displacements = np.zeros((self.num_nodes, 3))
        self.velocities = np.zeros((self.num_nodes, 3))
        self.accelerations = np.zeros((self.num_nodes, 3))
        
        # Forces
        self.internal_forces = np.zeros((self.num_nodes, 3))
        self.external_forces = np.zeros((self.num_nodes, 3))
        
        # Mass matrix (diagonal)
        element_area = (self.dimensions[0] * self.dimensions[1]) / self.num_elements
        node_mass = element_area * self.properties.thickness * 1e-3 * self.properties.density / 4
        self.mass_matrix = np.ones(self.num_nodes) * node_mass
        
        # Boundary conditions (fixed nodes)
        self.fixed_nodes = set()
        
        # Simulation time
        self.time = 0.0
    
    def set_boundary_conditions(self, fixed_nodes: List[int]) -> None:
        """
        Set boundary conditions by specifying fixed nodes.
        
        Args:
            fixed_nodes: List of node indices to fix
        """
        self.fixed_nodes = set(fixed_nodes)
    
    def apply_force(self, node_idx: int, force: Tuple[float, float, float]) -> None:
        """
        Apply external force to a specific node.
        
        Args:
            node_idx: Node index
            force: Force vector (x, y, z) in Newtons
        """
        self.external_forces[node_idx] += np.array(force)
    
    def apply_pressure(self, pressure: float) -> None:
        """
        Apply uniform pressure to the membrane.
        
        Args:
            pressure: Pressure in Pascals (positive = pushing up)
        """
        nx, ny = self.mesh_resolution
        element_area = (self.dimensions[0] * self.dimensions[1]) / self.num_elements
        force_per_node = pressure * element_area / 4  # Distribute to 4 nodes per element
        
        for elem_idx in range(self.num_elements):
            for node_idx in self.elements[elem_idx]:
                # Calculate normal vector (simplified as +z for now)
                normal = np.array([0.0, 0.0, 1.0])
                self.external_forces[node_idx] += force_per_node * normal
    
    def apply_aerodynamic_forces(self, airflow_velocity: Tuple[float, float, float]) -> None:
        """
        Apply aerodynamic forces based on membrane shape and airflow.
        
        Args:
            airflow_velocity: Airflow velocity vector (x, y, z) in m/s
        """
        if not self.config.enable_aero_effects:
            return
            
        airflow = np.array(airflow_velocity)
        airflow_speed = np.linalg.norm(airflow)
        if airflow_speed < 1e-6:
            return
            
        # Dynamic pressure
        q = 0.5 * self.config.air_density * airflow_speed**2
        
        # For each element, calculate aerodynamic force
        for elem_idx in range(self.num_elements):
            # Get nodes for this element
            nodes = [self.elements[elem_idx, i] for i in range(4)]
            
            # Calculate element normal vector using cross product
            v1 = self.nodes[nodes[1]] - self.nodes[nodes[0]]
            v2 = self.nodes[nodes[3]] - self.nodes[nodes[0]]
            normal = np.cross(v1, v2)
            normal_length = np.linalg.norm(normal)
            
            if normal_length > 1e-10:
                normal = normal / normal_length
                
                # Calculate element area
                element_area = 0.5 * normal_length
                
                # Calculate angle of attack
                cos_aoa = np.dot(normal, -airflow) / airflow_speed
                
                # Simple aerodynamic model (coefficient varies with angle of attack)
                cl = 2.0 * np.sin(np.arccos(cos_aoa)) * cos_aoa
                
                # Calculate force magnitude
                force_mag = q * element_area * cl
                
                # Force direction is perpendicular to surface
                force = force_mag * normal
                
                # Distribute force to element nodes
                for node_idx in nodes:
                    self.external_forces[node_idx] += force / 4
    
    def _calculate_internal_forces(self) -> None:
        """Calculate internal elastic forces based on membrane deformation."""
        # Reset internal forces
        self.internal_forces.fill(0.0)
        
        # Material properties
        E = self.properties.youngs_modulus * 1e6  # Convert to Pa
        nu = self.properties.poisson_ratio
        t = self.properties.thickness * 1e-3  # Convert to meters
        
        # Plane stress material matrix
        D = np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ]) * (E / (1 - nu**2))
        
        # For each element
        for elem_idx in range(self.num_elements):
            # Get nodes for this element
            node_indices = self.elements[elem_idx]
            
            # Get node coordinates (current configuration)
            x = np.zeros((4, 3))
            for i in range(4):
                node_idx = node_indices[i]
                x[i] = self.nodes[node_idx] + self.displacements[node_idx]
            
            # Calculate element strains and stresses (simplified)
            # This is a very simplified model - a real implementation would use
            # proper finite element formulations
            
            # Calculate element dimensions
            dx = np.mean([x[1, 0] - x[0, 0], x[2, 0] - x[3, 0]])
            dy = np.mean([x[3, 1] - x[0, 1], x[2, 1] - x[1, 1]])
            
            # Calculate element deformation gradients
            dz_dx = np.mean([
                (x[1, 2] - x[0, 2]) / dx,
                (x[2, 2] - x[3, 2]) / dx
            ])
            dz_dy = np.mean([
                (x[3, 2] - x[0, 2]) / dy,
                (x[2, 2] - x[1, 2]) / dy
            ])
            
            # Simplified strain calculation
            strain = np.array([
                0.5 * dz_dx**2,  # εxx
                0.5 * dz_dy**2,  # εyy
                dz_dx * dz_dy    # γxy
            ])
            
            # Calculate stress
            stress = D @ strain
            
            # Calculate nodal forces from stress (simplified)
            area = dx * dy
            
            # Force magnitudes
            fx = stress[0] * area * t
            fy = stress[1] * area * t
            fxy = stress[2] * area * t
            
            # Distribute forces to nodes (simplified)
            forces = np.array([
                [-fx/2 - fxy/2, -fy/2 - fxy/2, 0],
                [fx/2 - fxy/2, -fy/2 + fxy/2, 0],
                [fx/2 + fxy/2, fy/2 + fxy/2, 0],
                [-fx/2 + fxy/2, fy/2 - fxy/2, 0]
            ])
            
            # Add z-component based on membrane curvature
            for i in range(4):
                forces[i, 2] = -np.sqrt(fx**2 + fy**2) * (dz_dx + dz_dy) / 4
            
            # Apply forces to nodes
            for i in range(4):
                node_idx = node_indices[i]
                self.internal_forces[node_idx] += forces[i]
    
    def step(self) -> Dict[str, Any]:
        """
        Advance simulation by one time step.
        
        Returns:
            Dictionary with simulation state
        """
        dt = self.config.time_step
        
        # Calculate internal forces
        self._calculate_internal_forces()
        
        # Calculate total forces
        total_forces = self.external_forces - self.internal_forces
        
        # Add gravity
        total_forces[:, 2] -= self.mass_matrix * self.config.gravity
        
        # Calculate accelerations (F = ma)
        for i in range(self.num_nodes):
            if i not in self.fixed_nodes:
                self.accelerations[i] = total_forces[i] / self.mass_matrix[i]
        
        # Verlet integration
        self.displacements += self.velocities * dt + 0.5 * self.accelerations * dt**2
        old_accelerations = self.accelerations.copy()
        
        # Enforce boundary conditions
        for node_idx in self.fixed_nodes:
            self.displacements[node_idx] = 0.0
            self.velocities[node_idx] = 0.0
            self.accelerations[node_idx] = 0.0
        
        # Recalculate forces and accelerations
        self._calculate_internal_forces()
        total_forces = self.external_forces - self.internal_forces
        total_forces[:, 2] -= self.mass_matrix * self.config.gravity
        
        for i in range(self.num_nodes):
            if i not in self.fixed_nodes:
                self.accelerations[i] = total_forces[i] / self.mass_matrix[i]
        
        # Update velocities
        self.velocities += 0.5 * (old_accelerations + self.accelerations) * dt
        
        # Add damping
        self.velocities *= (1.0 - self.properties.damping_coefficient * dt)
        
        # Update time
        self.time += dt
        
        # Reset external forces for next step
        self.external_forces.fill(0.0)
        
        return {
            "time": self.time,
            "displacements": self.displacements.copy(),
            "velocities": self.velocities.copy(),
            "max_displacement": np.max(np.abs(self.displacements)),
            "strain_energy": self._calculate_strain_energy()
        }
    
    def _calculate_strain_energy(self) -> float:
        """Calculate total strain energy in the membrane."""
        # Simplified calculation
        return 0.5 * np.sum(self.internal_forces * self.displacements)
    
    def get_node_positions(self) -> np.ndarray:
        """
        Get current node positions.
        
        Returns:
            Array of node positions (N x 3)
        """
        return self.nodes + self.displacements
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.displacements.fill(0.0)
        self.velocities.fill(0.0)
        self.accelerations.fill(0.0)
        self.internal_forces.fill(0.0)
        self.external_forces.fill(0.0)
        self.time = 0.0


class BatWingMembrane(MembraneSimulation):
    """Specialized membrane simulation for bat wings."""
    
    def __init__(self, 
                properties: MembraneProperties,
                config: MembraneSimConfig,
                digit_positions: List[Tuple[float, float, float]],
                membrane_dimensions: Tuple[float, float] = (0.2, 0.15)):
        """
        Initialize bat wing membrane simulation.
        
        Args:
            properties: Physical properties of the membrane
            config: Simulation configuration
            digit_positions: 3D positions of bat wing digits
            membrane_dimensions: Physical dimensions in x and y directions
        """
        super().__init__(properties, config, (30, 20), membrane_dimensions)
        
        self.digit_positions = digit_positions
        self._setup_bat_wing_structure()
    
    def _setup_bat_wing_structure(self) -> None:
        """Set up bat wing structure with digits as constraints."""
        # Find nodes closest to digit positions
        digit_nodes = []
        for pos in self.digit_positions:
            # Find closest node
            distances = np.sum((self.nodes - np.array(pos))**2, axis=1)
            closest_node = np.argmin(distances)
            digit_nodes.append(closest_node)
        
        # Set these nodes as controlled by digit positions
        self.digit_nodes = digit_nodes
        
        # Initialize digit velocities
        self.digit_velocities = np.zeros((len(digit_nodes), 3))
    
    def update_digit_positions(self, new_positions: List[Tuple[float, float, float]]) -> None:
        """
        Update positions of digits controlling the membrane.
        
        Args:
            new_positions: New 3D positions for each digit
        """
        if len(new_positions) != len(self.digit_nodes):
            raise ValueError("Number of positions must match number of digits")
            
        dt = self.config.time_step
        
        for i, (node_idx, new_pos) in enumerate(zip(self.digit_nodes, new_positions)):
            old_pos = self.nodes[node_idx] + self.displacements[node_idx]
            new_pos = np.array(new_pos)
            
            # Calculate velocity
            self.digit_velocities[i] = (new_pos - old_pos) / dt
            
            # Update displacement to match new position
            self.displacements[node_idx] = new_pos - self.nodes[node_idx]
    
    def step(self) -> Dict[str, Any]:
        """
        Advance simulation by one time step with digit constraints.
        
        Returns:
            Dictionary with simulation state
        """
        # Store original digit positions and velocities
        for i, node_idx in enumerate(self.digit_nodes):
            # Force digits to follow prescribed motion
            self.velocities[node_idx] = self.digit_velocities[i]
        
        # Run standard step
        result = super().step()
        
        # Enforce digit positions
        for i, node_idx in enumerate(self.digit_nodes):
            # Ensure displacements maintain digit positions
            current_pos = self.nodes[node_idx] + self.displacements[node_idx]
            target_pos = self.nodes[node_idx] + self.displacements[node_idx]
            self.displacements[node_idx] = target_pos - self.nodes[node_idx]
        
        return result


class InsectWingMembrane(MembraneSimulation):
    """Specialized membrane simulation for insect wings with venation patterns."""
    
    def __init__(self, 
                properties: MembraneProperties,
                config: MembraneSimConfig,
                venation_pattern: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                membrane_dimensions: Tuple[float, float] = (0.05, 0.02)):
        """
        Initialize insect wing membrane simulation.
        
        Args:
            properties: Physical properties of the membrane
            config: Simulation configuration
            venation_pattern: List of vein segments as pairs of (x,y) coordinates
            membrane_dimensions: Physical dimensions in x and y directions
        """
        super().__init__(properties, config, (40, 20), membrane_dimensions)
        
        self.venation_pattern = venation_pattern
        self._setup_venation_structure()
    
    def _setup_venation_structure(self) -> None:
        """Set up insect wing venation structure."""
        # Find nodes closest to vein segments
        vein_nodes = set()
        
        for start, end in self.venation_pattern:
            # Convert to numpy arrays
            start_point = np.array([start[0], start[1], 0])
            end_point = np.array([end[0], end[1], 0])
            
            # Find nodes along this vein segment
            for node_idx in range(self.num_nodes):
                node_pos = self.nodes[node_idx]
                
                # Calculate distance from node to line segment
                line_vec = end_point - start_point
                line_length = np.linalg.norm(line_vec)
                line_unit_vec = line_vec / line_length
                
                node_to_start = node_pos - start_point
                projection = np.dot(node_to_start, line_unit_vec)
                
                # Check if projection is on the line segment
                if 0 <= projection <= line_length:
                    # Calculate perpendicular distance
                    perp_vec = node_to_start - projection * line_unit_vec
                    distance = np.linalg.norm(perp_vec)
                    
                    # If node is close to vein, add to vein nodes
                    if distance < 0.002:  # 2mm threshold
                        vein_nodes.add(node_idx)
        
        self.vein_nodes = list(vein_nodes)
        
        # Increase stiffness for vein nodes
        self.vein_stiffness_factor = 10.0
    
    def _calculate_internal_forces(self) -> None:
        """Calculate internal forces with modified stiffness for veins."""
        super()._calculate_internal_forces()
        
        # Increase forces for vein nodes to simulate higher stiffness
        for node_idx in self.vein_nodes:
            self.internal_forces[node_idx] *= self.vein_stiffness_factor


def create_dragonfly_wing_simulation() -> InsectWingMembrane:
    """Create a simulation of a dragonfly wing."""
    # Define membrane properties
    properties = MembraneProperties(
        thickness=0.01,  # 0.01mm
        youngs_modulus=1000.0,  # 1000 MPa
        poisson_ratio=0.3,
        density=1200.0,  # 1200 kg/m³
        damping_coefficient=0.1,
        anisotropic_ratio=2.0,
        fiber_orientation=30.0  # 30 degrees
    )
    
    # Define simulation config
    config = MembraneSimConfig(
        time_step=0.0005,
        enable_aero_effects=True
    )
    
    # Define venation pattern (simplified)
    venation_pattern = [
        # Leading edge
        ((0.0, 0.01), (0.05, 0.01)),
        # Main longitudinal veins
        ((0.0, 0.005), (0.05, 0.005)),
        ((0.0, 0.015), (0.05, 0.015)),
        # Cross veins
        ((0.01, 0.005), (0.01, 0.015)),
        ((0.02, 0.005), (0.02, 0.015)),
        ((0.03, 0.005), (0.03, 0.015)),
        ((0.04, 0.005), (0.04, 0.015))
    ]
    
    # Create simulation
    sim = InsectWingMembrane(
        properties=properties,
        config=config,
        venation_pattern=venation_pattern,
        membrane_dimensions=(0.05, 0.02)  # 5cm x 2cm
    )
    
    # Set boundary conditions (fix the wing base)
    fixed_nodes = []
    for i in range(sim.num_nodes):
        if sim.nodes[i, 0] < 0.005:  # First 5mm is the wing base
            fixed_nodes.append(i)
    
    sim.set_boundary_conditions(fixed_nodes)
    
    return sim


def create_bat_wing_simulation() -> BatWingMembrane:
    """Create a simulation of a bat wing."""
    # Define membrane properties
    properties = MembraneProperties(
        thickness=0.1,  # 0.1mm
        youngs_modulus=10.0,  # 10 MPa
        poisson_ratio=0.4,
        density=1100.0,  # 1100 kg/m³
        damping_coefficient=0.2,
        anisotropic_ratio=1.5,
        fiber_orientation=45.0  # 45 degrees
    )
    
    # Define simulation config
    config = MembraneSimConfig(
        time_step=0.001,
        enable_aero_effects=True,
        enable_contact=True
    )
    
    # Define digit positions (simplified)
    digit_positions = [
        (0.0, 0.0, 0.0),    # Wing base
        (0.1, 0.0, 0.0),    # Digit 1
        (0.15, 0.05, 0.0),  # Digit 2
        (0.18, 0.1, 0.0),   # Digit 3
        (0.2, 0.15, 0.0)    # Digit 4
    ]
    
    # Create simulation
    sim = BatWingMembrane(
        properties=properties,
        config=config,
        digit_positions=digit_positions,
        membrane_dimensions=(0.2, 0.15)  # 20cm x 15cm
    )
    
    # Set boundary conditions (fix the wing base)
    fixed_nodes = []
    for i in range(sim.num_nodes):
        if sim.nodes[i, 0] < 0.02 and sim.nodes[i, 1] < 0.02:
            fixed_nodes.append(i)
    
    sim.set_boundary_conditions(fixed_nodes)
    
    return sim


def run_simulation_demo() -> Dict[str, Any]:
    """Run a demonstration of the membrane simulation."""
    # Create dragonfly wing simulation
    dragonfly_sim = create_dragonfly_wing_simulation()
    
    # Apply airflow
    airflow = (5.0, 0.0, 0.0)  # 5 m/s in x direction
    
    # Run simulation for 100 steps
    results = []
    for _ in range(100):
        dragonfly_sim.apply_aerodynamic_forces(airflow)
        result = dragonfly_sim.step()
        results.append(result)
    
    # Return final state
    return {
        "final_time": results[-1]["time"],
        "max_displacement": results[-1]["max_displacement"],
        "final_shape": dragonfly_sim.get_node_positions()
    }


if __name__ == "__main__":
    # Run demo simulation
    result = run_simulation_demo()
    print(f"Simulation completed at t={result['final_time']:.3f}s")
    print(f"Maximum displacement: {result['max_displacement']:.6f}m")
"""
Physics-based simulation model for landing gear systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from src.landing_gear.base import NeuromorphicLandingGear, LandingGearSpecs, TelemetryData
from src.landing_gear.types import LandingGearType
from src.landing_gear.implementations import create_landing_gear


class TerrainType(Enum):
    """Types of terrain for landing simulation."""
    CONCRETE = "concrete"
    ASPHALT = "asphalt"
    GRASS = "grass"
    DIRT = "dirt"
    GRAVEL = "gravel"
    SAND = "sand"
    SNOW = "snow"
    ICE = "ice"
    WATER = "water"


@dataclass
class LandingEnvironment:
    """Environmental conditions for landing simulation."""
    terrain_type: TerrainType = TerrainType.CONCRETE
    terrain_friction: float = 0.8  # 0.0 to 1.0
    terrain_elasticity: float = 0.2  # 0.0 to 1.0
    terrain_roughness: float = 0.1  # 0.0 to 1.0
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # radians
    temperature: float = 15.0  # Celsius
    air_density: float = 1.225  # kg/m³
    gravity: float = 9.81  # m/s²
    altitude: float = 0.0  # meters above sea level


class LandingGearSimulation:
    """Unified physics-based simulation for all landing gear types."""
    
    def __init__(self, landing_gear: NeuromorphicLandingGear, environment: LandingEnvironment):
        """Initialize the simulation with a landing gear and environment."""
        self.landing_gear = landing_gear
        self.environment = environment
        self.time_step = 0.01  # seconds
        self.current_time = 0.0
        self.state = self._initialize_state()
        self.collision_points = self._generate_collision_points()
        self.telemetry_history = []
        self.max_telemetry_history = 1000
    
    def _initialize_state(self) -> Dict[str, Any]:
        """Initialize simulation state."""
        return {
            "position": np.zeros(3),  # x, y, z in meters
            "velocity": np.zeros(3),  # m/s
            "acceleration": np.zeros(3),  # m/s²
            "orientation": np.zeros(3),  # roll, pitch, yaw in radians
            "angular_velocity": np.zeros(3),  # rad/s
            "angular_acceleration": np.zeros(3),  # rad/s²
            "forces": np.zeros(3),  # N
            "moments": np.zeros(3),  # N·m
            "ground_contact": False,
            "compression": 0.0,  # 0.0 to 1.0
            "load": 0.0,  # N
            "energy_absorption": 0.0,  # J
            "stability": 1.0,  # 0.0 to 1.0
        }
    
    def _generate_collision_points(self) -> List[Dict[str, Any]]:
        """Generate collision points based on landing gear type."""
        gear_type = self.landing_gear.specs.gear_type
        dimensions = self.landing_gear.specs.dimensions
        
        # Default collision points (for most gear types)
        points = [
            {"position": np.array([0.0, 0.0, 0.0]), "radius": 0.1, "spring_constant": 50000.0, "damping": 5000.0}
        ]
        
        # Type-specific collision points
        if gear_type == LandingGearType.RETRACTABLE_MORPHING:
            # Main gear and nose gear configuration
            points = [
                {"position": np.array([-1.0, -1.0, 0.0]), "radius": 0.2, "spring_constant": 80000.0, "damping": 8000.0},
                {"position": np.array([-1.0, 1.0, 0.0]), "radius": 0.2, "spring_constant": 80000.0, "damping": 8000.0},
                {"position": np.array([2.0, 0.0, 0.0]), "radius": 0.15, "spring_constant": 60000.0, "damping": 6000.0}
            ]
        elif gear_type == LandingGearType.AIR_CUSHION:
            # Air cushion with distributed contact points
            cushion_radius = dimensions.get("radius", 1.0)
            points = []
            for i in range(8):
                angle = i * np.pi / 4
                pos = np.array([cushion_radius * np.cos(angle), cushion_radius * np.sin(angle), 0.0])
                points.append({"position": pos, "radius": 0.3, "spring_constant": 20000.0, "damping": 2000.0})
            # Center point
            points.append({"position": np.array([0.0, 0.0, 0.0]), "radius": 0.3, "spring_constant": 20000.0, "damping": 2000.0})
        
        return points
    
    def update(self, aircraft_state: Dict[str, Any], control_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Update simulation for one time step."""
        # Extract relevant aircraft state
        self.state["position"] = aircraft_state.get("position", np.zeros(3))
        self.state["velocity"] = aircraft_state.get("velocity", np.zeros(3))
        self.state["orientation"] = aircraft_state.get("orientation", np.zeros(3))
        aircraft_mass = aircraft_state.get("mass", 1000.0)
        
        # Apply landing gear control inputs
        if control_inputs.get("deploy", False) and not self.landing_gear.status["deployed"]:
            self.landing_gear.deploy()
        elif control_inputs.get("retract", False) and self.landing_gear.status["deployed"]:
            self.landing_gear.retract()
        
        # Skip physics if gear is not deployed
        if not self.landing_gear.status["deployed"]:
            self._record_telemetry(aircraft_state)
            self.current_time += self.time_step
            return self.state
        
        # Calculate forces and moments from landing gear
        forces, moments = self._calculate_gear_forces(aircraft_state)
        
        # Apply environmental forces (wind, etc.)
        env_forces = self._calculate_environmental_forces()
        total_forces = forces + env_forces
        
        # Update state
        self.state["forces"] = total_forces
        self.state["moments"] = moments
        
        # Calculate accelerations (F = ma)
        self.state["acceleration"] = total_forces / aircraft_mass
        
        # Record telemetry
        self._record_telemetry(aircraft_state)
        
        # Update time
        self.current_time += self.time_step
        
        return self.state
    
    def _calculate_gear_forces(self, aircraft_state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate forces and moments from landing gear."""
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        
        # Get aircraft properties
        aircraft_position = aircraft_state.get("position", np.zeros(3))
        aircraft_velocity = aircraft_state.get("velocity", np.zeros(3))
        aircraft_orientation = aircraft_state.get("orientation", np.zeros(3))
        
        # Create rotation matrix from aircraft orientation
        roll, pitch, yaw = aircraft_orientation
        # Replace simplified rotation matrix with proper implementation
        cos_r, cos_p, cos_y = np.cos([roll, pitch, yaw])
        sin_r, sin_p, sin_y = np.sin([roll, pitch, yaw])
        
        # Proper 3D rotation matrix
        rotation_matrix = np.array([
            [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
            [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
            [-sin_p, cos_p*sin_r, cos_p*cos_r]
        ])
        
        # Ground height at aircraft position (simplified)
        ground_height = 0.0
        
        # Check for ground contact and calculate forces for each collision point
        ground_contact = False
        total_compression = 0.0
        total_load = 0.0
        
        for point in self.collision_points:
            # Transform point to world coordinates
            point_pos_local = point["position"]
            point_pos_world = aircraft_position + rotation_matrix @ point_pos_local
            
            # Check for ground contact
            height_above_ground = point_pos_world[2] - ground_height
            if height_above_ground <= point["radius"]:
                ground_contact = True
                
                # Calculate penetration depth
                penetration = point["radius"] - height_above_ground
                total_compression += penetration / point["radius"]
                
                # Calculate spring force (F = -kx)
                spring_constant = point["spring_constant"]
                spring_force = np.array([0, 0, spring_constant * penetration])
                
                # Calculate damping force (F = -cv)
                damping = point["damping"]
                relative_velocity = aircraft_velocity.copy()
                relative_velocity[2] = min(relative_velocity[2], 0)  # Only consider downward velocity
                damping_force = -damping * relative_velocity
                
                # Calculate friction force
                if aircraft_velocity[0] != 0 or aircraft_velocity[1] != 0:
                    horizontal_velocity = np.array([aircraft_velocity[0], aircraft_velocity[1], 0])
                    horizontal_speed = np.linalg.norm(horizontal_velocity)
                    friction_direction = -horizontal_velocity / horizontal_speed
                    friction_force_magnitude = self.environment.terrain_friction * spring_force[2]
                    friction_force = np.array([
                        friction_direction[0] * friction_force_magnitude,
                        friction_direction[1] * friction_force_magnitude,
                        0
                    ])
                else:
                    friction_force = np.zeros(3)
                
                # Sum forces
                point_force = spring_force + damping_force + friction_force
                total_force += point_force
                total_load += spring_force[2]
                
                # Calculate moment (torque = r × F)
                moment_arm = point_pos_local
                point_moment = np.cross(moment_arm, point_force)
                total_moment += point_moment
        
        # Update state with ground contact info
        self.state["ground_contact"] = ground_contact
        if len(self.collision_points) > 0:
            self.state["compression"] = total_compression / len(self.collision_points)
        self.state["load"] = total_load
        
        # Apply gear-specific physics based on type
        gear_type = self.landing_gear.specs.gear_type
        
        # Add missing import for AdaptiveShockGear and AirCushionGear
        from src.landing_gear.implementations import AdaptiveShockGear, AirCushionGear
        
        if gear_type == LandingGearType.ADAPTIVE_SHOCK_ABSORBING:
            # Get current stiffness setting
            stiffness = 0.5  # Default
            if isinstance(self.landing_gear, AdaptiveShockGear):
                stiffness = self.landing_gear.shock_stiffness
            
            # Adjust forces based on stiffness
            damping_factor = 1.0 - (stiffness * 0.5)  # Lower stiffness = more damping
            total_force[2] *= (0.7 + stiffness * 0.3)  # Adjust vertical force
            
        elif gear_type == LandingGearType.AIR_CUSHION:
            # Air cushion provides more distributed force and better damping
            if isinstance(self.landing_gear, AirCushionGear) and self.landing_gear.cushion_inflated:
                pressure = self.landing_gear.cushion_pressure
                # Soften impact based on cushion pressure
                total_force[2] *= (0.6 + pressure * 0.4)
                # Better stability on rough terrain
                if self.environment.terrain_roughness > 0.5:
                    total_moment *= (1.0 - pressure * 0.5)  # Reduce moments for better stability
        
        return total_force, total_moment
    
    def _calculate_environmental_forces(self) -> np.ndarray:
        """Calculate forces from environmental factors."""
        forces = np.zeros(3)
        
        # Wind force (simplified)
        wind_speed = self.environment.wind_speed
        wind_direction = self.environment.wind_direction
        
        if wind_speed > 0:
            wind_force_magnitude = 0.5 * self.environment.air_density * wind_speed**2 * 0.1  # Simplified drag equation
            wind_force = np.array([
                wind_force_magnitude * np.cos(wind_direction),
                wind_force_magnitude * np.sin(wind_direction),
                0
            ])
            forces += wind_force
        
        return forces
    
    def _record_telemetry(self, aircraft_state: Dict[str, Any]) -> None:
        """Record telemetry data for this simulation step."""
        # Get telemetry from landing gear
        gear_telemetry = self.landing_gear.get_telemetry()
        
        # Enhance with simulation data
        gear_telemetry.load = self.state["load"]
        
        # Fix potential type error by ensuring vibration is a float
        acceleration_norm = np.linalg.norm(self.state["acceleration"])
        gear_telemetry.vibration = float(acceleration_norm * 0.01)
        
        # Add simulation-specific data
        gear_telemetry.additional_data.update({
            "ground_contact": self.state["ground_contact"],
            "compression": float(self.state["compression"]),  # Ensure float type
            "energy_absorption": float(self.state["energy_absorption"]),  # Ensure float type
            "stability": float(self.state["stability"]),  # Ensure float type
            "simulation_time": float(self.current_time)  # Ensure float type
        })
        
        # Store telemetry
        self.telemetry_history.append(gear_telemetry)
        
        # Trim history if needed
        if len(self.telemetry_history) > self.max_telemetry_history:
            self.telemetry_history = self.telemetry_history[-self.max_telemetry_history:]
    
    def get_telemetry_history(self) -> List[TelemetryData]:
        """Get the telemetry history."""
        return self.telemetry_history
    
    def reset(self) -> None:
        """Reset the simulation."""
        self.current_time = 0.0
        self.state = self._initialize_state()
        self.telemetry_history = []
        
    def calculate_landing_metrics(self) -> Dict[str, float]:
        """Calculate metrics for landing performance."""
        if not self.telemetry_history:
            return {"landing_quality": 0.0}
        
        # Find the landing moment (first ground contact)
        landing_index = 0
        found_landing = False
        for i, telemetry in enumerate(self.telemetry_history):
            if telemetry.additional_data.get("ground_contact", False):
                landing_index = i
                found_landing = True
                break
        
        # Return early if no landing detected
        if not found_landing:
            return {"landing_quality": 0.0}
        
        # Extract relevant telemetry around landing
        landing_telemetry = self.telemetry_history[landing_index:min(landing_index + 100, len(self.telemetry_history))]
        
        if not landing_telemetry:
            return {"landing_quality": 0.0}
        
        # Calculate metrics
        max_load = max([t.load for t in landing_telemetry])
        max_vibration = max([t.vibration for t in landing_telemetry])
        avg_stability = sum([t.additional_data.get("stability", 1.0) for t in landing_telemetry]) / len(landing_telemetry)
        
        # Calculate landing quality (0.0 to 1.0, higher is better)
        load_score = 1.0 - min(1.0, max_load / (self.landing_gear.specs.max_load_capacity * 1.5))
        vibration_score = 1.0 - min(1.0, max_vibration / 1.0)  # Assuming 1.0 is very high vibration
        stability_score = avg_stability
        
        # Weighted landing quality
        landing_quality = 0.4 * load_score + 0.4 * vibration_score + 0.2 * stability_score
        
        return {
            "landing_quality": landing_quality,
            "load_score": load_score,
            "vibration_score": vibration_score,
            "stability_score": stability_score,
            "max_load": max_load,
            "max_vibration": max_vibration,
            "avg_stability": avg_stability
        }
    
    def run_landing_simulation(self, initial_state: Dict[str, Any], duration: float = 10.0) -> Dict[str, Any]:
        """Run a complete landing simulation."""
        self.reset()
        
        # Initialize aircraft state - ensure numpy arrays for vector quantities
        aircraft_state = {
            "position": np.array(initial_state.get("position", np.array([0.0, 0.0, 10.0]))),
            "velocity": np.array(initial_state.get("velocity", np.array([0.0, 0.0, -2.0]))),
            "orientation": np.array(initial_state.get("orientation", np.zeros(3))),
            "angular_velocity": np.array(initial_state.get("angular_velocity", np.zeros(3))),
            "mass": initial_state.get("mass", 1000.0)
        }
        
        # Deploy landing gear
        control_inputs = {"deploy": True}
        
        # Run simulation loop
        num_steps = int(duration / self.time_step)
        for _ in range(num_steps):
            # Update simulation
            updated_state = self.update(aircraft_state, control_inputs)
            
            # Update aircraft state with simple physics (for demonstration)
            aircraft_state["position"] += aircraft_state["velocity"] * self.time_step
            aircraft_state["velocity"] += (self.state["acceleration"] + np.array([0, 0, -self.environment.gravity])) * self.time_step
            
            # Stop if aircraft has landed and stopped
            if (self.state["ground_contact"] and 
                np.linalg.norm(aircraft_state["velocity"]) < 0.1 and
                self.current_time > 2.0):
                break
        
        # Calculate landing metrics
        metrics = self.calculate_landing_metrics()
        
        return {
            "metrics": metrics,
            "telemetry": self.telemetry_history,
            "final_state": self.state,
            "simulation_time": self.current_time
        }
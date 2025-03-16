from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from src.airframe.base import AirframeBase

class AirframeSimulation:
    """Physics-based simulation for airframe testing."""
    
    def __init__(self, airframe: AirframeBase, config: Dict[str, Any]):
        self.airframe = airframe
        self.config = config
        self.state = self._initialize_state()
        self.gravity = config.get("gravity", 9.81)
        self.air_density = config.get("air_density", 1.225)  # kg/m^3 at sea level
        self.time_step = config.get("time_step", 0.01)  # seconds
        
    def _initialize_state(self) -> Dict[str, np.ndarray]:
        """Initialize simulation state."""
        return {
            "position": np.zeros(3),
            "velocity": np.zeros(3),
            "acceleration": np.zeros(3),
            "orientation": np.zeros(3),  # roll, pitch, yaw in radians
            "angular_velocity": np.zeros(3),
            "forces": np.zeros(3),
            "moments": np.zeros(3)
        }
    
    def calculate_aerodynamic_forces(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate aerodynamic forces and moments."""
        velocity = self.state["velocity"]
        airspeed = np.linalg.norm(velocity)
        
        if airspeed < 0.1:
            return np.zeros(3), np.zeros(3)
        
        # Calculate angle of attack and sideslip
        if velocity[0] != 0:
            alpha = np.arctan2(velocity[2], velocity[0])
            beta = np.arcsin(velocity[1] / airspeed)
        else:
            alpha, beta = 0.0, 0.0
        
        # Get aerodynamic coefficients from airframe
        flight_conditions = {
            "airspeed": airspeed,
            "alpha": alpha,
            "beta": beta,
            "altitude": self.state["position"][2],
            "mach": airspeed / 340.0  # Approximate Mach number
        }
        
        aero_coeffs = self.airframe.calculate_aerodynamic_coefficients(flight_conditions)
        
        # Calculate forces and moments
        dynamic_pressure = 0.5 * self.air_density * airspeed**2
        reference_area = self.airframe.properties.get("reference_area", 1.0)
        
        lift = dynamic_pressure * reference_area * aero_coeffs.get("lift", 0.0)
        drag = dynamic_pressure * reference_area * aero_coeffs.get("drag", 0.0)
        side_force = dynamic_pressure * reference_area * aero_coeffs.get("side", 0.0)
        
        # Transform to body frame
        forces = np.array([-drag, side_force, -lift])
        
        # Calculate moments
        reference_length = self.airframe.properties.get("reference_length", 1.0)
        roll_moment = dynamic_pressure * reference_area * reference_length * aero_coeffs.get("roll", 0.0)
        pitch_moment = dynamic_pressure * reference_area * reference_length * aero_coeffs.get("pitch", 0.0)
        yaw_moment = dynamic_pressure * reference_area * reference_length * aero_coeffs.get("yaw", 0.0)
        
        moments = np.array([roll_moment, pitch_moment, yaw_moment])
        
        return forces, moments
    
    def update(self, control_inputs: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Update simulation state based on control inputs."""
        # Calculate aerodynamic forces and moments
        aero_forces, aero_moments = self.calculate_aerodynamic_forces()
        
        # Add gravity
        gravity_force = np.array([0, 0, self.gravity * self.airframe.properties.get("mass", 1.0)])
        
        # Add propulsion forces based on control inputs
        throttle = control_inputs.get("throttle", 0.0)
        max_thrust = self.airframe.properties.get("max_thrust", 100.0)
        thrust_force = np.array([max_thrust * throttle, 0, 0])
        
        # Sum forces and moments
        total_forces = aero_forces + gravity_force + thrust_force
        total_moments = aero_moments
        
        # Add control surface effects
        for surface, deflection in control_inputs.items():
            if surface in ["elevator", "aileron", "rudder"]:
                # Simplified control effectiveness
                effectiveness = self.airframe.properties.get(f"{surface}_effectiveness", 0.1)
                if surface == "elevator":
                    total_moments[1] += deflection * effectiveness
                elif surface == "aileron":
                    total_moments[0] += deflection * effectiveness
                elif surface == "rudder":
                    total_moments[2] += deflection * effectiveness
        
        # Update accelerations
        mass = self.airframe.properties.get("mass", 1.0)
        inertia = self.airframe.properties.get("inertia", np.array([1.0, 1.0, 1.0]))
        
        linear_acceleration = total_forces / mass
        angular_acceleration = total_moments / inertia
        
        # Update velocities
        self.state["velocity"] += linear_acceleration * self.time_step
        self.state["angular_velocity"] += angular_acceleration * self.time_step
        
        # Update position and orientation
        self.state["position"] += self.state["velocity"] * self.time_step
        self.state["orientation"] += self.state["angular_velocity"] * self.time_step
        
        # Store forces and accelerations
        self.state["forces"] = total_forces
        self.state["moments"] = total_moments
        self.state["acceleration"] = linear_acceleration
        
        return self.state
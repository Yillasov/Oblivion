"""
Enhanced Flight Dynamics Physics Engine

A lightweight physics engine for UCAV flight dynamics simulation with improved accuracy.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger
from src.simulation.aerodynamics.ucav_model import UCAVGeometry

logger = get_logger("flight_dynamics")


@dataclass
class RigidBodyState:
    """State of a rigid body in 3D space."""
    
    # Position (m) in Earth frame
    position: np.ndarray  # [x, y, z]
    
    # Velocity (m/s) in body frame
    velocity: np.ndarray  # [u, v, w]
    
    # Orientation as quaternion [w, x, y, z] (scalar first)
    quaternion: np.ndarray
    
    # Orientation as Euler angles (rad) [roll, pitch, yaw]
    orientation: np.ndarray
    
    # Angular velocity (rad/s) in body frame [p, q, r]
    angular_velocity: np.ndarray
    
    # Mass properties
    mass: float  # kg
    inertia: np.ndarray  # 3x3 inertia tensor
    
    # Control surface deflections (rad)
    control_surfaces: Optional[Dict[str, float]] = None
    
    # Fuel state
    fuel_mass: float = 0.0
    fuel_capacity: float = 0.0


class FlightDynamicsEngine:
    """
    Enhanced flight dynamics engine for UCAV simulation.
    """
    
    def __init__(self, geometry: UCAVGeometry, mass: float, inertia: Optional[np.ndarray] = None,
                fuel_capacity: float = 0.0):
        """Initialize the flight dynamics engine."""
        self.geometry = geometry
        
        # Create initial state with quaternion
        quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        # Initialize control surfaces
        controls = {
            'aileron': 0.0,
            'elevator': 0.0,
            'rudder': 0.0,
            'throttle': 0.0
        }
        
        self.state = RigidBodyState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=quat,
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3),
            mass=mass,
            inertia=inertia if inertia is not None else self._estimate_inertia(mass, geometry),
            control_surfaces=controls,
            fuel_mass=fuel_capacity,
            fuel_capacity=fuel_capacity
        )
        
        # Simulation parameters
        self.gravity = 9.81  # m/s^2
        self.dt = 0.01  # s
        
        # Atmospheric model parameters
        self.rho_sl = 1.225  # Sea level density (kg/m^3)
        self.temp_sl = 288.15  # Sea level temperature (K)
        self.pressure_sl = 101325  # Sea level pressure (Pa)
        
        # Wind model
        self.wind_velocity = np.zeros(3)  # Wind velocity in Earth frame
        self.turbulence_intensity = 0.0  # Turbulence intensity (0-1)
        
        # Ground interaction
        self.ground_elevation = 0.0  # Ground elevation (m)
        self.ground_friction = 0.02  # Ground friction coefficient
        
        # Fuel consumption model
        self.fuel_flow_rate = 0.0  # kg/s
        self.specific_fuel_consumption = 0.5  # kg/kN/hr
        
        logger.info(f"Enhanced flight dynamics engine initialized with mass {mass} kg")
    
    def _estimate_inertia(self, mass: float, geometry: UCAVGeometry) -> np.ndarray:
        """Estimate inertia tensor based on geometry."""
        # Improved estimation with wingspan and length
        length = geometry.length
        wingspan = geometry.wingspan
        chord = geometry.mean_chord
        
        # Moments of inertia with better scaling
        Ixx = mass * (0.05 * length**2 + 0.08 * wingspan**2)  # Roll inertia
        Iyy = mass * (0.15 * length**2 + 0.01 * wingspan**2)  # Pitch inertia
        Izz = mass * (0.15 * length**2 + 0.09 * wingspan**2)  # Yaw inertia
        
        # Products of inertia (simplified)
        Ixy = 0.0
        Ixz = -mass * 0.05 * length * chord  # For typical forward CG
        Iyz = 0.0
        
        # Create full inertia tensor
        return np.array([
            [Ixx, -Ixy, -Ixz],
            [-Ixy, Iyy, -Iyz],
            [-Ixz, -Iyz, Izz]
        ])
    
    def _quaternion_to_dcm(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to direction cosine matrix."""
        w, x, y, z = q
        
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    def _quaternion_derivative(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Calculate quaternion derivative from angular velocity."""
        w, x, y, z = q
        p, q, r = omega
        
        # Quaternion derivative
        q_dot = 0.5 * np.array([
            -x*p - y*q - z*r,
            w*p + y*r - z*q,
            w*q + z*p - x*r,
            w*r + x*q - y*p
        ])
        
        return q_dot
    
    def _quaternion_to_euler(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles."""
        w, x, y, z = q
        
        # Roll (phi)
        phi = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        
        # Pitch (theta)
        theta = np.arcsin(2*(w*y - z*x))
        
        # Yaw (psi)
        psi = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        return np.array([phi, theta, psi])
    
    def _atmosphere(self, altitude: float) -> Tuple[float, float, float]:
        """Calculate atmospheric properties at given altitude."""
        # International Standard Atmosphere model (simplified)
        if altitude < 11000:  # Troposphere
            temp = self.temp_sl - 0.0065 * altitude
            pressure = self.pressure_sl * (temp / self.temp_sl) ** 5.255
        else:  # Stratosphere (simplified)
            temp = 216.65
            pressure = 22632 * np.exp(-0.00015769 * (altitude - 11000))
        
        # Air density
        rho = pressure / (287.05 * temp)
        
        # Speed of sound
        a = np.sqrt(1.4 * 287.05 * temp)
        
        return rho, temp, a
    
    def _control_effectiveness(self) -> Dict[str, np.ndarray]:
        """Calculate control surface effectiveness."""
        # Simple control effectiveness model
        q_bar = 0.5 * self._get_density() * np.sum(self.state.velocity[:2]**2)
        
        # Control effectiveness coefficients
        aileron_effect = np.array([0.1, 0.0, 0.01]) * q_bar * self.geometry.wing_area * self.geometry.wingspan
        elevator_effect = np.array([0.0, 0.05, 0.0]) * q_bar * self.geometry.wing_area * self.geometry.mean_chord
        rudder_effect = np.array([0.0, 0.0, 0.03]) * q_bar * self.geometry.wing_area * self.geometry.wingspan
        
        return {
            'aileron': aileron_effect,
            'elevator': elevator_effect,
            'rudder': rudder_effect
        }
    
    def _get_density(self) -> float:
        """Get air density at current altitude."""
        altitude = -self.state.position[2]  # z is down
        rho, _, _ = self._atmosphere(altitude)
        return rho
    
    def set_controls(self, controls: Dict[str, float]):
        """Set control surface deflections."""
        if self.state.control_surfaces is None:
            self.state.control_surfaces = {}
            
        for control, value in controls.items():
            if control in self.state.control_surfaces:
                self.state.control_surfaces[control] = value
    
    def step_rk4(self, forces: Dict[str, float], moments: Dict[str, float]) -> RigidBodyState:
        """Advance simulation using 4th order Runge-Kutta integration."""
        # Current state
        x = np.concatenate([
            self.state.position,
            self.state.velocity,
            self.state.quaternion,
            self.state.angular_velocity
        ])
        
        # RK4 integration
        k1 = self.dt * self._derivatives(x, forces, moments)
        k2 = self.dt * self._derivatives(x + 0.5*k1, forces, moments)
        k3 = self.dt * self._derivatives(x + 0.5*k2, forces, moments)
        k4 = self.dt * self._derivatives(x + k3, forces, moments)
        
        # Update state
        x_new = x + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Extract updated state
        self.state.position = x_new[0:3]
        self.state.velocity = x_new[3:6]
        
        # Normalize quaternion
        self.state.quaternion = x_new[6:10]
        self.state.quaternion /= np.linalg.norm(self.state.quaternion)
        
        self.state.angular_velocity = x_new[10:13]
        
        # Update Euler angles
        self.state.orientation = self._quaternion_to_euler(self.state.quaternion)
        
        return self.state
    
    def _derivatives(self, state_vector: np.ndarray, forces: Dict[str, float], 
                    moments: Dict[str, float]) -> np.ndarray:
        """Calculate state derivatives for integration."""
        # Extract state components
        pos = state_vector[0:3]
        vel = state_vector[3:6]
        quat = state_vector[6:10]
        omega = state_vector[10:13]
        
        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)
        
        # DCM from body to earth
        dcm = self._quaternion_to_dcm(quat)
        
        # Position derivative (velocity in earth frame)
        pos_dot = dcm @ vel
        
        # Force in body frame
        force_body = np.array([
            forces.get('x', 0.0) + forces.get('thrust', 0.0) - forces.get('drag', 0.0),
            forces.get('y', 0.0) + forces.get('side_force', 0.0),
            forces.get('z', 0.0) - forces.get('lift', 0.0)
        ])
        
        # Add gravity (in body frame)
        gravity_body = dcm.T @ np.array([0, 0, self.gravity * self.state.mass])
        force_body += gravity_body
        
        # Velocity derivative (acceleration in body frame)
        vel_dot = force_body / self.state.mass - np.cross(omega, vel)
        
        # Quaternion derivative
        quat_dot = self._quaternion_derivative(quat, omega)
        
        # Moment in body frame
        moment_body = np.array([
            moments.get('roll', 0.0) + moments.get('roll_moment', 0.0),
            moments.get('pitch', 0.0) + moments.get('pitch_moment', 0.0),
            moments.get('yaw', 0.0) + moments.get('yaw_moment', 0.0)
        ])
        
        # Add control surface effects
        if self.state.control_surfaces:
            effectiveness = self._control_effectiveness()
            for control, deflection in self.state.control_surfaces.items():
                if control in effectiveness:
                    moment_body += effectiveness[control] * deflection
        
        # Angular acceleration
        I = self.state.inertia
        omega_dot = np.linalg.solve(I, moment_body - np.cross(omega, I @ omega))
        
        # Combine all derivatives
        return np.concatenate([pos_dot, vel_dot, quat_dot, omega_dot])
    
    def set_wind(self, velocity: np.ndarray, turbulence_intensity: float = 0.0):
        """Set wind conditions."""
        self.wind_velocity = velocity
        self.turbulence_intensity = max(0.0, min(1.0, turbulence_intensity))
    
    def _apply_wind_effects(self, vel_body: np.ndarray) -> np.ndarray:
        """Apply wind effects to get airspeed in body frame."""
        # Convert wind from earth to body frame
        dcm = self._quaternion_to_dcm(self.state.quaternion)
        wind_body = dcm.T @ self.wind_velocity
        
        # Add turbulence (simplified Dryden model)
        if self.turbulence_intensity > 0:
            # Scale turbulence with intensity and airspeed
            airspeed = np.linalg.norm(vel_body)
            turb_scale = self.turbulence_intensity * airspeed * 0.1
            
            # Random turbulence components
            turbulence = np.random.normal(0, turb_scale, 3)
            wind_body += turbulence
        
        # Airspeed = ground speed - wind speed
        return vel_body - wind_body
    
    def _handle_ground_interaction(self):
        """Handle ground interaction and collision."""
        # Check if below ground
        if self.state.position[2] > self.ground_elevation:
            # Calculate penetration depth
            depth = self.state.position[2] - self.ground_elevation
            
            # Apply ground reaction force
            if depth > 0:
                # Reset position to ground level
                self.state.position[2] = self.ground_elevation
                
                # Reflect vertical velocity with damping
                if self.state.velocity[2] > 0:
                    self.state.velocity[2] *= -0.5  # 50% energy loss
                
                # Apply friction to horizontal velocity
                horiz_vel = np.array([self.state.velocity[0], self.state.velocity[1], 0])
                horiz_speed = np.linalg.norm(horiz_vel)
                
                if horiz_speed > 0.1:
                    friction_decel = self.ground_friction * self.gravity
                    friction_factor = max(0, 1 - friction_decel * self.dt / horiz_speed)
                    self.state.velocity[0] *= friction_factor
                    self.state.velocity[1] *= friction_factor
    
    def _update_fuel(self):
        """Update fuel mass based on consumption."""
        if self.state.fuel_mass > 0 and self.fuel_flow_rate > 0:
            # Calculate fuel consumption for this time step
            fuel_used = self.fuel_flow_rate * self.dt
            
            # Update fuel mass
            self.state.fuel_mass = max(0.0, self.state.fuel_mass - fuel_used)
            
            # Update total mass
            self.state.mass = (self.state.mass - self.state.fuel_mass) + self.state.fuel_mass
            
            # Update inertia if needed (simplified)
            if fuel_used > 0.1:  # Only recalculate periodically
                dry_mass = self.state.mass - self.state.fuel_mass
                self.state.inertia = self._estimate_inertia(dry_mass, self.geometry)
    
    def set_throttle(self, throttle: float):
        """Set engine throttle and calculate fuel flow rate."""
        if self.state.control_surfaces is not None:
            # Clamp throttle to 0-1 range
            throttle = max(0.0, min(1.0, throttle))
            self.state.control_surfaces['throttle'] = throttle
            
            # Calculate thrust (simplified)
            max_thrust = 10000.0  # N, should be based on engine model
            thrust = throttle * max_thrust
            
            # Calculate fuel flow rate
            self.fuel_flow_rate = thrust * self.specific_fuel_consumption / 3600.0  # kg/s
    
    def step(self, forces: Dict[str, float], moments: Dict[str, float]) -> RigidBodyState:
        """Advance the simulation by one time step (uses RK4)."""
        # Update fuel state
        self._update_fuel()
        
        # Apply wind effects to get aerodynamic forces
        airspeed_body = self._apply_wind_effects(self.state.velocity)
        
        # Run RK4 integration
        state = self.step_rk4(forces, moments)
        
        # Handle ground interaction
        self._handle_ground_interaction()
        
        return state
    
    def get_airdata(self) -> Dict[str, float]:
        """Calculate air data parameters from current state."""
        # Apply wind to get true airspeed
        vel_air = self._apply_wind_effects(self.state.velocity)
        
        # Extract velocity components
        u, v, w = vel_air
        
        # Calculate airspeed
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        
        # Calculate angle of attack and sideslip (in degrees)
        alpha = np.degrees(np.arctan2(w, u)) if abs(u) > 1e-6 else 0.0
        beta = np.degrees(np.arcsin(v / airspeed)) if airspeed > 1e-6 else 0.0
        
        # Calculate altitude (negative of z position, since z is down)
        altitude = -self.state.position[2]
        
        # Get atmospheric data
        rho, temp, a = self._atmosphere(altitude)
        
        # Calculate Mach number
        mach = airspeed / a if a > 0 else 0.0
        
        # Dynamic pressure
        q = 0.5 * rho * airspeed**2
        
        return {
            'airspeed': airspeed,
            'alpha': alpha,
            'beta': beta,
            'altitude': altitude,
            'mach': mach,
            'dynamic_pressure': q,
            'density': rho,
            'temperature': temp,
            'fuel_remaining': self.state.fuel_mass,
            'fuel_percent': 100 * self.state.fuel_mass / self.state.fuel_capacity if self.state.fuel_capacity > 0 else 0
        }


def create_default_dynamics_engine(geometry: UCAVGeometry) -> FlightDynamicsEngine:
    """
    Create a default flight dynamics engine with typical parameters.
    
    Args:
        geometry: UCAV geometric parameters
        
    Returns:
        FlightDynamicsEngine: Default flight dynamics engine
    """
    # Estimate mass based on wing area (simplified)
    mass = 100.0 + 50.0 * geometry.wing_area  # kg
    
    # Create and return engine
    return FlightDynamicsEngine(geometry, mass)
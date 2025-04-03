#!/usr/bin/env python3
"""
Model Reference Adaptive Control (MRAC) for Flight Systems

Basic implementation of an adaptive flight controller using MRAC architecture.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("mrac_flight_controller")


@dataclass
class FlightState:
    """Aircraft state variables."""
    
    roll: float = 0.0    # Roll angle (rad)
    pitch: float = 0.0   # Pitch angle (rad)
    yaw: float = 0.0     # Yaw angle (rad)
    p: float = 0.0       # Roll rate (rad/s)
    q: float = 0.0       # Pitch rate (rad/s)
    r: float = 0.0       # Yaw rate (rad/s)
    v_x: float = 0.0     # Forward velocity (m/s)
    v_y: float = 0.0     # Lateral velocity (m/s)
    v_z: float = 0.0     # Vertical velocity (m/s)


class ReferenceModel:
    """Desired dynamics model."""
    
    def __init__(self, natural_freq: float = 2.0, damping: float = 0.7):
        self.wn = natural_freq
        self.zeta = damping
        
    def compute_response(self, command: float, state: float, rate: float, dt: float) -> Tuple[float, float]:
        """Compute desired state and rate."""
        # Second-order reference model
        acc = self.wn**2 * (command - state) - 2 * self.zeta * self.wn * rate
        new_rate = rate + acc * dt
        new_state = state + new_rate * dt
        return new_state, new_rate


class AdaptiveController:
    """MRAC-based flight controller."""
    
    def __init__(self):
        """Initialize adaptive controller."""
        # Adaptation gains
        self.gamma_x = 1.0  # State feedback adaptation gain
        self.gamma_r = 2.0  # Reference adaptation gain
        
        # Adaptive parameters
        self.theta_x = np.zeros(3)  # State feedback parameters
        self.theta_r = np.zeros(3)  # Reference parameters
        
        # Reference models
        self.roll_ref = ReferenceModel(2.0, 0.7)
        self.pitch_ref = ReferenceModel(1.5, 0.8)
        self.yaw_ref = ReferenceModel(1.0, 0.9)
        
        logger.info("Initialized MRAC flight controller")
    
    def adapt_parameters(self, error: float, state: FlightState, command: List[float], dt: float):
        """Update adaptive parameters."""
        # State vector
        x = np.array([state.roll, state.pitch, state.yaw])
        
        # Update parameters
        self.theta_x += self.gamma_x * error * x * dt
        self.theta_r += self.gamma_r * error * np.array(command) * dt
        
        # Parameter projection to ensure stability
        self.theta_x = np.clip(self.theta_x, -5.0, 5.0)
        self.theta_r = np.clip(self.theta_r, -5.0, 5.0)
    
    def compute_control(self, state: FlightState, command: List[float], dt: float) -> List[float]:
        """Compute control inputs."""
        # Get reference model responses
        roll_d, roll_rate_d = self.roll_ref.compute_response(command[0], state.roll, state.p, dt)
        pitch_d, pitch_rate_d = self.pitch_ref.compute_response(command[1], state.pitch, state.q, dt)
        yaw_d, yaw_rate_d = self.yaw_ref.compute_response(command[2], state.yaw, state.r, dt)
        
        # Compute tracking errors
        roll_error = state.roll - roll_d
        pitch_error = state.pitch - pitch_d
        yaw_error = state.yaw - yaw_d
        
        # Adapt parameters
        total_error = np.sqrt(roll_error**2 + pitch_error**2 + yaw_error**2)
        self.adapt_parameters(total_error, state, command, dt)
        
        # Compute control inputs
        control = -(self.theta_x * np.array([state.roll, state.pitch, state.yaw]) +
                   self.theta_r * np.array(command))
        
        return control.tolist()


class MRACFlightController:
    """Main flight control system."""
    
    def __init__(self):
        """Initialize flight controller."""
        self.controller = AdaptiveController()
        self.current_state = FlightState()
        self.dt = 0.01  # Control interval (s)
        
    def update(self, state: FlightState, desired_attitude: List[float]) -> List[float]:
        """Update controller and get control inputs."""
        self.current_state = state
        
        # Compute control inputs
        control_inputs = self.controller.compute_control(
            state, desired_attitude, self.dt
        )
        
        logger.debug(f"Control inputs: {control_inputs}")
        return control_inputs
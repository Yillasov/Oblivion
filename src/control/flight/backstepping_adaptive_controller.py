#!/usr/bin/env python3
"""
Backstepping Adaptive Flight Controller

Implementation of a nonlinear adaptive flight control system using backstepping
and neural adaptation for robust trajectory tracking.
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

logger = get_logger("backstepping_adaptive_controller")


@dataclass
class AircraftState:
    """Aircraft state variables."""
    position: np.ndarray    # [x, y, z] position
    velocity: np.ndarray    # [vx, vy, vz] velocity
    attitude: np.ndarray    # [phi, theta, psi] Euler angles
    angular_rate: np.ndarray  # [p, q, r] angular rates
    
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.zeros(3)
        self.angular_rate = np.zeros(3)


class NeuralApproximator:
    """Neural network for uncertainty approximation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize neural approximator."""
        self.W = np.random.randn(hidden_dim, input_dim) * 0.1
        self.V = np.random.randn(output_dim, hidden_dim) * 0.1
        self.gamma_w = 0.1  # W adaptation rate
        self.gamma_v = 0.2  # V adaptation rate
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.hidden = np.tanh(self.W @ x)
        return self.V @ self.hidden
    
    def adapt(self, x: np.ndarray, error: np.ndarray, dt: float):
        """Update weights based on error."""
        hidden_grad = 1.0 - self.hidden**2  # tanh derivative
        
        # Update output weights
        dV = -self.gamma_v * np.outer(error, self.hidden)
        self.V += dV * dt
        
        # Update hidden weights
        dW = -self.gamma_w * np.outer(
            (self.V.T @ error) * hidden_grad, x)
        self.W += dW * dt


class BacksteppingController:
    """Backstepping adaptive controller."""
    
    def __init__(self):
        """Initialize controller."""
        # Control gains
        self.k1 = np.diag([2.0, 2.0, 2.0])  # Position error gain
        self.k2 = np.diag([1.5, 1.5, 1.5])  # Velocity error gain
        self.k3 = np.diag([1.0, 1.0, 1.0])  # Attitude error gain
        self.k4 = np.diag([0.5, 0.5, 0.5])  # Angular rate error gain
        
        # Neural approximators
        self.pos_nn = NeuralApproximator(9, 12, 3)  # Position dynamics
        self.att_nn = NeuralApproximator(12, 15, 3)  # Attitude dynamics
        
        logger.info("Initialized backstepping adaptive controller")
    
    def compute_virtual_control(self, state: AircraftState, 
                              desired_pos: np.ndarray,
                              desired_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute virtual control inputs."""
        # Position error
        pos_error = state.position - desired_pos
        vel_error = state.velocity - desired_vel
        
        # Virtual control (desired acceleration)
        virtual_control = -self.k1 @ pos_error - self.k2 @ vel_error
        
        # Neural compensation
        state_vec = np.concatenate([pos_error, vel_error, state.attitude])
        uncertainty_comp = self.pos_nn.forward(state_vec)
        virtual_control -= uncertainty_comp
        
        return virtual_control, pos_error
    
    def compute_control(self, state: AircraftState, 
                       desired_state: AircraftState,
                       dt: float) -> np.ndarray:
        """Compute control inputs using backstepping."""
        # First layer: position control
        virtual_acc, pos_error = self.compute_virtual_control(
            state, desired_state.position, desired_state.velocity)
        
        # Attitude error
        att_error = state.attitude - desired_state.attitude
        rate_error = state.angular_rate - desired_state.angular_rate
        
        # Neural compensation for attitude dynamics
        state_vec = np.concatenate([att_error, rate_error, 
                                  state.attitude, state.angular_rate])
        uncertainty_comp = self.att_nn.forward(state_vec)
        
        # Control law
        control = (-self.k3 @ att_error - self.k4 @ rate_error - 
                  uncertainty_comp + virtual_acc)
        
        # Adapt neural networks
        self.pos_nn.adapt(state_vec[:9], pos_error, dt)
        self.att_nn.adapt(state_vec, att_error, dt)
        
        return control


class AdaptiveFlightController:
    """Main adaptive flight control system."""
    
    def __init__(self):
        """Initialize flight controller."""
        self.controller = BacksteppingController()
        self.current_state = AircraftState()
        self.desired_state = AircraftState()
        self.dt = 0.01
        
        logger.info("Initialized adaptive flight control system")
    
    def update(self, state: AircraftState, 
               desired_state: AircraftState) -> np.ndarray:
        """Update controller and get control inputs."""
        self.current_state = state
        self.desired_state = desired_state
        
        # Compute control inputs
        control = self.controller.compute_control(
            state, desired_state, self.dt)
        
        logger.debug(f"Control inputs: {control}")
        return control
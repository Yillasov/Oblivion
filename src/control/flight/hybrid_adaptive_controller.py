#!/usr/bin/env python3
"""
Hybrid Adaptive Flight Controller

Implementation of a hybrid adaptive control system combining direct and indirect
adaptation with sliding mode control for robust flight performance.
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
from scipy.integrate import odeint

from src.core.utils.logging_framework import get_logger

logger = get_logger("hybrid_adaptive_controller")


@dataclass
class FlightDynamics:
    """Aircraft dynamics parameters."""
    
    mass: float = 10.0
    inertia: np.ndarray = np.diag([0.8, 0.8, 1.2])
    arm_length: float = 0.2
    thrust_coeff: float = 1e-3
    drag_coeff: float = 2e-4


class ParameterEstimator:
    """Online parameter estimation."""
    
    def __init__(self, num_params: int):
        self.params = np.ones(num_params)
        self.covariance = 100.0 * np.eye(num_params)
        self.forgetting = 0.98
        
    def update(self, error: np.ndarray, regressor: np.ndarray, dt: float):
        """Update parameter estimates using RLS."""
        # Compute gain
        gain = (self.covariance @ regressor) / (
            self.forgetting + regressor.T @ self.covariance @ regressor)
        
        # Update parameters
        self.params += gain * error * dt
        
        # Update covariance
        self.covariance = (self.covariance - 
            np.outer(gain, regressor.T @ self.covariance)) / self.forgetting


class SlidingModeController:
    """Sliding mode control component."""
    
    def __init__(self):
        self.lambda_s = 2.0  # Sliding surface slope
        self.eta = 0.1      # Reaching law gain
        self.phi = 0.05     # Boundary layer thickness
        
    def compute_control(self, error: np.ndarray, error_dot: np.ndarray) -> np.ndarray:
        """Compute sliding mode control law."""
        # Compute sliding surface
        s = error_dot + self.lambda_s * error
        
        # Continuous approximation of sign function
        sat = np.clip(s / self.phi, -1, 1)
        
        return -self.eta * sat


class DirectAdaptiveComponent:
    """Direct adaptive control component."""
    
    def __init__(self, state_dim: int):
        self.weights = np.zeros((state_dim, state_dim))
        self.gamma = 1.0  # Adaptation gain
        
    def adapt(self, error: np.ndarray, state: np.ndarray, dt: float):
        """Update adaptive weights."""
        self.weights += self.gamma * np.outer(error, state) * dt
        
    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """Compute direct adaptive control."""
        return self.weights @ state


class HybridController:
    """Hybrid adaptive flight controller."""
    
    def __init__(self, dynamics: FlightDynamics):
        """Initialize hybrid controller."""
        self.dynamics = dynamics
        self.state_dim = 6  # [attitude, rates]
        
        # Controller components
        self.param_estimator = ParameterEstimator(4)  # Mass and inertia
        self.sliding_mode = SlidingModeController()
        self.direct_adaptive = DirectAdaptiveComponent(self.state_dim)
        
        # Control gains
        self.K_p = np.diag([5.0, 5.0, 5.0])
        self.K_d = np.diag([2.0, 2.0, 2.0])
        
        logger.info("Initialized hybrid adaptive controller")
    
    def compute_control(self, state: np.ndarray, 
                       reference: np.ndarray,
                       dt: float) -> np.ndarray:
        """Compute hybrid control input."""
        # Split state
        attitude = state[:3]
        rates = state[3:]
        
        # Compute errors
        att_error = attitude - reference[:3]
        rate_error = rates - reference[3:]
        
        # Build regressor
        regressor = np.concatenate([attitude, rates])
        
        # Update parameter estimates
        self.param_estimator.update(att_error, regressor, dt)
        
        # Compute control components
        u_baseline = -self.K_p @ att_error - self.K_d @ rate_error
        u_sliding = self.sliding_mode.compute_control(att_error, rate_error)
        u_direct = self.direct_adaptive.compute_control(np.concatenate([att_error, rate_error]))
        
        # Adapt direct component
        self.direct_adaptive.adapt(att_error, state, dt)
        
        # Combine control signals
        control = u_baseline + u_sliding + u_direct
        
        # Apply control allocation
        B = self.compute_control_effectiveness(self.param_estimator.params)
        u_allocated = np.linalg.pinv(B) @ control
        
        return np.clip(u_allocated, -1.0, 1.0)
    
    def compute_control_effectiveness(self, params: np.ndarray) -> np.ndarray:
        """Compute control effectiveness matrix."""
        # Simplified allocation matrix
        B = np.array([
            [params[0], params[0], params[0], params[0]],
            [params[1], -params[1], -params[1], params[1]],
            [params[2], params[2], -params[2], -params[2]]
        ])
        return B


class HybridFlightController:
    """Main hybrid adaptive flight control system."""
    
    def __init__(self):
        """Initialize flight controller."""
        self.dynamics = FlightDynamics()
        self.controller = HybridController(self.dynamics)
        self.dt = 0.01
        
        logger.info("Initialized hybrid adaptive flight control system")
    
    def update(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Update controller and get control inputs."""
        control = self.controller.compute_control(state, reference, self.dt)
        
        logger.debug(f"Control inputs: {control}")
        return control
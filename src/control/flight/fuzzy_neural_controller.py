"""
Fuzzy-Neural Adaptive Flight Controller

Implementation of a flight control system combining fuzzy logic inference
with neural adaptation for robust flight control.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("fuzzy_neural_controller")


@dataclass
class FuzzyRules:
    """Fuzzy rule base structure."""
    
    error_sets = ['negative', 'zero', 'positive']
    rate_sets = ['decreasing', 'stable', 'increasing']
    output_sets = ['decrease_high', 'decrease_low', 'maintain', 
                  'increase_low', 'increase_high']


class FuzzyInferenceSystem:
    """Fuzzy logic controller component."""
    
    def __init__(self):
        self.rules = FuzzyRules()
        self.rule_weights = np.ones((3, 3, 3))  # Initial rule weights
        self.learning_rate = 0.1
    
    def membership(self, x: float, set_type: str) -> np.ndarray:
        """Compute membership values."""
        centers = {'negative': -1.0, 'zero': 0.0, 'positive': 1.0}
        widths = {'negative': 0.5, 'zero': 0.3, 'positive': 0.5}
        
        memberships = []
        for set_name in self.rules.error_sets:
            center = centers[set_name]
            width = widths[set_name]
            membership = np.exp(-(x - center)**2 / (2 * width**2))
            memberships.append(membership)
        
        return np.array(memberships)
    
    def defuzzify(self, rule_activations: np.ndarray) -> float:
        """Defuzzify output using center of gravity."""
        output_centers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        return np.sum(rule_activations * output_centers) / np.sum(rule_activations)


class AdaptiveNetwork(nn.Module):
    """Neural network for adaptive compensation."""
    
    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FuzzyNeuralController:
    """Combined fuzzy-neural adaptive controller."""
    
    def __init__(self):
        """Initialize controller components."""
        self.fuzzy_system = FuzzyInferenceSystem()
        self.adaptive_network = AdaptiveNetwork()
        
        # Control parameters
        self.attitude_gains = np.array([1.0, 1.0, 1.0])
        self.rate_gains = np.array([0.5, 0.5, 0.5])
        
        logger.info("Initialized fuzzy-neural controller")
    
    def compute_fuzzy_control(self, error: np.ndarray, 
                            error_rate: np.ndarray) -> np.ndarray:
        """Compute fuzzy control component."""
        control = np.zeros(3)
        
        for axis in range(3):
            # Compute memberships
            error_memberships = self.fuzzy_system.membership(error[axis], 'error')
            rate_memberships = self.fuzzy_system.membership(error_rate[axis], 'rate')
            
            # Rule evaluation
            rule_activations = np.zeros(5)
            for i, e_mem in enumerate(error_memberships):
                for j, r_mem in enumerate(rate_memberships):
                    rule_strength = e_mem * r_mem * self.fuzzy_system.rule_weights[i,j,axis]
                    rule_activations += rule_strength
            
            # Defuzzification
            control[axis] = self.fuzzy_system.defuzzify(rule_activations)
        
        return control
    
    def update(self, state: np.ndarray, desired_state: np.ndarray, 
               dt: float) -> np.ndarray:
        """Update controller and compute control inputs."""
        # Extract relevant states
        attitude = state[0:3]
        rates = state[3:6]
        
        # Compute errors
        attitude_error = desired_state[0:3] - attitude
        rate_error = desired_state[3:6] - rates
        
        # Fuzzy control computation
        u_fuzzy = self.compute_fuzzy_control(attitude_error, rate_error)
        
        # Neural adaptation
        state_tensor = torch.FloatTensor(np.concatenate([attitude_error, rate_error]))
        with torch.no_grad():
            u_adaptive = self.adaptive_network(state_tensor).numpy()
        
        # Train neural network
        self.adaptive_network.optimizer.zero_grad()
        prediction = self.adaptive_network(state_tensor.requires_grad_())
        loss = torch.mean(prediction * torch.FloatTensor(attitude_error))
        loss.backward()
        self.adaptive_network.optimizer.step()
        
        # Combine control signals
        control = (self.attitude_gains * u_fuzzy + 
                  self.rate_gains * u_adaptive)
        
        # Update fuzzy rules
        error_total = np.sum(np.abs(attitude_error))
        self.fuzzy_system.rule_weights *= (1.0 - self.fuzzy_system.learning_rate * error_total)
        self.fuzzy_system.rule_weights = np.clip(self.fuzzy_system.rule_weights, 0.1, 2.0)
        
        return np.clip(control, -1.0, 1.0)


class FuzzyNeuralFlightSystem:
    """Main flight control system."""
    
    def __init__(self):
        """Initialize flight control system."""
        self.controller = FuzzyNeuralController()
        self.dt = 0.01
        
        logger.info("Initialized fuzzy-neural flight control system")
    
    def step(self, state: np.ndarray, desired_state: np.ndarray) -> np.ndarray:
        """Execute one control step."""
        control = self.controller.update(state, desired_state, self.dt)
        
        logger.debug(f"Control inputs: {control}")
        return control
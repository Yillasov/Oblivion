#!/usr/bin/env python3
"""
Optimization Algorithms for Neuromorphic Training

Provides various optimization algorithms for training spiking neural networks.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("optimization")


class OptimizationAlgorithm(ABC):
    """Base class for optimization algorithms."""
    
    @abstractmethod
    def optimize(self, parameters: Dict[str, Any], gradients: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize parameters based on gradients.
        
        Args:
            parameters: Current parameters
            gradients: Parameter gradients
            
        Returns:
            Dict[str, Any]: Updated parameters
        """
        pass


class SGDOptimizer(OptimizationAlgorithm):
    """Simple Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
        """
        self.learning_rate = learning_rate
    
    def optimize(self, parameters: Dict[str, Any], gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SGD update rule."""
        updated_params = {}
        for key, param in parameters.items():
            if key in gradients:
                updated_params[key] = param - self.learning_rate * gradients[key]
            else:
                updated_params[key] = param
        return updated_params


class AdamOptimizer(OptimizationAlgorithm):
    """Adam optimizer implementation."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep
    
    def optimize(self, parameters: Dict[str, Any], gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Adam update rule."""
        self.t += 1
        updated_params = {}
        
        for key, param in parameters.items():
            if key not in gradients:
                updated_params[key] = param
                continue
                
            # Initialize moments if needed
            if key not in self.m:
                self.m[key] = 0
                self.v[key] = 0
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_params[key] = param - self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)
            
        return updated_params


class OptimizerRegistry:
    """Registry for optimization algorithms."""
    
    _optimizers: Dict[str, Type[OptimizationAlgorithm]] = {}
    
    @classmethod
    def register(cls, name: str, optimizer_class: Type[OptimizationAlgorithm]) -> None:
        """Register an optimizer."""
        cls._optimizers[name.lower()] = optimizer_class
    
    @classmethod
    def get_optimizer(cls, name: str, **kwargs) -> Optional[OptimizationAlgorithm]:
        """Get optimizer by name with optional parameters."""
        optimizer_class = cls._optimizers.get(name.lower())
        if optimizer_class:
            return optimizer_class(**kwargs)
        return None


# Register built-in optimizers
OptimizerRegistry.register("sgd", SGDOptimizer)
OptimizerRegistry.register("adam", AdamOptimizer)
"""
Learning algorithms for optimizing landing procedures.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

from src.landing_gear.base import NeuromorphicLandingGear, TelemetryData
from src.landing_gear.neuromorphic_control import NeuromorphicLandingController


class LearningStrategy(Enum):
    """Learning strategies for landing optimization."""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    ONLINE = "online"


@dataclass
class LandingExperience:
    """Data structure to store landing experiences for learning."""
    telemetry: List[TelemetryData]
    landing_conditions: Dict[str, Any]
    adaptations: Dict[str, float]
    performance_metrics: Dict[str, float]
    timestamp: float


class LandingOptimizer:
    """Simple learning system for optimizing landing procedures."""
    
    def __init__(self, controller: NeuromorphicLandingController, strategy: LearningStrategy = LearningStrategy.REINFORCEMENT):
        """Initialize with a neuromorphic landing controller."""
        self.controller = controller
        self.strategy = strategy
        self.experiences = []
        self.max_experiences = 100
        self.learning_rate = 0.05
        
    def record_landing(self, 
                      landing_conditions: Dict[str, Any], 
                      performance_metrics: Dict[str, float]) -> None:
        """Record a landing experience for learning."""
        experience = LandingExperience(
            telemetry=self.controller.telemetry_buffer.copy(),
            landing_conditions=landing_conditions.copy(),
            adaptations=self.controller.adapt(landing_conditions),
            performance_metrics=performance_metrics.copy(),
            timestamp=np.datetime64('now').astype(float)
        )
        
        self.experiences.append(experience)
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)
            
        # Learn from this experience
        self._learn_from_experience(experience)
    
    def _learn_from_experience(self, experience: LandingExperience) -> None:
        """Learn from a landing experience."""
        if self.strategy == LearningStrategy.REINFORCEMENT:
            self._reinforcement_learning(experience)
        elif self.strategy == LearningStrategy.SUPERVISED:
            self._supervised_learning(experience)
        elif self.strategy == LearningStrategy.ONLINE:
            self._online_learning(experience)
    
    def _reinforcement_learning(self, experience: LandingExperience) -> None:
        """Simple reinforcement learning implementation."""
        # Calculate reward based on performance metrics
        smoothness = experience.performance_metrics.get("landing_smoothness", 0.5)
        efficiency = experience.performance_metrics.get("energy_efficiency", 0.5)
        speed = experience.performance_metrics.get("deployment_speed_score", 0.5)
        
        # Combined reward (higher is better)
        reward = smoothness * 0.5 + efficiency * 0.3 + speed * 0.2
        
        # Update weights based on reward
        for key in self.controller.weights:
            # Adjust weights proportionally to reward
            adjustment = (reward - 0.5) * self.learning_rate
            # Apply adjustment to the weights that contributed most
            self.controller.weights[key][0] += adjustment * 0.6  # Current data
            self.controller.weights[key][1] += adjustment * 0.3  # Historical data
            self.controller.weights[key][2] += adjustment * 0.1  # Predicted data
            
            # Normalize weights
            self.controller.weights[key] /= np.sum(self.controller.weights[key])
    
    def _supervised_learning(self, experience: LandingExperience) -> None:
        """Simple supervised learning implementation."""
        # For simplicity, we'll just use a predefined optimal weight set
        if experience.performance_metrics.get("landing_smoothness", 0) < 0.7:
            # If landing wasn't smooth enough, adjust weights toward known good values
            optimal_weights = {
                "shock_absorption": np.array([0.5, 0.3, 0.2]),
                "deployment_speed": np.array([0.6, 0.3, 0.1]),
                "stability": np.array([0.4, 0.4, 0.2])
            }
            
            for key in self.controller.weights:
                self.controller.weights[key] = (
                    (1 - self.learning_rate) * self.controller.weights[key] + 
                    self.learning_rate * optimal_weights[key]
                )
    
    def _online_learning(self, experience: LandingExperience) -> None:
        """Simple online learning implementation."""
        # Adjust weights based on most recent performance only
        smoothness = experience.performance_metrics.get("landing_smoothness", 0.5)
        
        # If landing was very smooth, slightly increase weight of current data
        if smoothness > 0.8:
            for key in self.controller.weights:
                self.controller.weights[key][0] += self.learning_rate * 0.1
                self.controller.weights[key] /= np.sum(self.controller.weights[key])
        # If landing was rough, increase weight of predicted data
        elif smoothness < 0.4:
            for key in self.controller.weights:
                self.controller.weights[key][2] += self.learning_rate * 0.1
                self.controller.weights[key] /= np.sum(self.controller.weights[key])
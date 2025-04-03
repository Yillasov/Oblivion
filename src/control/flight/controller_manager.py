#!/usr/bin/env python3
"""
Flight Controller Decision Framework

Manages multiple flight controllers and implements intelligent switching
based on flight conditions and performance metrics.
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
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger
from src.control.flight.neural_robust_controller import NeuralRobustController
from src.control.flight.fuzzy_neural_controller import FuzzyNeuralController
from src.control.flight.dragonfly_controller import DragonflightSystem
from src.control.flight.backstepping_adaptive_controller import BacksteppingController
from src.control.flight.mrac_flight_controller import MRACFlightController

logger = get_logger("controller_manager")


class FlightMode(Enum):
    TAKEOFF = "takeoff"
    HOVER = "hover"
    CRUISE = "cruise"
    AGILE = "agile"
    LANDING = "landing"
    RECOVERY = "recovery"


@dataclass
class ControllerPerformance:
    """Performance metrics for controller evaluation."""
    
    tracking_error: float = 0.0
    control_effort: float = 0.0
    stability_margin: float = 0.0
    adaptation_speed: float = 0.0
    
    def compute_score(self) -> float:
        """Compute overall performance score."""
        weights = [0.4, 0.3, 0.2, 0.1]
        metrics = [
            1.0 / (1.0 + self.tracking_error),
            1.0 / (1.0 + self.control_effort),
            self.stability_margin,
            self.adaptation_speed
        ]
        return float(np.dot(weights, metrics))


class ControllerManager:
    """Intelligent controller selection and switching framework."""
    
    def __init__(self):
        """Initialize controller manager."""
        # Initialize available controllers
        self.controllers = {
            "neural_robust": NeuralRobustController(),
            "fuzzy_neural": FuzzyNeuralController(),
            "dragonfly": DragonflightSystem(),
            "backstepping": BacksteppingController(),
            "mrac": MRACFlightController()
        }
        
        # Performance tracking
        self.performance_history: Dict[str, List[ControllerPerformance]] = {
            name: [] for name in self.controllers.keys()
        }
        
        # Current state
        self.active_controller = "neural_robust"
        self.current_mode = FlightMode.HOVER
        
        # Switching parameters
        self.switch_cooldown = 2.0  # Minimum time between switches
        self.last_switch_time = 0.0
        self.switch_threshold = 0.15  # Performance improvement threshold
        
        logger.info("Initialized controller manager")
    
    def evaluate_performance(self, state: np.ndarray, 
                           desired_state: np.ndarray,
                           control: np.ndarray) -> ControllerPerformance:
        """Evaluate controller performance."""
        metrics = ControllerPerformance()
        
        # Tracking error
        metrics.tracking_error = float(np.linalg.norm(desired_state - state))
        
        # Control effort
        metrics.control_effort = float(np.linalg.norm(control))
        
        # Stability margin estimation
        angular_rates = np.linalg.norm(state[3:6])
        metrics.stability_margin = float(1.0 / (1.0 + angular_rates))
        
        # Adaptation speed (if history exists)
        if len(self.performance_history[self.active_controller]) > 0:
            prev_error = self.performance_history[self.active_controller][-1].tracking_error
            metrics.adaptation_speed = float(max(0, (prev_error - metrics.tracking_error) / prev_error))
        
        return metrics
    
    def determine_flight_mode(self, state: np.ndarray, 
                            desired_state: np.ndarray) -> FlightMode:
        """Determine current flight mode."""
        velocity = np.linalg.norm(state[3:6])
        height = state[2]
        angular_rates = np.linalg.norm(state[3:6])
        
        if height < 2.0 and velocity < 1.0:
            return FlightMode.TAKEOFF if height < 0.5 else FlightMode.LANDING
        elif angular_rates > 2.0:
            return FlightMode.AGILE
        elif velocity > 5.0:
            return FlightMode.CRUISE
        elif np.any(np.abs(state - desired_state) > 5.0):
            return FlightMode.RECOVERY
        else:
            return FlightMode.HOVER
    
    def select_controller(self, mode: FlightMode, time_now: float) -> str:
        """Select best controller for current mode."""
        if time_now - self.last_switch_time < self.switch_cooldown:
            return self.active_controller
        
        # Get recent performance scores
        scores = {}
        for name in self.controllers.keys():
            if len(self.performance_history[name]) > 0:
                recent_metrics = self.performance_history[name][-5:]
                scores[name] = np.mean([m.compute_score() for m in recent_metrics])
            else:
                scores[name] = 0.0
        
        # Mode-specific preferences
        mode_weights = {
            FlightMode.TAKEOFF: {"neural_robust": 1.2, "backstepping": 1.1},
            FlightMode.HOVER: {"fuzzy_neural": 1.2, "dragonfly": 1.1},
            FlightMode.CRUISE: {"neural_robust": 1.2, "mrac": 1.1},
            FlightMode.AGILE: {"dragonfly": 1.3, "backstepping": 1.1},
            FlightMode.LANDING: {"fuzzy_neural": 1.2, "neural_robust": 1.1},
            FlightMode.RECOVERY: {"neural_robust": 1.3, "backstepping": 1.2}
        }
        
        # Apply mode-specific weights
        for controller, weight in mode_weights[mode].items():
            scores[controller] *= weight
        
        best_controller = max(scores.items(), key=lambda x: x[1])[0]
        
        # Check if switch is warranted
        if (best_controller != self.active_controller and 
            scores[best_controller] > scores[self.active_controller] + self.switch_threshold):
            self.last_switch_time = time_now
            return best_controller
        
        return self.active_controller
    
    def update(self, state: np.ndarray, desired_state: np.ndarray, 
               time_now: float) -> np.ndarray:
        """Update controller manager and compute control inputs."""
        # Determine flight mode
        new_mode = self.determine_flight_mode(state, desired_state)
        if new_mode != self.current_mode:
            logger.info(f"Flight mode changed: {self.current_mode} -> {new_mode}")
            self.current_mode = new_mode
        
        # Get control from current controller
        control = self.controllers[self.active_controller].update(state, desired_state)
        
        # Evaluate performance
        performance = self.evaluate_performance(state, desired_state, control)
        self.performance_history[self.active_controller].append(performance)
        
        # Consider controller switching
        new_controller = self.select_controller(self.current_mode, time_now)
        if new_controller != self.active_controller:
            logger.info(f"Switching controller: {self.active_controller} -> {new_controller}")
            self.active_controller = new_controller
        
        return control
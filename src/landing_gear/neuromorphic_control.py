"""
Neuromorphic control system for adaptive landing gear behavior.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

from src.landing_gear.base import NeuromorphicLandingGear, TelemetryData


class AdaptiveMode(Enum):
    """Adaptive modes for landing gear control."""
    NORMAL = "normal"
    ROUGH_TERRAIN = "rough_terrain"
    HIGH_SPEED = "high_speed"
    EMERGENCY = "emergency"
    STEALTH = "stealth"


class NeuromorphicLandingController:
    """Simple neuromorphic controller for adaptive landing gear behavior."""
    
    def __init__(self, landing_gear: NeuromorphicLandingGear):
        """Initialize with a landing gear system."""
        self.landing_gear = landing_gear
        self.mode = AdaptiveMode.NORMAL
        self.learning_rate = 0.1
        self.weights = {
            "shock_absorption": np.array([0.6, 0.2, 0.2]),  # [current, historical, predicted]
            "deployment_speed": np.array([0.7, 0.2, 0.1]),
            "stability": np.array([0.5, 0.3, 0.2])
        }
        self.telemetry_buffer = []
        self.max_buffer_size = 50
    
    def process_telemetry(self, telemetry: TelemetryData) -> Dict[str, float]:
        """Process telemetry data and adapt landing gear behavior."""
        # Add to buffer
        self.telemetry_buffer.append(telemetry)
        if len(self.telemetry_buffer) > self.max_buffer_size:
            self.telemetry_buffer.pop(0)
        
        # Extract features
        current_load = telemetry.load
        current_vibration = telemetry.vibration
        
        # Calculate historical averages
        if len(self.telemetry_buffer) > 1:
            historical_load = sum(t.load for t in self.telemetry_buffer[:-1]) / (len(self.telemetry_buffer) - 1)
            historical_vibration = sum(t.vibration for t in self.telemetry_buffer[:-1]) / (len(self.telemetry_buffer) - 1)
        else:
            historical_load = current_load
            historical_vibration = current_vibration
        
        # Simple prediction (linear extrapolation)
        predicted_load = current_load + (current_load - historical_load)
        predicted_vibration = current_vibration + (current_vibration - historical_vibration)
        
        # Apply neuromorphic weights to determine adaptations
        shock_absorption = np.dot(
            self.weights["shock_absorption"], 
            [current_vibration, historical_vibration, predicted_vibration]
        )
        
        deployment_speed = np.dot(
            self.weights["deployment_speed"],
            [current_load, historical_load, predicted_load]
        )
        
        stability = 1.0 - (shock_absorption * 0.5)
        
        # Return adaptations
        return {
            "shock_absorption": min(max(shock_absorption, 0.1), 1.0),
            "deployment_speed": min(max(deployment_speed, 0.1), 1.0),
            "stability": min(max(stability, 0.1), 1.0)
        }
    
    def adapt(self, landing_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt landing gear behavior based on conditions."""
        # Set mode based on conditions
        if landing_conditions.get("emergency", False):
            self.mode = AdaptiveMode.EMERGENCY
        elif landing_conditions.get("terrain_roughness", 0) > 0.7:
            self.mode = AdaptiveMode.ROUGH_TERRAIN
        elif landing_conditions.get("approach_speed", 0) > 0.8:
            self.mode = AdaptiveMode.HIGH_SPEED
        elif landing_conditions.get("stealth_required", False):
            self.mode = AdaptiveMode.STEALTH
        else:
            self.mode = AdaptiveMode.NORMAL
        
        # Get current telemetry
        telemetry = self.landing_gear.get_telemetry()
        
        # Process telemetry for adaptations
        adaptations = self.process_telemetry(telemetry)
        
        # Apply mode-specific adjustments
        if self.mode == AdaptiveMode.ROUGH_TERRAIN:
            adaptations["shock_absorption"] *= 1.5
        elif self.mode == AdaptiveMode.HIGH_SPEED:
            adaptations["deployment_speed"] *= 1.3
        elif self.mode == AdaptiveMode.EMERGENCY:
            adaptations["deployment_speed"] *= 2.0
        elif self.mode == AdaptiveMode.STEALTH:
            # Slower, quieter deployment
            adaptations["deployment_speed"] *= 0.7
        
        # Apply adaptations to landing gear
        self._apply_adaptations(adaptations)
        
        return {
            "mode": self.mode.value,
            "adaptations": adaptations
        }
    
    def _apply_adaptations(self, adaptations: Dict[str, float]) -> None:
        """Apply adaptations to the landing gear."""
        # In a real system, this would adjust physical parameters
        # For this example, we'll just update the landing gear's additional data
        telemetry = self.landing_gear.get_telemetry()
        telemetry.additional_data.update({
            "adaptive_control": adaptations
        })
        
        # Update learning weights based on performance
        if len(self.telemetry_buffer) > 5:
            self._update_weights()
    
    def _update_weights(self) -> None:
        """Update neuromorphic weights based on performance."""
        # Simple weight update based on recent performance
        recent_vibrations = [t.vibration for t in self.telemetry_buffer[-5:]]
        vibration_trend = recent_vibrations[-1] - recent_vibrations[0]
        
        # If vibration is increasing, adjust weights to emphasize prediction
        if vibration_trend > 0:
            self.weights["shock_absorption"] += np.array([-0.05, -0.02, 0.07]) * self.learning_rate
            # Normalize weights
            self.weights["shock_absorption"] /= np.sum(self.weights["shock_absorption"])
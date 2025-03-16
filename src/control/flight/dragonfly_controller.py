"""
Dragonfly-Inspired Adaptive Flight Controller

Implementation of a bio-inspired flight control system based on dragonfly
flight dynamics with multi-wing coordination and sensory adaptation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from src.core.utils.logging_framework import get_logger

logger = get_logger("dragonfly_controller")


class WingState(Enum):
    HOVER = "hover"
    FORWARD = "forward"
    TURN = "turn"
    BRAKE = "brake"


@dataclass
class WingConfiguration:
    """Wing parameters for each wing pair."""
    
    frequency: float = 30.0      # Wing beat frequency (Hz)
    amplitude: float = 60.0      # Stroke amplitude (degrees)
    phase: float = 0.0          # Phase offset
    twist: float = 15.0         # Wing twist angle
    attack_angle: float = 10.0  # Angle of attack


class CompoundEye:
    """Bio-inspired visual processing system."""
    
    def __init__(self, num_ommatidia: int = 100):
        self.num_sensors = num_sensors = num_ommatidia
        self.visual_field = np.zeros(num_sensors)
        self.motion_memory = np.zeros(num_sensors)
        self.adaptation_rate = 0.2
    
    def process_visual_input(self, visual_input: np.ndarray, dt: float) -> np.ndarray:
        """Process visual information like compound eyes."""
        # Motion detection
        motion = np.diff(visual_input, append=visual_input[0])
        
        # Temporal adaptation
        self.motion_memory = ((1 - self.adaptation_rate) * self.motion_memory + 
                            self.adaptation_rate * motion)
        
        return self.motion_memory


class WingController:
    """Adaptive wing control system."""
    
    def __init__(self):
        self.fore_wings = WingConfiguration()
        self.hind_wings = WingConfiguration()
        self.phase_coupling = 0.5
        self.learning_rate = 0.1
    
    def adapt_wing_parameters(self, performance: float, state: WingState):
        """Adapt wing parameters based on flight performance."""
        if state == WingState.HOVER:
            self.fore_wings.frequency = 35.0
            self.fore_wings.phase = np.pi
            self.hind_wings.phase = 0.0
        elif state == WingState.FORWARD:
            self.fore_wings.frequency = 28.0
            self.fore_wings.amplitude = 70.0
            self.phase_coupling = 0.8
        
        # Performance-based adaptation
        if performance < 0.7:
            self.fore_wings.attack_angle += self.learning_rate
            self.hind_wings.attack_angle += self.learning_rate
        
        # Keep parameters within biological limits
        self._constrain_parameters()
    
    def _constrain_parameters(self):
        """Constrain parameters to biological limits."""
        self.fore_wings.attack_angle = np.clip(self.fore_wings.attack_angle, 5, 25)
        self.hind_wings.attack_angle = np.clip(self.hind_wings.attack_angle, 5, 25)
        self.fore_wings.frequency = np.clip(self.fore_wings.frequency, 20, 40)


class DragonflyCPG:
    """Central Pattern Generator for wing coordination."""
    
    def __init__(self):
        self.oscillator_state = np.zeros(4)  # States for 4 wings
        self.coupling_weights = np.array([
            [0, 0.5, 0.3, 0.3],
            [0.5, 0, 0.3, 0.3],
            [0.3, 0.3, 0, 0.5],
            [0.3, 0.3, 0.5, 0]
        ])
    
    def generate_rhythm(self, frequency: float, dt: float) -> np.ndarray:
        """Generate coordinated wing beat patterns."""
        omega = 2 * np.pi * frequency
        
        # Update oscillator states
        for i in range(4):
            coupling = np.sum(self.coupling_weights[i] * np.sin(
                self.oscillator_state - self.oscillator_state[i]))
            self.oscillator_state[i] += (omega + coupling) * dt
        
        return np.sin(self.oscillator_state)


class DragonflyController:
    """Main dragonfly-inspired flight controller."""
    
    def __init__(self):
        """Initialize controller components."""
        self.visual_system = CompoundEye()
        self.wing_controller = WingController()
        self.cpg = DragonflyCPG()
        self.current_state = WingState.HOVER
        
        logger.info("Initialized dragonfly-inspired controller")
    
    def update(self, state: np.ndarray, desired_state: np.ndarray,
               visual_input: np.ndarray, dt: float) -> np.ndarray:
        """Update controller and compute control inputs."""
        # Process visual information
        motion_signals = self.visual_system.process_visual_input(
            visual_input, dt)
        
        # Determine flight state
        velocity = np.linalg.norm(state[3:6])
        if velocity < 0.1:
            self.current_state = WingState.HOVER
        elif velocity > 1.0:
            self.current_state = WingState.FORWARD
        
        # Compute flight performance
        position_error = np.linalg.norm(desired_state[0:3] - state[0:3])
        performance = float(1.0 / (1.0 + position_error))  # Explicitly convert to float
        
        # Adapt wing parameters
        self.wing_controller.adapt_wing_parameters(performance, self.current_state)
        
        # Generate wing beat patterns
        wing_commands = self.cpg.generate_rhythm(
            self.wing_controller.fore_wings.frequency, dt)
        
        # Convert wing commands to control inputs
        control = np.zeros(3)
        control[0] = np.mean(wing_commands[:2])  # Roll
        control[1] = np.mean(wing_commands[2:])  # Pitch
        control[2] = np.sum(wing_commands * [1, -1, 1, -1]) * 0.25  # Yaw
        
        return np.clip(control, -1.0, 1.0)


class DragonflightSystem:
    """Complete dragonfly-inspired flight system."""
    
    def __init__(self):
        """Initialize flight system."""
        self.controller = DragonflyController()
        self.dt = 0.01
        
        logger.info("Initialized dragonfly-inspired flight system")
    
    def step(self, state: np.ndarray, desired_state: np.ndarray,
             visual_input: np.ndarray) -> np.ndarray:
        """Execute one control step."""
        control = self.controller.update(state, desired_state, 
                                       visual_input, self.dt)
        
        logger.debug(f"Control inputs: {control}")
        return control
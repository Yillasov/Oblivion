#!/usr/bin/env python3
"""
Salamander-inspired Central Pattern Generator

Implementation of a dual-mode CPG network for amphibian-like locomotion patterns.
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
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from src.core.utils.logging_framework import get_logger

logger = get_logger("salamander_cpg_controller")


class LocomotionMode(Enum):
    SWIMMING = "swimming"
    WALKING = "walking"
    TRANSITION = "transition"


@dataclass
class SegmentParams:
    """Parameters for body segment oscillator."""
    
    frequency: float = 1.0        # Base frequency (Hz)
    amplitude: float = 1.0        # Oscillation amplitude
    phase_lag: float = 0.25       # Phase lag between segments
    adaptation_speed: float = 0.1  # Speed of mode adaptation


class SegmentOscillator:
    """Body segment oscillator with dual-mode capability."""
    
    def __init__(self, segment_id: int, params: Optional[SegmentParams] = None):
        """Initialize segment oscillator."""
        self.params = params or SegmentParams()
        self.segment_id = segment_id
        
        # State variables
        self.phase = np.random.uniform(0, 2*np.pi)
        self.amplitude = self.params.amplitude
        self.mode_factor = 0.0  # 0: swimming, 1: walking
        
        # Neural activity
        self.left_activity = 0.0
        self.right_activity = 0.0
    
    def update(self, drive: float, neighbors: List[float], dt: float) -> Tuple[float, float]:
        """Update oscillator state."""
        # Update phase
        freq = self.params.frequency * (1.0 + 0.5 * self.mode_factor)
        self.phase += 2*np.pi * freq * dt
        
        # Compute coupling influence
        coupling = sum(np.sin(n - self.phase - self.params.phase_lag) 
                      for n in neighbors)
        
        # Update neural activities
        self.left_activity = self.amplitude * (1 + np.sin(self.phase + coupling))
        self.right_activity = self.amplitude * (1 + np.sin(self.phase + np.pi + coupling))
        
        return self.left_activity, self.right_activity


class LimbController:
    """Controller for limb coordination."""
    
    def __init__(self, num_joints: int):
        """Initialize limb controller."""
        self.oscillators = [SegmentOscillator(i) for i in range(num_joints)]
        self.phase_bias = 2*np.pi / num_joints
        self.coupling_strength = 0.3
    
    def update(self, body_phase: float, dt: float) -> List[float]:
        """Update limb oscillators."""
        joint_angles = []
        for i, osc in enumerate(self.oscillators):
            # Phase coupling with body and other joints
            phase_diff = body_phase - osc.phase + i * self.phase_bias
            coupling = self.coupling_strength * np.sin(phase_diff)
            
            # Update oscillator
            left, right = osc.update(1.0, [body_phase], dt)
            joint_angles.append(left - right)
        
        return joint_angles


class SalamanderCPG:
    """Salamander-inspired CPG network."""
    
    def __init__(self, num_segments: int = 8, num_limbs: int = 4):
        """Initialize CPG network."""
        self.segments = [SegmentOscillator(i) for i in range(num_segments)]
        self.limbs = LimbController(num_limbs)
        
        self.mode = LocomotionMode.WALKING
        self.drive = 1.0
        self.t = 0.0
        self.dt = 0.01
        
        logger.info(f"Created Salamander CPG with {num_segments} segments "
                   f"and {num_limbs} limbs")
    
    def set_mode(self, mode: LocomotionMode, transition_time: float = 1.0):
        """Set locomotion mode with smooth transition."""
        self.mode = mode
        target = 1.0 if mode == LocomotionMode.WALKING else 0.0
        
        for segment in self.segments:
            segment.mode_factor += (target - segment.mode_factor) * transition_time
    
    def step(self, drive: Optional[float] = None) -> Tuple[List[float], List[float]]:
        """Generate next locomotion pattern."""
        self.t += self.dt
        if drive is not None:
            self.drive = drive
        
        # Update body segments
        segment_phases = [seg.phase for seg in self.segments]
        body_angles = []
        
        for i, segment in enumerate(self.segments):
            # Get neighbor phases for coupling
            neighbors = segment_phases[max(0, i-1):min(len(self.segments), i+2)]
            
            # Update segment
            left, right = segment.update(self.drive, neighbors, self.dt)
            body_angles.append(left - right)
        
        # Update limbs based on nearest body segment
        limb_angles = self.limbs.update(self.segments[len(self.segments)//2].phase, 
                                      self.dt)
        
        return body_angles, limb_angles
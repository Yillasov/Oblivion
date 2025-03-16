"""
Adaptive Kuramoto-based Central Pattern Generator

Implementation of coupled phase oscillators with adaptive synchronization.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("adaptive_cpg_controller")


@dataclass
class KuramotoParams:
    """Parameters for Kuramoto oscillator."""
    
    natural_freq: float = 1.0    # Natural frequency (Hz)
    coupling_strength: float = 0.5
    adaptation_rate: float = 0.1
    amplitude: float = 1.0
    phase_bias: float = 0.0


class AdaptiveOscillator:
    """Phase oscillator with frequency adaptation."""
    
    def __init__(self, params: Optional[KuramotoParams] = None):
        """Initialize adaptive oscillator."""
        self.params = params or KuramotoParams()
        self.phase = np.random.uniform(0, 2*np.pi)
        self.freq = self.params.natural_freq
        self.amplitude = self.params.amplitude
        self.learning_history = []
    
    def update(self, coupling_term: float, dt: float) -> Tuple[float, float]:
        """Update oscillator state."""
        # Update phase
        self.phase += 2*np.pi * self.freq * dt + coupling_term
        self.phase = self.phase % (2*np.pi)
        
        # Adapt frequency
        freq_adaptation = self.params.adaptation_rate * np.sin(coupling_term) * dt
        self.freq += freq_adaptation
        self.learning_history.append(self.freq)
        
        # Generate outputs
        x = self.amplitude * np.cos(self.phase)
        v = -self.amplitude * self.freq * np.sin(self.phase)
        
        return x, v


class RhythmicPattern:
    """Rhythmic pattern generator using coupled oscillators."""
    
    def __init__(self, num_oscillators: int):
        """Initialize pattern generator."""
        self.oscillators = []
        self.coupling_matrix = np.zeros((num_oscillators, num_oscillators))
        
        # Create oscillators with different natural frequencies
        for i in range(num_oscillators):
            params = KuramotoParams(
                natural_freq=1.0 + 0.1*i,
                phase_bias=2*np.pi*i/num_oscillators
            )
            self.oscillators.append(AdaptiveOscillator(params))
        
        self._setup_coupling()
    
    def _setup_coupling(self):
        """Setup coupling topology."""
        n = len(self.oscillators)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Bidirectional ring coupling
                    if abs(i-j) == 1 or abs(i-j) == n-1:
                        self.coupling_matrix[i,j] = 0.5


class AdaptiveCPGController:
    """Adaptive CPG controller with phase synchronization."""
    
    def __init__(self, num_joints: int):
        """Initialize controller."""
        self.pattern = RhythmicPattern(num_joints)
        self.t = 0.0
        self.dt = 0.01
        self.phase_history = []
        
        logger.info(f"Created adaptive CPG controller with {num_joints} oscillators")
    
    def compute_phase_coherence(self) -> float:
        """Compute phase coherence of the network."""
        phases = [osc.phase for osc in self.pattern.oscillators]
        r = np.abs(np.mean(np.exp(1j * np.array(phases))))
        return r
    
    def step(self, sensory_feedback: Optional[List[float]] = None) -> List[float]:
        """Generate next pattern step."""
        self.t += self.dt
        
        # Update oscillators
        positions = []
        velocities = []
        n = len(self.pattern.oscillators)
        
        for i, oscillator in enumerate(self.pattern.oscillators):
            # Calculate coupling term
            coupling = 0.0
            for j, other in enumerate(self.pattern.oscillators):
                if i != j:
                    weight = self.pattern.coupling_matrix[i,j]
                    phase_diff = other.phase - oscillator.phase
                    coupling += weight * np.sin(phase_diff)
            
            # Add sensory feedback if provided
            if sensory_feedback:
                coupling += 0.1 * sensory_feedback[i]
            
            # Update oscillator
            pos, vel = oscillator.update(coupling, self.dt)
            positions.append(pos)
            velocities.append(vel)
        
        # Store phase coherence
        self.phase_history.append(self.compute_phase_coherence())
        
        return positions
    
    def get_phase_coherence_history(self) -> List[float]:
        """Return phase coherence history."""
        return self.phase_history
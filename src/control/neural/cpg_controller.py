"""
Bio-inspired Central Pattern Generator

Implementation of neural oscillators for generating rhythmic motor patterns.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("cpg_controller")


@dataclass
class OscillatorParams:
    """Parameters for neural oscillator."""
    
    tau: float = 10.0         # Time constant
    beta: float = 2.5         # Coupling strength
    mu: float = 1.0          # Amplitude control
    omega: float = 2.0       # Intrinsic frequency (rad/s)
    phi: float = 0.0         # Phase offset


class MatsuokaNeuron:
    """Implementation of Matsuoka oscillator neuron."""
    
    def __init__(self):
        """Initialize Matsuoka neuron."""
        self.x = 0.0          # Membrane potential
        self.v = 0.0          # Self-inhibition
        self.y = 0.0          # Output
        self.input_sum = 0.0  # Sum of inputs
    
    def update(self, params: OscillatorParams, dt: float):
        """Update neuron state."""
        # Update membrane potential
        dx = (-self.x - self.v + self.input_sum) / params.tau
        self.x += dx * dt
        
        # Update self-inhibition
        dv = (-self.v + self.y) / params.tau
        self.v += dv * dt
        
        # Update output (half-wave rectification)
        self.y = max(0.0, self.x)


class CPGOscillator:
    """Coupled oscillator for pattern generation."""
    
    def __init__(self, params: Optional[OscillatorParams] = None):
        """Initialize oscillator pair."""
        self.params = params or OscillatorParams()
        self.neuron1 = MatsuokaNeuron()
        self.neuron2 = MatsuokaNeuron()
        
        # Mutual inhibition weights
        self.w12 = self.params.beta
        self.w21 = self.params.beta
    
    def update(self, dt: float) -> Tuple[float, float]:
        """Update oscillator state."""
        # Set mutual inhibition
        self.neuron1.input_sum = -self.w12 * self.neuron2.y
        self.neuron2.input_sum = -self.w21 * self.neuron1.y
        
        # Update neurons
        self.neuron1.update(self.params, dt)
        self.neuron2.update(self.params, dt)
        
        # Generate output
        out1 = self.params.mu * self.neuron1.y
        out2 = self.params.mu * self.neuron2.y
        
        return out1, out2


class CPGNetwork:
    """Network of coupled oscillators."""
    
    def __init__(self, num_oscillators: int):
        """Initialize CPG network."""
        self.oscillators = []
        self.coupling_weights = np.zeros((num_oscillators, num_oscillators))
        self.phase_biases = np.zeros((num_oscillators, num_oscillators))
        
        # Create oscillators with different phases
        for i in range(num_oscillators):
            params = OscillatorParams(
                phi=2 * np.pi * i / num_oscillators
            )
            self.oscillators.append(CPGOscillator(params))
        
        self._initialize_coupling()
    
    def _initialize_coupling(self):
        """Initialize coupling weights and phase biases."""
        n = len(self.oscillators)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Nearest neighbor coupling
                    if abs(i - j) == 1 or abs(i - j) == n-1:
                        self.coupling_weights[i,j] = 0.5
                        # Phase difference between adjacent oscillators
                        self.phase_biases[i,j] = 2 * np.pi / n


class CPGController:
    """Central Pattern Generator controller."""
    
    def __init__(self, num_joints: int):
        """Initialize CPG controller."""
        self.network = CPGNetwork(num_joints)
        self.t = 0.0
        self.dt = 0.01  # Time step (s)
        
        logger.info(f"Created CPG controller with {num_joints} oscillators")
    
    def step(self, amplitude_scale: float = 1.0, 
             frequency_scale: float = 1.0) -> List[float]:
        """Generate next pattern step."""
        self.t += self.dt
        
        # Update oscillator parameters based on control inputs
        for osc in self.network.oscillators:
            osc.params.mu = amplitude_scale
            osc.params.omega *= frequency_scale
        
        # Update network state
        outputs = []
        for i, oscillator in enumerate(self.network.oscillators):
            # Calculate coupling inputs
            coupling_input = 0.0
            for j, other_osc in enumerate(self.network.oscillators):
                if i != j:
                    weight = self.network.coupling_weights[i,j]
                    phase_bias = self.network.phase_biases[i,j]
                    coupling_input += weight * np.sin(
                        other_osc.params.phi - oscillator.params.phi - phase_bias
                    )
            
            # Update oscillator
            out1, out2 = oscillator.update(self.dt)
            
            # Use difference between neurons as output
            outputs.append(out1 - out2)
        
        return outputs
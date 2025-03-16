"""
Izhikevich Spiking Neural Network Controller

A biologically-inspired SNN implementation using Izhikevich neuron model.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("izh_snn_controller")


@dataclass
class IzhikevichParams:
    """Parameters for Izhikevich neuron model."""
    
    a: float  # Recovery rate
    b: float  # Sensitivity of recovery variable
    c: float  # After-spike reset value of membrane potential
    d: float  # After-spike reset of recovery variable
    
    @classmethod
    def regular_spiking(cls) -> 'IzhikevichParams':
        """Regular spiking neuron parameters."""
        return cls(a=0.02, b=0.2, c=-65.0, d=8.0)
    
    @classmethod
    def fast_spiking(cls) -> 'IzhikevichParams':
        """Fast spiking neuron parameters."""
        return cls(a=0.1, b=0.2, c=-65.0, d=2.0)
    
    @classmethod
    def chattering(cls) -> 'IzhikevichParams':
        """Chattering neuron parameters."""
        return cls(a=0.02, b=0.2, c=-50.0, d=2.0)


class IzhikevichNeuron:
    """Izhikevich neuron implementation."""
    
    def __init__(self, params: Optional[IzhikevichParams] = None):
        """Initialize Izhikevich neuron."""
        self.params = params or IzhikevichParams.regular_spiking()
        self.v = -65.0  # Membrane potential
        self.u = self.params.b * self.v  # Recovery variable
        self.spike_history = []
    
    def update(self, I: float, dt: float) -> bool:
        """
        Update neuron state.
        
        Args:
            I: Input current
            dt: Time step
            
        Returns:
            bool: True if neuron spiked
        """
        # Update membrane potential and recovery variable
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (self.params.a * (self.params.b * self.v - self.u)) * dt
        
        self.v += dv
        self.u += du
        
        # Check for spike
        if self.v >= 30.0:
            self.v = self.params.c
            self.u += self.params.d
            return True
            
        return False


class AdaptiveSynapse:
    """Adaptive synapse with short-term plasticity."""
    
    def __init__(self, weight: float = 1.0, tau_d: float = 200.0, tau_f: float = 600.0):
        """
        Initialize adaptive synapse.
        
        Args:
            weight: Initial synaptic weight
            tau_d: Depression time constant
            tau_f: Facilitation time constant
        """
        self.base_weight = weight
        self.tau_d = tau_d
        self.tau_f = tau_f
        
        self.D = 1.0  # Depression variable
        self.F = 1.0  # Facilitation variable
        self.last_spike_time = 0.0
    
    def get_weight(self, t: float) -> float:
        """Get current effective weight."""
        dt = t - self.last_spike_time
        
        # Update depression and facilitation
        self.D = 1.0 - (1.0 - self.D) * np.exp(-dt / self.tau_d)
        self.F = 1.0 + (self.F - 1.0) * np.exp(-dt / self.tau_f)
        
        return self.base_weight * self.D * self.F
    
    def update(self, t: float):
        """Update synapse state after spike."""
        self.D *= 0.9  # Depression effect
        self.F *= 1.1  # Facilitation effect
        self.last_spike_time = t


class IzhikevichSNNController:
    """SNN controller using Izhikevich neurons."""
    
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        """Initialize controller."""
        # Create neuron layers with different behaviors
        self.input_layer = [IzhikevichNeuron(IzhikevichParams.fast_spiking()) 
                           for _ in range(num_inputs)]
        self.hidden_layer = [IzhikevichNeuron(IzhikevichParams.regular_spiking()) 
                           for _ in range(num_hidden)]
        self.output_layer = [IzhikevichNeuron(IzhikevichParams.chattering()) 
                           for _ in range(num_outputs)]
        
        # Initialize synaptic connections with adaptive synapses
        self.ih_synapses = [[AdaptiveSynapse(weight=np.random.randn() * 0.1) 
                            for _ in range(num_hidden)] for _ in range(num_inputs)]
        self.ho_synapses = [[AdaptiveSynapse(weight=np.random.randn() * 0.1) 
                            for _ in range(num_outputs)] for _ in range(num_hidden)]
        
        self.t = 0.0
        self.dt = 0.5  # ms
        
        logger.info(f"Created Izhikevich SNN controller with {num_inputs} inputs, "
                   f"{num_hidden} hidden, and {num_outputs} outputs")
    
    def step(self, inputs: List[float]) -> List[float]:
        """Process one time step."""
        self.t += self.dt
        
        # Process input layer
        input_spikes = []
        for i, (neuron, input_val) in enumerate(zip(self.input_layer, inputs)):
            if neuron.update(input_val * 20.0, self.dt):  # Scale input
                input_spikes.append(i)
        
        # Process hidden layer
        hidden_currents = [0.0] * len(self.hidden_layer)
        for i in input_spikes:
            for j, synapse in enumerate(self.ih_synapses[i]):
                hidden_currents[j] += synapse.get_weight(self.t)
                synapse.update(self.t)
        
        hidden_spikes = []
        for i, neuron in enumerate(self.hidden_layer):
            if neuron.update(hidden_currents[i], self.dt):
                hidden_spikes.append(i)
        
        # Process output layer
        output_currents = [0.0] * len(self.output_layer)
        for i in hidden_spikes:
            for j, synapse in enumerate(self.ho_synapses[i]):
                output_currents[j] += synapse.get_weight(self.t)
                synapse.update(self.t)
        
        # Update output neurons
        outputs = []
        for neuron, current in zip(self.output_layer, output_currents):
            spiked = neuron.update(current, self.dt)
            outputs.append(float(spiked))
        
        return outputs
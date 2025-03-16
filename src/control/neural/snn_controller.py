"""
Spiking Neural Network Controller

A simplified implementation of a spiking neural network for control systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("snn_controller")


@dataclass
class LIFNeuronParams:
    """Parameters for Leaky Integrate-and-Fire neuron."""
    
    tau_m: float = 10.0        # Membrane time constant (ms)
    v_rest: float = -70.0      # Resting potential (mV)
    v_thresh: float = -55.0    # Threshold potential (mV)
    v_reset: float = -75.0     # Reset potential (mV)
    r_membrane: float = 10.0   # Membrane resistance (MΩ)


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron model."""
    
    def __init__(self, params: Optional[LIFNeuronParams] = None):
        """Initialize LIF neuron."""
        self.params = params or LIFNeuronParams()
        self.v_membrane = self.params.v_rest
        self.last_spike_time = 0.0
        self.refractory_period = 2.0  # ms
        
    def update(self, current: float, dt: float, t: float) -> bool:
        """
        Update neuron state.
        
        Args:
            current: Input current (nA)
            dt: Time step (ms)
            t: Current time (ms)
            
        Returns:
            bool: True if neuron spiked
        """
        # Check refractory period
        if t - self.last_spike_time < self.refractory_period:
            return False
        
        # Update membrane potential
        dv = (-(self.v_membrane - self.params.v_rest) + 
              self.params.r_membrane * current) / self.params.tau_m
        
        self.v_membrane += dv * dt
        
        # Check for spike
        if self.v_membrane >= self.params.v_thresh:
            self.v_membrane = self.params.v_reset
            self.last_spike_time = t
            return True
            
        return False


class Synapse:
    """Basic synapse model with weight and delay."""
    
    def __init__(self, weight: float = 1.0, delay: float = 1.0):
        """
        Initialize synapse.
        
        Args:
            weight: Synaptic weight
            delay: Synaptic delay (ms)
        """
        self.weight = weight
        self.delay = delay
        self.spike_queue: List[Tuple[float, float]] = []  # (time, current)
    
    def propagate_spike(self, t: float) -> float:
        """
        Propagate spikes and return current.
        
        Args:
            t: Current time (ms)
            
        Returns:
            float: Output current
        """
        # Remove old spikes
        while self.spike_queue and self.spike_queue[0][0] <= t:
            _, current = self.spike_queue.pop(0)
            return current
            
        return 0.0
    
    def add_spike(self, t: float):
        """Add spike to queue."""
        arrival_time = t + self.delay
        self.spike_queue.append((arrival_time, self.weight))


class SNNController:
    """Spiking Neural Network controller."""
    
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        """
        Initialize SNN controller.
        
        Args:
            num_inputs: Number of input neurons
            num_hidden: Number of hidden neurons
            num_outputs: Number of output neurons
        """
        # Create neuron layers
        self.input_layer = [LIFNeuron() for _ in range(num_inputs)]
        self.hidden_layer = [LIFNeuron() for _ in range(num_hidden)]
        self.output_layer = [LIFNeuron() for _ in range(num_outputs)]
        
        # Create synaptic connections
        self.input_hidden_synapses = np.random.randn(num_inputs, num_hidden)
        self.hidden_output_synapses = np.random.randn(num_hidden, num_outputs)
        
        # Create synapse objects
        self.ih_synapses = [[Synapse(weight=w) for w in row] 
                           for row in self.input_hidden_synapses]
        self.ho_synapses = [[Synapse(weight=w) for w in row] 
                           for row in self.hidden_output_synapses]
        
        self.t = 0.0  # Current time
        self.dt = 0.1  # Time step (ms)
        
        logger.info(f"Created SNN controller with {num_inputs} inputs, "
                   f"{num_hidden} hidden, and {num_outputs} outputs")
    
    def step(self, inputs: List[float]) -> List[float]:
        """
        Process one time step.
        
        Args:
            inputs: Input values
            
        Returns:
            List[float]: Output values
        """
        self.t += self.dt
        
        # Process input layer
        input_spikes = []
        for i, (neuron, input_val) in enumerate(zip(self.input_layer, inputs)):
            if neuron.update(input_val, self.dt, self.t):
                input_spikes.append(i)
        
        # Process hidden layer
        hidden_currents = [0.0] * len(self.hidden_layer)
        for i in input_spikes:
            for j, synapse in enumerate(self.ih_synapses[i]):
                synapse.add_spike(self.t)
                hidden_currents[j] += synapse.propagate_spike(self.t)
        
        hidden_spikes = []
        for i, neuron in enumerate(self.hidden_layer):
            if neuron.update(hidden_currents[i], self.dt, self.t):
                hidden_spikes.append(i)
        
        # Process output layer
        output_currents = [0.0] * len(self.output_layer)
        for i in hidden_spikes:
            for j, synapse in enumerate(self.ho_synapses[i]):
                synapse.add_spike(self.t)
                output_currents[j] += synapse.propagate_spike(self.t)
        
        # Update output neurons and collect results
        outputs = []
        for neuron, current in zip(self.output_layer, output_currents):
            spiked = neuron.update(current, self.dt, self.t)
            outputs.append(float(spiked))
        
        return outputs
#!/usr/bin/env python3
"""
Temporal Neural Network Controller

A temporal-based SNN implementation for control systems with feedback mechanisms.
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
from typing import List, Tuple, Dict, Optional, Deque
from collections import deque
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("tnn_snn_controller")


@dataclass
class TNNeuronParams:
    """Parameters for temporal neuron."""
    
    tau_m: float = 15.0       # Membrane time constant (ms)
    tau_f: float = 5.0        # Forward time window (ms)
    tau_b: float = 10.0       # Backward time window (ms)
    v_thresh: float = -50.0   # Threshold potential (mV)
    v_reset: float = -65.0    # Reset potential (mV)
    t_refract: float = 2.5    # Refractory period (ms)


class TemporalNeuron:
    """Temporal neuron with forward and backward integration."""
    
    def __init__(self, params: Optional[TNNeuronParams] = None):
        """Initialize temporal neuron."""
        self.params = params or TNNeuronParams()
        self.v = self.params.v_reset
        self.last_spike = float('-inf')
        self.forward_trace = 0.0
        self.backward_trace = 0.0
        self.spike_history: Deque[float] = deque(maxlen=100)
    
    def update(self, t: float, I_forward: float, I_backward: float, dt: float) -> bool:
        """Update neuron state with temporal integration."""
        if t - self.last_spike < self.params.t_refract:
            return False
        
        # Update temporal traces
        self.forward_trace *= np.exp(-dt / self.params.tau_f)
        self.backward_trace *= np.exp(-dt / self.params.tau_b)
        
        # Integrate inputs
        self.forward_trace += I_forward * dt
        self.backward_trace += I_backward * dt
        
        # Update membrane potential
        self.v += (-self.v + self.forward_trace - self.backward_trace) * dt / self.params.tau_m
        
        # Check for spike
        if self.v >= self.params.v_thresh:
            self.v = self.params.v_reset
            self.last_spike = t
            self.spike_history.append(t)
            return True
        
        return False


class TemporalSynapse:
    """Synapse with temporal dynamics."""
    
    def __init__(self, weight: float = 1.0, delay: float = 1.0):
        """Initialize temporal synapse."""
        self.weight = weight
        self.delay = delay
        self.trace = 0.0
        self.last_update = 0.0
        self.temporal_window: List[Tuple[float, float]] = []
    
    def update(self, t: float, pre_spike: bool, post_spike: bool, dt: float) -> float:
        """Update synapse state and return current."""
        # Update temporal window
        self.temporal_window = [(time, val) for time, val in self.temporal_window 
                              if t - time < 20.0]  # 20ms window
        
        if pre_spike:
            self.temporal_window.append((t + self.delay, self.weight))
        
        # Calculate current contribution
        current = sum(val for time, val in self.temporal_window if time <= t)
        
        # Update trace
        self.trace *= np.exp(-(t - self.last_update) / 10.0)
        if post_spike:
            self.trace += 0.1
        
        self.last_update = t
        return current


class TNNController:
    """Temporal Neural Network controller."""
    
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        """Initialize TNN controller."""
        # Create neuron layers
        self.input_layer = [TemporalNeuron() for _ in range(num_inputs)]
        self.hidden_layer = [TemporalNeuron() for _ in range(num_hidden)]
        self.output_layer = [TemporalNeuron() for _ in range(num_outputs)]
        
        # Create synaptic connections
        self.ih_synapses = [[TemporalSynapse(weight=np.random.randn() * 0.1)
                            for _ in range(num_hidden)]
                           for _ in range(num_inputs)]
        
        self.hh_synapses = [[TemporalSynapse(weight=np.random.randn() * 0.1)
                            for _ in range(num_hidden)]
                           for _ in range(num_hidden)]
        
        self.ho_synapses = [[TemporalSynapse(weight=np.random.randn() * 0.1)
                            for _ in range(num_outputs)]
                           for _ in range(num_hidden)]
        
        self.t = 0.0
        self.dt = 0.1  # Time step (ms)
        
        logger.info(f"Created TNN controller with {num_inputs} inputs, "
                   f"{num_hidden} hidden, and {num_outputs} outputs")
    
    def step(self, inputs: List[float], feedback: List[float]) -> List[float]:
        """Process one time step with feedback."""
        self.t += self.dt
        
        # Process input layer with feedback
        input_spikes = []
        for i, (neuron, input_val) in enumerate(zip(self.input_layer, inputs)):
            I_forward = input_val
            I_backward = feedback[i % len(feedback)] if feedback else 0.0
            
            if neuron.update(self.t, I_forward, I_backward, self.dt):
                input_spikes.append(i)
        
        # Process hidden layer
        hidden_spikes = []
        for i, neuron in enumerate(self.hidden_layer):
            # Forward current from inputs
            I_forward = sum(self.ih_synapses[j][i].update(self.t, j in input_spikes, 
                          False, self.dt) for j in range(len(self.input_layer)))
            
            # Recurrent current from hidden layer
            I_recurrent = sum(self.hh_synapses[j][i].update(self.t, j in hidden_spikes,
                            False, self.dt) for j in range(len(self.hidden_layer)))
            
            if neuron.update(self.t, I_forward + I_recurrent, 0.0, self.dt):
                hidden_spikes.append(i)
        
        # Process output layer
        outputs = []
        for i, neuron in enumerate(self.output_layer):
            I_forward = sum(self.ho_synapses[j][i].update(self.t, j in hidden_spikes,
                          False, self.dt) for j in range(len(self.hidden_layer)))
            
            spiked = neuron.update(self.t, I_forward, 0.0, self.dt)
            outputs.append(float(spiked))
        
        return outputs
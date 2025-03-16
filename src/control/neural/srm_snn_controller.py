"""
Spike Response Model (SRM) Neural Network Controller

An efficient SNN implementation using the Spike Response Model.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("srm_snn_controller")


@dataclass
class SRMNeuronParams:
    """Parameters for SRM neuron."""
    
    tau_m: float = 10.0        # Membrane time constant (ms)
    tau_s: float = 2.0         # Synaptic time constant (ms)
    tau_r: float = 2.0         # Rise time constant (ms)
    eta_reset: float = -5.0    # Reset kernel amplitude
    v_thresh: float = -50.0    # Firing threshold (mV)
    t_refract: float = 3.0     # Refractory period (ms)


class SRMNeuron:
    """Spike Response Model neuron implementation."""
    
    def __init__(self, params: Optional[SRMNeuronParams] = None):
        """Initialize SRM neuron."""
        self.params = params or SRMNeuronParams()
        self.v = 0.0
        self.last_spike = float('-inf')
        self.spike_times: Deque[float] = deque(maxlen=100)
        
    def eta(self, t_diff: float) -> float:
        """Compute eta (spike after-potential)."""
        if t_diff <= 0:
            return 0.0
        return self.params.eta_reset * np.exp(-t_diff / self.params.tau_m)
    
    def epsilon(self, t_diff: float) -> float:
        """Compute epsilon (post-synaptic potential)."""
        if t_diff <= 0:
            return 0.0
        return (np.exp(-t_diff / self.params.tau_s) - 
                np.exp(-t_diff / self.params.tau_r))
    
    def update(self, t: float, I_syn: float) -> bool:
        """Update neuron state."""
        # Check refractory period
        if t - self.last_spike < self.params.t_refract:
            return False
        
        # Compute membrane potential
        self.v = sum(self.eta(t - t_s) for t_s in self.spike_times)
        self.v += I_syn
        
        # Check for spike
        if self.v >= self.params.v_thresh:
            self.last_spike = t
            self.spike_times.append(t)
            return True
        
        return False


class AdaptiveSRMSynapse:
    """Adaptive synapse for SRM neurons."""
    
    def __init__(self, weight: float = 1.0, tau_adapt: float = 100.0):
        """Initialize synapse."""
        self.base_weight = weight
        self.tau_adapt = tau_adapt
        self.adaptation = 1.0
        self.last_spike = 0.0
        self.spike_times: Deque[float] = deque(maxlen=100)
    
    def update(self, t: float, spike: bool = False) -> float:
        """Update synapse state and return effective weight."""
        if spike:
            self.spike_times.append(t)
            self.adaptation *= 0.9  # Short-term depression
        
        # Recovery
        dt = t - self.last_spike
        self.adaptation += (1.0 - self.adaptation) * (1 - np.exp(-dt / self.tau_adapt))
        
        if spike:
            self.last_spike = t
        
        return self.base_weight * self.adaptation


class SRMSNNController:
    """SNN controller using SRM neurons."""
    
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        """Initialize controller."""
        # Create neuron layers
        self.input_layer = [SRMNeuron() for _ in range(num_inputs)]
        self.hidden_layer = [SRMNeuron() for _ in range(num_hidden)]
        self.output_layer = [SRMNeuron() for _ in range(num_outputs)]
        
        # Create synaptic connections
        self.ih_synapses = [[AdaptiveSRMSynapse(weight=np.random.randn() * 0.1)
                            for _ in range(num_hidden)]
                           for _ in range(num_inputs)]
        self.ho_synapses = [[AdaptiveSRMSynapse(weight=np.random.randn() * 0.1)
                            for _ in range(num_outputs)]
                           for _ in range(num_hidden)]
        
        self.t = 0.0
        self.dt = 0.1  # Time step (ms)
        
        logger.info(f"Created SRM-based SNN controller with {num_inputs} inputs, "
                   f"{num_hidden} hidden, and {num_outputs} outputs")
    
    def step(self, inputs: List[float]) -> List[float]:
        """Process one time step."""
        self.t += self.dt
        
        # Process input layer
        input_spikes = []
        for i, (neuron, input_val) in enumerate(zip(self.input_layer, inputs)):
            if neuron.update(self.t, input_val):
                input_spikes.append(i)
        
        # Process hidden layer
        hidden_spikes = []
        for i, neuron in enumerate(self.hidden_layer):
            # Calculate synaptic input
            I_syn = 0.0
            for j, spike in enumerate(input_spikes):
                weight = self.ih_synapses[j][i].update(self.t, True)
                I_syn += weight * neuron.epsilon(self.t - self.t)
            
            if neuron.update(self.t, I_syn):
                hidden_spikes.append(i)
        
        # Process output layer
        outputs = []
        for i, neuron in enumerate(self.output_layer):
            # Calculate synaptic input
            I_syn = 0.0
            for j, spike in enumerate(hidden_spikes):
                weight = self.ho_synapses[j][i].update(self.t, True)
                I_syn += weight * neuron.epsilon(self.t - self.t)
            
            spiked = neuron.update(self.t, I_syn)
            outputs.append(float(spiked))
        
        return outputs
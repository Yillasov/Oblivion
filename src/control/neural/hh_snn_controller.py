"""
Hodgkin-Huxley Spiking Neural Network Controller

A detailed biophysical SNN implementation using Hodgkin-Huxley neuron model.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("hh_snn_controller")


@dataclass
class HHChannelParams:
    """Ion channel parameters for Hodgkin-Huxley model."""
    
    g_max: float  # Maximum conductance (mS/cm²)
    E_rev: float  # Reversal potential (mV)
    
    # Channel kinetics parameters
    alpha_m: Tuple[float, float, float]  # (A, B, C) for m-gate
    beta_m: Tuple[float, float, float]   # (A, B, C) for m-gate
    alpha_h: Tuple[float, float, float]  # (A, B, C) for h-gate
    beta_h: Tuple[float, float, float]   # (A, B, C) for h-gate


class HHNeuron:
    """Hodgkin-Huxley neuron model."""
    
    def __init__(self):
        """Initialize HH neuron."""
        # Membrane parameters
        self.C_m = 1.0  # Membrane capacitance (µF/cm²)
        self.E_leak = -65.0  # Leak reversal potential (mV)
        self.g_leak = 0.3  # Leak conductance (mS/cm²)
        
        # State variables
        self.V = -65.0  # Membrane potential (mV)
        self.m = 0.05  # Na+ activation
        self.h = 0.6   # Na+ inactivation
        self.n = 0.32  # K+ activation
        
        # Channel parameters
        self.Na_channel = HHChannelParams(
            g_max=120.0,
            E_rev=50.0,
            alpha_m=(0.1, 25.0, 10.0),
            beta_m=(4.0, 0.0, 18.0),
            alpha_h=(0.07, 0.0, 20.0),
            beta_h=(1.0, 30.0, 10.0)
        )
        
        self.K_channel = HHChannelParams(
            g_max=36.0,
            E_rev=-77.0,
            alpha_m=(0.01, 10.0, 10.0),
            beta_m=(0.125, 0.0, 80.0),
            alpha_h=(0.0, 0.0, 0.0),
            beta_h=(0.0, 0.0, 0.0)
        )
    
    def _alpha_beta(self, V: float, params: Tuple[float, float, float]) -> float:
        """Calculate alpha or beta rate."""
        A, B, C = params
        return A * (V + B) / (1 - np.exp(-(V + B) / C))
    
    def update(self, I_ext: float, dt: float) -> bool:
        """Update neuron state."""
        # Channel dynamics
        alpha_m = self._alpha_beta(self.V, self.Na_channel.alpha_m)
        beta_m = self._alpha_beta(self.V, self.Na_channel.beta_m)
        alpha_h = self._alpha_beta(self.V, self.Na_channel.alpha_h)
        beta_h = self._alpha_beta(self.V, self.Na_channel.beta_h)
        alpha_n = self._alpha_beta(self.V, self.K_channel.alpha_m)
        beta_n = self._alpha_beta(self.V, self.K_channel.beta_m)
        
        # Update gate variables
        self.m += dt * (alpha_m * (1 - self.m) - beta_m * self.m)
        self.h += dt * (alpha_h * (1 - self.h) - beta_h * self.h)
        self.n += dt * (alpha_n * (1 - self.n) - beta_n * self.n)
        
        # Calculate ionic currents
        I_Na = self.Na_channel.g_max * self.m**3 * self.h * (self.V - self.Na_channel.E_rev)
        I_K = self.K_channel.g_max * self.n**4 * (self.V - self.K_channel.E_rev)
        I_leak = self.g_leak * (self.V - self.E_leak)
        
        # Update membrane potential
        dV = (-I_Na - I_K - I_leak + I_ext) / self.C_m
        self.V += dV * dt
        
        # Detect spike
        return self.V >= 30.0


class BiophysicalSynapse:
    """Biophysical synapse model."""
    
    def __init__(self, E_rev: float = 0.0, tau_rise: float = 0.5, tau_decay: float = 5.0):
        """Initialize synapse."""
        self.E_rev = E_rev
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.g = 0.0  # Conductance
        self.g_max = 0.1  # Maximum conductance
    
    def update(self, dt: float, V_post: float, spike: bool = False) -> float:
        """Update synapse and return current."""
        if spike:
            self.g = self.g_max
        
        # Double exponential conductance
        self.g *= np.exp(-dt / self.tau_decay)
        
        return self.g * (V_post - self.E_rev)


class HHSNNController:
    """SNN controller using Hodgkin-Huxley neurons."""
    
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        """Initialize controller."""
        # Create neuron layers
        self.input_layer = [HHNeuron() for _ in range(num_inputs)]
        self.hidden_layer = [HHNeuron() for _ in range(num_hidden)]
        self.output_layer = [HHNeuron() for _ in range(num_outputs)]
        
        # Create synaptic connections
        self.ih_synapses = [[BiophysicalSynapse() for _ in range(num_hidden)] 
                           for _ in range(num_inputs)]
        self.ho_synapses = [[BiophysicalSynapse() for _ in range(num_outputs)] 
                           for _ in range(num_hidden)]
        
        self.dt = 0.01  # Integration time step (ms)
        
        logger.info(f"Created HH-based SNN controller with {num_inputs} inputs, "
                   f"{num_hidden} hidden, and {num_outputs} outputs")
    
    def step(self, inputs: List[float]) -> List[float]:
        """Process one time step."""
        # Process input layer
        input_spikes = []
        for i, (neuron, input_val) in enumerate(zip(self.input_layer, inputs)):
            if neuron.update(input_val * 10.0, self.dt):
                input_spikes.append(i)
        
        # Process hidden layer
        hidden_spikes = []
        for i, neuron in enumerate(self.hidden_layer):
            # Accumulate synaptic currents
            I_syn = 0.0
            for j, spike in enumerate(input_spikes):
                I_syn += self.ih_synapses[j][i].update(self.dt, neuron.V, spike)
            
            if neuron.update(I_syn, self.dt):
                hidden_spikes.append(i)
        
        # Process output layer
        outputs = []
        for i, neuron in enumerate(self.output_layer):
            # Accumulate synaptic currents
            I_syn = 0.0
            for j, spike in enumerate(hidden_spikes):
                I_syn += self.ho_synapses[j][i].update(self.dt, neuron.V, spike)
            
            spiked = neuron.update(I_syn, self.dt)
            outputs.append(float(spiked))
        
        return outputs
"""
Liquid State Machine Neural Network Controller

A reservoir computing approach to SNN-based control using LSM.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.spatial.distance import euclidean

from src.core.utils.logging_framework import get_logger

logger = get_logger("lsm_snn_controller")


@dataclass
class LSMNeuronParams:
    """Parameters for LSM neuron."""
    
    tau_m: float = 20.0       # Membrane time constant (ms)
    v_rest: float = -70.0     # Resting potential (mV)
    v_thresh: float = -55.0   # Threshold potential (mV)
    v_reset: float = -75.0    # Reset potential (mV)
    t_refract: float = 2.0    # Refractory period (ms)


class LSMNeuron:
    """Liquid State Machine neuron."""
    
    def __init__(self, position: Tuple[float, float, float], 
                 params: Optional[LSMNeuronParams] = None):
        """Initialize LSM neuron with 3D position in liquid."""
        self.position = position
        self.params = params or LSMNeuronParams()
        
        self.v = self.params.v_rest
        self.last_spike = float('-inf')
        self.spike_trace = 0.0
        
    def update(self, t: float, I_total: float, dt: float) -> bool:
        """Update neuron state."""
        if t - self.last_spike < self.params.t_refract:
            return False
        
        # Update membrane potential
        dv = (-(self.v - self.params.v_rest) + I_total) / self.params.tau_m
        self.v += dv * dt
        
        # Update spike trace
        self.spike_trace *= np.exp(-dt / 20.0)  # Decay with 20ms time constant
        
        # Check for spike
        if self.v >= self.params.v_thresh:
            self.v = self.params.v_reset
            self.last_spike = t
            self.spike_trace = 1.0
            return True
            
        return False


class LiquidColumn:
    """Column of neurons in the liquid."""
    
    def __init__(self, height: float, num_neurons: int):
        """Initialize column of neurons."""
        self.neurons = [
            LSMNeuron(position=(0.0, 0.0, h))
            for h in np.linspace(0, height, num_neurons)
        ]
        self.connections: Dict[Tuple[int, int], float] = {}
        
        # Create internal connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize internal connections based on distance."""
        for i, pre in enumerate(self.neurons):
            for j, post in enumerate(self.neurons):
                if i != j:
                    dist = euclidean(pre.position, post.position)
                    if dist < 1.0:  # Connection probability drops with distance
                        prob = np.exp(-dist)
                        if np.random.random() < prob:
                            self.connections[(i, j)] = np.random.normal(0.5, 0.1)


class LiquidStateController:
    """SNN controller using Liquid State Machine."""
    
    def __init__(self, num_inputs: int, num_outputs: int, 
                 liquid_columns: int = 5, neurons_per_column: int = 10):
        """Initialize LSM controller."""
        # Create input and output layers
        self.input_layer = [LSMNeuron((0, 0, 0)) for _ in range(num_inputs)]
        self.output_layer = [LSMNeuron((0, 0, 0)) for _ in range(num_outputs)]
        
        # Create liquid columns
        self.columns = [LiquidColumn(height=1.0, num_neurons=neurons_per_column)
                       for _ in range(liquid_columns)]
        
        # Create input and output weights
        self.input_weights = np.random.randn(num_inputs, liquid_columns) * 0.1
        self.output_weights = np.random.randn(liquid_columns * neurons_per_column, 
                                            num_outputs) * 0.1
        
        self.t = 0.0
        self.dt = 0.1  # Time step (ms)
        
        logger.info(f"Created LSM controller with {num_inputs} inputs, "
                   f"{liquid_columns} columns, and {num_outputs} outputs")
    
    def step(self, inputs: List[float]) -> List[float]:
        """Process one time step."""
        self.t += self.dt
        
        # Process input layer
        input_currents = []
        for i, (neuron, input_val) in enumerate(zip(self.input_layer, inputs)):
            if neuron.update(self.t, input_val * 20.0, self.dt):
                input_currents.extend(self.input_weights[i])
            else:
                input_currents.extend([0.0] * len(self.columns))
        
        # Process liquid
        liquid_states = []
        for col, current in zip(self.columns, input_currents):
            column_states = []
            for i, neuron in enumerate(col.neurons):
                # Calculate total input current
                I_syn = current
                for (pre, post), weight in col.connections.items():
                    if post == i and col.neurons[pre].spike_trace > 0:
                        I_syn += weight * col.neurons[pre].spike_trace
                
                # Update neuron
                spiked = neuron.update(self.t, I_syn, self.dt)
                column_states.append(neuron.spike_trace)
            
            liquid_states.extend(column_states)
        
        # Process output layer
        outputs = []
        liquid_states = np.array(liquid_states)
        output_currents = np.dot(liquid_states, self.output_weights)
        
        for neuron, current in zip(self.output_layer, output_currents):
            spiked = neuron.update(self.t, current, self.dt)
            outputs.append(float(spiked))
        
        return outputs
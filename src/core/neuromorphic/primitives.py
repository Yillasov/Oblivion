#!/usr/bin/env python3
"""
Neuromorphic Computing Primitives

This module provides fundamental building blocks for neuromorphic computing,
including neuron models, synaptic plasticity rules, and network structures.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NeuronModel:
    """Base class for all neuron models."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the neuron model with parameters.
        
        Args:
            params: Dictionary of model parameters
        """
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset the neuron state."""
        self.potential = 0.0
        self.last_spike_time = -1.0
        self.spike_history = []
    
    def update(self, inputs: float, time: float) -> bool:
        """
        Update neuron state and determine if it spikes.
        
        Args:
            inputs: Total input to the neuron
            time: Current simulation time
            
        Returns:
            bool: True if neuron spikes, False otherwise
        """
        raise NotImplementedError("Subclasses must implement update method")


class LIFNeuron(NeuronModel):
    """Leaky Integrate-and-Fire neuron model."""
    
    def update(self, inputs: float, time: float) -> bool:
        """
        Update LIF neuron state and determine if it spikes.
        
        Args:
            inputs: Total input to the neuron
            time: Current simulation time
            
        Returns:
            bool: True if neuron spikes, False otherwise
        """
        # Extract parameters
        tau = self.params.get('tau', 10.0)  # Membrane time constant (ms)
        threshold = self.params.get('threshold', 1.0)  # Firing threshold
        refractory_period = self.params.get('refractory_period', 2.0)  # Refractory period (ms)
        
        # Check if in refractory period
        if time - self.last_spike_time < refractory_period:
            return False
        
        # Update membrane potential
        dt = 1.0  # Assuming 1ms time step
        decay = np.exp(-dt / tau)
        self.potential = decay * self.potential + inputs
        
        # Check for spike
        if self.potential >= threshold:
            self.potential = 0.0  # Reset potential
            self.last_spike_time = time
            self.spike_history.append(time)
            return True
        
        return False


class IzhikevichNeuron(NeuronModel):
    """Izhikevich neuron model."""
    
    def reset(self):
        """Reset the neuron state."""
        super().reset()
        self.recovery = 0.0  # Recovery variable
    
    def update(self, inputs: float, time: float) -> bool:
        """
        Update Izhikevich neuron state and determine if it spikes.
        
        Args:
            inputs: Total input to the neuron
            time: Current simulation time
            
        Returns:
            bool: True if neuron spikes, False otherwise
        """
        # Extract parameters (default to regular spiking)
        a = self.params.get('a', 0.02)  # Recovery time constant
        b = self.params.get('b', 0.2)   # Sensitivity of recovery variable
        c = self.params.get('c', -65.0)  # Post-spike reset value for potential
        d = self.params.get('d', 8.0)   # Post-spike reset value for recovery
        
        # Update membrane potential and recovery variable
        dt = 0.5  # Smaller time step for numerical stability
        
        self.potential += dt * (0.04 * self.potential**2 + 5 * self.potential + 140 - self.recovery + inputs)
        self.recovery += dt * a * (b * self.potential - self.recovery)
        
        # Check for spike
        if self.potential >= 30.0:
            self.potential = c
            self.recovery += d
            self.last_spike_time = time
            self.spike_history.append(time)
            return True
        
        return False


class SynapticPlasticityRule:
    """Base class for synaptic plasticity rules."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the plasticity rule with parameters.
        
        Args:
            params: Dictionary of rule parameters
        """
        self.params = params
    
    def update_weight(self, weight: float, pre_spikes: List[float], post_spikes: List[float], 
                      current_time: float) -> float:
        """
        Update synaptic weight based on pre and post-synaptic activity.
        
        Args:
            weight: Current synaptic weight
            pre_spikes: List of pre-synaptic spike times
            post_spikes: List of post-synaptic spike times
            current_time: Current simulation time
            
        Returns:
            float: Updated synaptic weight
        """
        raise NotImplementedError("Subclasses must implement update_weight method")


class STDPRule(SynapticPlasticityRule):
    """Spike-Timing-Dependent Plasticity rule."""
    
    def update_weight(self, weight: float, pre_spikes: List[float], post_spikes: List[float], 
                      current_time: float) -> float:
        """
        Update synaptic weight using STDP rule.
        
        Args:
            weight: Current synaptic weight
            pre_spikes: List of pre-synaptic spike times
            post_spikes: List of post-synaptic spike times
            current_time: Current simulation time
            
        Returns:
            float: Updated synaptic weight
        """
        # Extract parameters
        a_plus = self.params.get('a_plus', 0.01)    # Potentiation strength
        a_minus = self.params.get('a_minus', 0.0105)  # Depression strength
        tau_plus = self.params.get('tau_plus', 20.0)   # Potentiation time constant (ms)
        tau_minus = self.params.get('tau_minus', 20.0)  # Depression time constant (ms)
        w_min = self.params.get('w_min', 0.0)       # Minimum weight
        w_max = self.params.get('w_max', 1.0)       # Maximum weight
        
        # Filter recent spikes
        recent_pre_spikes = [t for t in pre_spikes if current_time - t <= 5 * tau_plus]
        recent_post_spikes = [t for t in post_spikes if current_time - t <= 5 * tau_minus]
        
        # No recent spikes, no weight change
        if not recent_pre_spikes and not recent_post_spikes:
            return weight
        
        # Calculate weight change
        dw = 0.0
        
        # For each post-synaptic spike, check for preceding pre-synaptic spikes
        for post_time in recent_post_spikes:
            for pre_time in recent_pre_spikes:
                dt = pre_time - post_time
                if dt < 0:  # Pre before post: potentiation
                    dw += a_plus * np.exp(dt / tau_plus)
        
        # For each pre-synaptic spike, check for preceding post-synaptic spikes
        for pre_time in recent_pre_spikes:
            for post_time in recent_post_spikes:
                dt = pre_time - post_time
                if dt > 0:  # Post before pre: depression
                    dw -= a_minus * np.exp(-dt / tau_minus)
        
        # Update weight with bounds
        new_weight = weight + dw
        return np.clip(new_weight, w_min, w_max)


class NetworkTopology:
    """Base class for network topologies."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the network topology with parameters.
        
        Args:
            params: Dictionary of topology parameters
        """
        self.params = params
    
    def generate_connections(self, pre_neurons: int, post_neurons: int) -> List[Tuple[int, int, float]]:
        """
        Generate synaptic connections between neuron populations.
        
        Args:
            pre_neurons: Number of pre-synaptic neurons
            post_neurons: Number of post-synaptic neurons
            
        Returns:
            List[Tuple[int, int, float]]: List of connections (pre_id, post_id, weight)
        """
        raise NotImplementedError("Subclasses must implement generate_connections method")


class FullyConnectedTopology(NetworkTopology):
    """Fully connected network topology."""
    
    def generate_connections(self, pre_neurons: int, post_neurons: int) -> List[Tuple[int, int, float]]:
        """
        Generate fully connected synaptic connections.
        
        Args:
            pre_neurons: Number of pre-synaptic neurons
            post_neurons: Number of post-synaptic neurons
            
        Returns:
            List[Tuple[int, int, float]]: List of connections (pre_id, post_id, weight)
        """
        # Extract parameters
        weight_mean = self.params.get('weight_mean', 0.5)
        weight_std = self.params.get('weight_std', 0.1)
        
        connections = []
        
        for i in range(pre_neurons):
            for j in range(post_neurons):
                # Generate random weight
                weight = np.random.normal(weight_mean, weight_std)
                connections.append((i, j, weight))
        
        return connections


class RandomTopology(NetworkTopology):
    """Random network topology with connection probability."""
    
    def generate_connections(self, pre_neurons: int, post_neurons: int) -> List[Tuple[int, int, float]]:
        """
        Generate random synaptic connections.
        
        Args:
            pre_neurons: Number of pre-synaptic neurons
            post_neurons: Number of post-synaptic neurons
            
        Returns:
            List[Tuple[int, int, float]]: List of connections (pre_id, post_id, weight)
        """
        # Extract parameters
        connection_prob = self.params.get('connection_prob', 0.1)
        weight_mean = self.params.get('weight_mean', 0.5)
        weight_std = self.params.get('weight_std', 0.1)
        
        connections = []
        
        for i in range(pre_neurons):
            for j in range(post_neurons):
                # Decide whether to create connection
                if np.random.random() < connection_prob:
                    # Generate random weight
                    weight = np.random.normal(weight_mean, weight_std)
                    connections.append((i, j, weight))
        
        return connections


# Factory functions to create instances

def create_neuron_model(model_type: str, params: Dict[str, Any]) -> NeuronModel:
    """
    Create a neuron model instance.
    
    Args:
        model_type: Type of neuron model ('lif', 'izhikevich')
        params: Model parameters
        
    Returns:
        NeuronModel: Instance of the specified neuron model
    """
    model_map = {
        'lif': LIFNeuron,
        'izhikevich': IzhikevichNeuron
    }
    
    if model_type.lower() not in model_map:
        raise ValueError(f"Unknown neuron model type: {model_type}")
    
    return model_map[model_type.lower()](params)


def create_plasticity_rule(rule_type: str, params: Dict[str, Any]) -> SynapticPlasticityRule:
    """
    Create a synaptic plasticity rule instance.
    
    Args:
        rule_type: Type of plasticity rule ('stdp')
        params: Rule parameters
        
    Returns:
        SynapticPlasticityRule: Instance of the specified plasticity rule
    """
    rule_map = {
        'stdp': STDPRule
    }
    
    if rule_type.lower() not in rule_map:
        raise ValueError(f"Unknown plasticity rule type: {rule_type}")
    
    return rule_map[rule_type.lower()](params)


def create_network_topology(topology_type: str, params: Dict[str, Any]) -> NetworkTopology:
    """
    Create a network topology instance.
    
    Args:
        topology_type: Type of network topology ('fully_connected', 'random')
        params: Topology parameters
        
    Returns:
        NetworkTopology: Instance of the specified network topology
    """
    topology_map = {
        'fully_connected': FullyConnectedTopology,
        'random': RandomTopology
    }
    
    if topology_type.lower() not in topology_map:
        raise ValueError(f"Unknown network topology type: {topology_type}")
    
    return topology_map[topology_type.lower()](params)
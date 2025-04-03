#!/usr/bin/env python3
"""
Simple STDP Learning Algorithm for Neuromorphic Hardware

Implements a basic Spike-Timing-Dependent Plasticity (STDP) learning rule
for neuromorphic hardware integration.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from src.core.utils.logging_framework import get_logger

logger = get_logger("simple_stdp")


class SimpleSTDP:
    """
    Simple implementation of Spike-Timing-Dependent Plasticity (STDP) learning rule.
    
    STDP is a biological learning mechanism where the strength of connections
    between neurons is adjusted based on the relative timing of spikes.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 time_window: float = 20.0,  # ms
                 a_plus: float = 1.0,
                 a_minus: float = -1.0,
                 tau_plus: float = 20.0,  # ms
                 tau_minus: float = 20.0):  # ms
        """
        Initialize STDP learning algorithm.
        
        Args:
            learning_rate: Overall learning rate
            time_window: Time window for considering spike pairs (ms)
            a_plus: Amplitude of potentiation
            a_minus: Amplitude of depression
            tau_plus: Time constant for potentiation (ms)
            tau_minus: Time constant for depression (ms)
        """
        self.learning_rate = learning_rate
        self.time_window = time_window
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        
        # Spike history for each neuron
        self.spike_history = {}
        
        logger.info("Initialized simple STDP learning algorithm")
    
    def reset(self):
        """Reset the learning algorithm state."""
        self.spike_history = {}
        logger.debug("Reset STDP learning state")
    
    def record_spike(self, neuron_id: int, spike_time: float):
        """
        Record a spike for a neuron.
        
        Args:
            neuron_id: ID of the neuron that spiked
            spike_time: Time of the spike in ms
        """
        if neuron_id not in self.spike_history:
            self.spike_history[neuron_id] = []
        
        # Add spike to history
        self.spike_history[neuron_id].append(spike_time)
        
        # Keep only recent spikes within time window
        self.spike_history[neuron_id] = [t for t in self.spike_history[neuron_id] 
                                        if spike_time - t <= self.time_window]
    
    def compute_weight_change(self, pre_id: int, post_id: int, current_time: float) -> float:
        """
        Compute weight change for a synapse based on STDP rule.
        
        Args:
            pre_id: ID of presynaptic neuron
            post_id: ID of postsynaptic neuron
            current_time: Current simulation time in ms
            
        Returns:
            float: Weight change value
        """
        # If either neuron has no spike history, no weight change
        if pre_id not in self.spike_history or post_id not in self.spike_history:
            return 0.0
        
        weight_change = 0.0
        
        # Get spike histories
        pre_spikes = self.spike_history[pre_id]
        post_spikes = self.spike_history[post_id]
        
        # For each post-synaptic spike
        for t_post in post_spikes:
            # For each pre-synaptic spike
            for t_pre in pre_spikes:
                # Compute time difference
                delta_t = t_post - t_pre
                
                # Apply STDP rule
                if delta_t > 0:  # Post after pre (potentiation)
                    weight_change += self.a_plus * np.exp(-delta_t / self.tau_plus)
                elif delta_t < 0:  # Pre after post (depression)
                    weight_change += self.a_minus * np.exp(delta_t / self.tau_minus)
        
        return weight_change * self.learning_rate
    
    def update_weights(self, connections: Dict[Tuple[int, int], float], 
                      current_time: float) -> Dict[Tuple[int, int], float]:
        """
        Update weights for all connections based on STDP rule.
        
        Args:
            connections: Dictionary mapping (pre_id, post_id) to weight
            current_time: Current simulation time in ms
            
        Returns:
            Dict[Tuple[int, int], float]: Updated connection weights
        """
        updated_connections = connections.copy()
        
        for (pre_id, post_id), weight in connections.items():
            # Compute weight change
            delta_w = self.compute_weight_change(pre_id, post_id, current_time)
            
            # Update weight
            new_weight = weight + delta_w
            
            # Apply bounds (0 to 1)
            new_weight = max(0.0, min(1.0, new_weight))
            
            updated_connections[(pre_id, post_id)] = new_weight
            
            if abs(delta_w) > 0.001:
                logger.debug(f"Updated weight ({pre_id}, {post_id}): {weight:.4f} -> {new_weight:.4f}")
        
        return updated_connections
    
    def apply_to_hardware(self, hardware_interface, connections: Dict[Tuple[int, int], float]):
        """
        Apply updated weights to neuromorphic hardware.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
            connections: Dictionary mapping (pre_id, post_id) to weight
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to format expected by hardware
            hw_connections = []
            for (pre_id, post_id), weight in connections.items():
                hw_connections.append((pre_id, post_id, weight))
            
            # Update weights on hardware
            hardware_interface.update_synaptic_weights(hw_connections)
            logger.info(f"Applied {len(connections)} weight updates to hardware")
            return True
        except Exception as e:
            logger.error(f"Failed to apply weights to hardware: {str(e)}")
            return False


# Example integration with neuromorphic hardware
def create_simple_learning_network(hardware_interface, num_neurons: int = 10):
    """
    Create a simple learning network on neuromorphic hardware.
    
    Args:
        hardware_interface: Interface to neuromorphic hardware
        num_neurons: Number of neurons in the network
        
    Returns:
        Tuple[List[int], Dict[Tuple[int, int], float], SimpleSTDP]: 
            Neuron IDs, connections, and STDP instance
    """
    # Initialize STDP learning rule
    stdp = SimpleSTDP()
    
    # Allocate neurons on hardware
    neuron_params = {
        'threshold': 1.0,
        'decay': 0.9,
        'refactory_period': 1.0
    }
    neuron_ids = hardware_interface.allocate_neurons(num_neurons, neuron_params)
    
    # Create initial random connections
    connections = {}
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i != j and np.random.random() < 0.3:  # 30% connectivity
                connections[(neuron_ids[i], neuron_ids[j])] = np.random.random() * 0.5
    
    # Create synapses on hardware
    hw_connections = [(pre, post, weight) for (pre, post), weight in connections.items()]
    hardware_interface.create_synapses(hw_connections)
    
    logger.info(f"Created learning network with {num_neurons} neurons and {len(connections)} connections")
    
    return neuron_ids, connections, stdp
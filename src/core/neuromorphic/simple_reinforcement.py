#!/usr/bin/env python3
"""
Simple Reinforcement Learning Algorithm for Neuromorphic Hardware

Implements a basic reward-based learning rule for neuromorphic hardware integration.
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

logger = get_logger("simple_reinforcement")


class SimpleReinforcement:
    """
    Simple implementation of reward-based reinforcement learning.
    
    This algorithm adjusts connection weights based on reward signals,
    strengthening connections that lead to positive outcomes.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.05,
                 eligibility_decay: float = 0.9,
                 reward_window: int = 10):
        """
        Initialize reinforcement learning algorithm.
        
        Args:
            learning_rate: Rate at which connections are modified
            eligibility_decay: Decay rate for eligibility traces
            reward_window: Number of timesteps to consider for eligibility
        """
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay
        self.reward_window = reward_window
        
        # Eligibility traces for each connection
        self.eligibility_traces = {}
        
        # Activity history for each neuron
        self.activity_history = {}
        
        # Current timestep
        self.timestep = 0
        
        logger.info("Initialized simple reinforcement learning algorithm")
    
    def reset(self):
        """Reset the learning algorithm state."""
        self.eligibility_traces = {}
        self.activity_history = {}
        self.timestep = 0
        logger.debug("Reset reinforcement learning state")
    
    def record_activity(self, neuron_id: int, activity: float):
        """
        Record neuron activity for the current timestep.
        
        Args:
            neuron_id: ID of the neuron
            activity: Activity level (0.0 to 1.0)
        """
        if neuron_id not in self.activity_history:
            self.activity_history[neuron_id] = {}
        
        self.activity_history[neuron_id][self.timestep] = activity
    
    def update_eligibility(self, connections: Dict[Tuple[int, int], float]):
        """
        Update eligibility traces for all connections.
        
        Args:
            connections: Dictionary mapping (pre_id, post_id) to weight
        """
        # Decay existing eligibility traces
        for key in self.eligibility_traces:
            self.eligibility_traces[key] *= self.eligibility_decay
        
        # Update eligibility based on current activity
        for (pre_id, post_id) in connections:
            if pre_id in self.activity_history and post_id in self.activity_history:
                # Get current activities
                pre_activity = self.activity_history[pre_id].get(self.timestep, 0.0)
                post_activity = self.activity_history[post_id].get(self.timestep, 0.0)
                
                # Update eligibility trace
                key = (pre_id, post_id)
                if key not in self.eligibility_traces:
                    self.eligibility_traces[key] = 0.0
                
                # Eligibility increases when both pre and post neurons are active
                self.eligibility_traces[key] += pre_activity * post_activity
        
        self.timestep += 1
    
    def apply_reward(self, connections: Dict[Tuple[int, int], float], 
                    reward: float) -> Dict[Tuple[int, int], float]:
        """
        Apply reward signal to update connection weights.
        
        Args:
            connections: Dictionary mapping (pre_id, post_id) to weight
            reward: Reward signal (-1.0 to 1.0)
            
        Returns:
            Dict[Tuple[int, int], float]: Updated connection weights
        """
        updated_connections = connections.copy()
        
        # Apply reward to eligible connections
        for (pre_id, post_id), weight in connections.items():
            key = (pre_id, post_id)
            if key in self.eligibility_traces and self.eligibility_traces[key] > 0:
                # Update weight based on eligibility and reward
                delta_w = self.learning_rate * reward * self.eligibility_traces[key]
                new_weight = weight + delta_w
                
                # Apply bounds (0 to 1)
                new_weight = max(0.0, min(1.0, new_weight))
                
                updated_connections[key] = new_weight
                
                if abs(delta_w) > 0.001:
                    logger.debug(f"Updated weight ({pre_id}, {post_id}): {weight:.4f} -> {new_weight:.4f}")
        
        return updated_connections
    
    def clean_history(self):
        """Remove old activity history to save memory."""
        cutoff = self.timestep - self.reward_window
        if cutoff > 0:
            for neuron_id in self.activity_history:
                self.activity_history[neuron_id] = {
                    t: a for t, a in self.activity_history[neuron_id].items() 
                    if t >= cutoff
                }
    
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
def create_reinforcement_network(hardware_interface, num_neurons: int = 10):
    """
    Create a simple reinforcement learning network on neuromorphic hardware.
    
    Args:
        hardware_interface: Interface to neuromorphic hardware
        num_neurons: Number of neurons in the network
        
    Returns:
        Tuple[List[int], Dict[Tuple[int, int], float], SimpleReinforcement]: 
            Neuron IDs, connections, and RL instance
    """
    # Initialize reinforcement learning rule
    rl = SimpleReinforcement()
    
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
    
    logger.info(f"Created reinforcement learning network with {num_neurons} neurons and {len(connections)} connections")
    
    return neuron_ids, connections, rl
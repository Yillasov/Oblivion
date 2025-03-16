"""
Simple Hebbian Learning Algorithm for Neuromorphic Hardware

Implements a basic Hebbian learning rule for neuromorphic hardware integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from src.core.utils.logging_framework import get_logger

logger = get_logger("simple_hebbian")


class SimpleHebbian:
    """
    Simple implementation of Hebbian learning rule.
    
    Hebbian learning follows the principle "neurons that fire together, wire together",
    strengthening connections between neurons that are active simultaneously.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 decay_rate: float = 0.0001,
                 activity_threshold: float = 0.5):
        """
        Initialize Hebbian learning algorithm.
        
        Args:
            learning_rate: Rate at which connections strengthen
            decay_rate: Rate at which unused connections weaken
            activity_threshold: Minimum activity level to consider a neuron active
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.activity_threshold = activity_threshold
        
        # Current activity level of each neuron
        self.neuron_activity = {}
        
        logger.info("Initialized simple Hebbian learning algorithm")
    
    def reset(self):
        """Reset the learning algorithm state."""
        self.neuron_activity = {}
        logger.debug("Reset Hebbian learning state")
    
    def update_activity(self, neuron_id: int, activity_value: float):
        """
        Update activity level for a neuron.
        
        Args:
            neuron_id: ID of the neuron
            activity_value: Current activity level (0.0 to 1.0)
        """
        self.neuron_activity[neuron_id] = activity_value
    
    def is_active(self, neuron_id: int) -> bool:
        """
        Check if a neuron is currently active.
        
        Args:
            neuron_id: ID of the neuron
            
        Returns:
            bool: True if neuron is active, False otherwise
        """
        if neuron_id not in self.neuron_activity:
            return False
        
        return self.neuron_activity[neuron_id] >= self.activity_threshold
    
    def update_weights(self, connections: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """
        Update weights for all connections based on Hebbian rule.
        
        Args:
            connections: Dictionary mapping (pre_id, post_id) to weight
            
        Returns:
            Dict[Tuple[int, int], float]: Updated connection weights
        """
        updated_connections = connections.copy()
        
        for (pre_id, post_id), weight in connections.items():
            # Check if both neurons are active
            pre_active = self.is_active(pre_id)
            post_active = self.is_active(post_id)
            
            if pre_active and post_active:
                # Strengthen connection (Hebbian learning)
                delta_w = self.learning_rate * self.neuron_activity[pre_id] * self.neuron_activity[post_id]
                new_weight = weight + delta_w
            else:
                # Weaken connection (passive decay)
                new_weight = weight * (1.0 - self.decay_rate)
            
            # Apply bounds (0 to 1)
            new_weight = max(0.0, min(1.0, new_weight))
            
            updated_connections[(pre_id, post_id)] = new_weight
            
            if abs(new_weight - weight) > 0.001:
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
def create_hebbian_network(hardware_interface, num_neurons: int = 10):
    """
    Create a simple Hebbian learning network on neuromorphic hardware.
    
    Args:
        hardware_interface: Interface to neuromorphic hardware
        num_neurons: Number of neurons in the network
        
    Returns:
        Tuple[List[int], Dict[Tuple[int, int], float], SimpleHebbian]: 
            Neuron IDs, connections, and Hebbian instance
    """
    # Initialize Hebbian learning rule
    hebbian = SimpleHebbian()
    
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
    
    logger.info(f"Created Hebbian network with {num_neurons} neurons and {len(connections)} connections")
    
    return neuron_ids, connections, hebbian
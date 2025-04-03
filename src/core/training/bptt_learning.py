#!/usr/bin/env python3
"""
Backpropagation Through Time (BPTT) Learning Component for Simulation Training

Implements a simplified BPTT algorithm for training spiking neural networks.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time

from src.core.utils.logging_framework import get_logger

logger = get_logger("bptt_learning")


class BPTTLearningComponent:
    """
    Backpropagation Through Time Learning Component for simulation training.
    
    Implements a simplified BPTT algorithm for training spiking neural networks
    in simulated neuromorphic hardware environments.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BPTT learning component.
        
        Args:
            config: Configuration parameters for BPTT learning
        """
        self.config = config or {}
        
        # Learning parameters
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.time_window = self.config.get('time_window', 10)  # Time steps to backpropagate
        self.batch_size = self.config.get('batch_size', 1)
        
        # Connection weights
        self.connections = {}
        
        # Store activity history for BPTT
        self.activity_history = []
        self.target_history = []
        self.current_batch = 0
        
        logger.info("Initialized BPTT learning component")
    
    def initialize(self, initial_connections: Dict[Tuple[int, int], float]) -> None:
        """
        Initialize connections with weights.
        
        Args:
            initial_connections: Dictionary mapping (pre_id, post_id) to weight
        """
        self.connections = initial_connections.copy()
        self.activity_history = []
        self.target_history = []
        self.current_batch = 0
        
        logger.info(f"Initialized {len(self.connections)} connections")
    
    def process_spikes(self, spike_events: Dict[int, List[float]]) -> None:
        """
        Process spike events and store for BPTT.
        
        Args:
            spike_events: Dictionary mapping neuron IDs to spike times
        """
        # Convert spike events to activity vector
        neuron_ids = set(spike_events.keys()).union(
            set(pre for pre, _ in self.connections.keys()),
            set(post for _, post in self.connections.keys())
        )
        max_id = max(neuron_ids) if neuron_ids else 0
        
        # Create activity vector (binary: spiked or not in this time step)
        activity = np.zeros(max_id + 1)
        for neuron_id, spike_times in spike_events.items():
            if spike_times:  # If there are any spikes
                activity[neuron_id] = 1.0
        
        # Store activity for BPTT
        self.activity_history.append(activity)
        
        # Keep only the last time_window steps
        if len(self.activity_history) > self.time_window:
            self.activity_history.pop(0)
    
    def set_target(self, target: np.ndarray) -> None:
        """
        Set target output for current time step.
        
        Args:
            target: Target output vector
        """
        self.target_history.append(target)
        
        # Keep only the last time_window steps
        if len(self.target_history) > self.time_window:
            self.target_history.pop(0)
    
    def set_reward(self, reward: float) -> None:
        """
        Set reward signal (no-op for BPTT learning).
        
        This method exists for interface compatibility with other learning components.
        
        Args:
            reward: Reward value (-1.0 to 1.0)
        """
        logger.debug("Reward signals not used in BPTT learning")
        pass
    
    def update_weights(self) -> None:
        """
        Update weights using BPTT algorithm.
        """
        if len(self.activity_history) < 2 or len(self.target_history) < 1:
            return
        
        # Increment batch counter
        self.current_batch += 1
        
        # Only update weights after collecting a full batch
        if self.current_batch < self.batch_size:
            return
        
        self.current_batch = 0
        
        # Simple BPTT implementation
        # For each time step in our history (going backwards)
        gradients = {conn: 0.0 for conn in self.connections}
        
        for t in range(len(self.activity_history) - 1, 0, -1):
            # Current activity and previous activity
            curr_act = self.activity_history[t]
            prev_act = self.activity_history[t-1]
            
            # Target for current time step (if available)
            target = self.target_history[min(t, len(self.target_history)-1)]
            
            # Compute error (simplified)
            error = target - curr_act
            
            # Update gradients for each connection
            for (pre_id, post_id), weight in self.connections.items():
                if pre_id < len(prev_act) and post_id < len(curr_act):
                    # Simplified gradient calculation
                    if post_id < len(target):
                        # For output neurons
                        delta = error[post_id] * prev_act[pre_id]
                    else:
                        # For hidden neurons (simplified)
                        delta = 0.1 * prev_act[pre_id] * curr_act[post_id] * (1 - curr_act[post_id])
                    
                    gradients[(pre_id, post_id)] += delta
        
        # Apply gradients
        for conn, grad in gradients.items():
            self.connections[conn] += self.learning_rate * grad
        
        logger.debug(f"Updated {len(self.connections)} connections using BPTT")
    
    def get_connections(self) -> Dict[Tuple[int, int], float]:
        """
        Get current connection weights.
        
        Returns:
            Dict[Tuple[int, int], float]: Current connection weights
        """
        return self.connections
    
    def apply_to_hardware(self, hardware_interface) -> bool:
        """
        Apply updated weights to hardware.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
            
        Returns:
            bool: Success status
        """
        try:
            # Update weights first
            self.update_weights()
            
            # Convert connections to format expected by hardware
            hw_connections = [(pre, post, weight) for (pre, post), weight in self.connections.items()]
            
            # Update synapses on hardware
            hardware_interface.update_synapses(hw_connections)
            
            logger.info(f"Applied {len(hw_connections)} connection updates to hardware")
            return True
        except Exception as e:
            logger.error(f"Failed to apply connections to hardware: {str(e)}")
            return False


def create_bptt_component(config: Optional[Dict[str, Any]] = None) -> BPTTLearningComponent:
    """
    Create a BPTT learning component with the given configuration.
    
    Args:
        config: Configuration parameters
        
    Returns:
        BPTTLearningComponent: Initialized BPTT component
    """
    return BPTTLearningComponent(config)
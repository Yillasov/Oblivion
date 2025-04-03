#!/usr/bin/env python3
"""
Reinforcement Learning Component for Simulation Training

Implements a simplified reward-based learning rule for use with the simulation training framework.
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
from src.core.neuromorphic.simple_reinforcement import SimpleReinforcement

logger = get_logger("reinforcement_learning")


class ReinforcementLearningComponent:
    """
    Reinforcement Learning Component for simulation training.
    
    Implements reward-based learning for spiking neural networks
    in simulated neuromorphic hardware environments.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reinforcement learning component.
        
        Args:
            config: Configuration parameters for reinforcement learning
        """
        self.config = config or {}
        
        # Create reinforcement learning algorithm with parameters from config
        self.rl = SimpleReinforcement(
            learning_rate=self.config.get('learning_rate', 0.05),
            eligibility_decay=self.config.get('eligibility_decay', 0.9),
            reward_window=self.config.get('reward_window', 10)
        )
        
        # Connection weights
        self.connections = {}
        
        # Current reward signal
        self.current_reward = 0.0
        
        logger.info("Initialized reinforcement learning component")
    
    def initialize(self, initial_connections: Dict[Tuple[int, int], float]) -> None:
        """
        Initialize connections with weights.
        
        Args:
            initial_connections: Dictionary mapping (pre_id, post_id) to weight
        """
        self.connections = initial_connections.copy()
        self.rl.reset()
        
        logger.info(f"Initialized {len(self.connections)} connections")
    
    def process_spikes(self, spike_events: Dict[int, List[float]]) -> None:
        """
        Process spike events and update neuron activities.
        
        Args:
            spike_events: Dictionary mapping neuron IDs to spike times
        """
        # Update neuron activity based on spike count
        for neuron_id, spike_times in spike_events.items():
            # Simple activity measure: normalize spike count
            activity = min(1.0, len(spike_times) / 10.0)  # Cap at 1.0
            self.rl.record_activity(neuron_id, activity)
        
        # Update eligibility traces
        self.rl.update_eligibility(self.connections)
    
    def set_reward(self, reward: float) -> None:
        """
        Set the current reward signal.
        
        Args:
            reward: Reward value (-1.0 to 1.0)
        """
        self.current_reward = reward
        
        # Apply reward to update weights
        self.connections = self.rl.apply_reward(self.connections, reward)
    
    def set_target(self, target: np.ndarray) -> None:
        """
        Set target output (no-op for Reinforcement learning).
        
        This method exists for interface compatibility with other learning components.
        Reinforcement learning doesn't use explicit target outputs, so this method does nothing.
        
        Args:
            target: Target output vector
        """
        logger.debug("Target outputs not directly used in Reinforcement learning")
        pass
    
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
            # Convert connections to format expected by hardware
            hw_connections = [(pre, post, weight) for (pre, post), weight in self.connections.items()]
            
            # Update synapses on hardware
            hardware_interface.update_synapses(hw_connections)
            
            logger.info(f"Applied {len(hw_connections)} connection updates to hardware")
            return True
        except Exception as e:
            logger.error(f"Failed to apply connections to hardware: {str(e)}")
            return False


def create_reinforcement_component(config: Optional[Dict[str, Any]] = None) -> ReinforcementLearningComponent:
    """
    Create a reinforcement learning component with the given configuration.
    
    Args:
        config: Configuration parameters
        
    Returns:
        ReinforcementLearningComponent: Initialized reinforcement learning component
    """
    return ReinforcementLearningComponent(config)
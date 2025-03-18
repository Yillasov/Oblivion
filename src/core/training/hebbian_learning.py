"""
Hebbian Learning Component for Simulation Training

Implements a simplified Hebbian learning rule for use with the simulation training framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time

from src.core.utils.logging_framework import get_logger
from src.core.neuromorphic.simple_hebbian import SimpleHebbian

logger = get_logger("hebbian_learning")


class HebbianLearningComponent:
    """
    Hebbian Learning Component for simulation training.
    
    Implements "neurons that fire together, wire together" principle
    for simulated neuromorphic hardware environments.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Hebbian learning component.
        
        Args:
            config: Configuration parameters for Hebbian learning
        """
        self.config = config or {}
        
        # Create Hebbian algorithm with parameters from config
        self.hebbian = SimpleHebbian(
            learning_rate=self.config.get('learning_rate', 0.01),
            decay_rate=self.config.get('decay_rate', 0.0001),
            activity_threshold=self.config.get('activity_threshold', 0.5)
        )
        
        # Connection weights
        self.connections = {}
        
        logger.info("Initialized Hebbian learning component")
    
    def initialize(self, initial_connections: Dict[Tuple[int, int], float]) -> None:
        """
        Initialize connections with weights.
        
        Args:
            initial_connections: Dictionary mapping (pre_id, post_id) to weight
        """
        self.connections = initial_connections.copy()
        self.hebbian.reset()
        
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
            self.hebbian.update_activity(neuron_id, activity)
        
        # Update weights based on current activities
        self.connections = self.hebbian.update_weights(self.connections)
    
    def get_connections(self) -> Dict[Tuple[int, int], float]:
        """
        Get current connection weights.
        
        Returns:
            Dict[Tuple[int, int], float]: Current connection weights
        """
        return self.connections
    
    def set_reward(self, reward: float) -> None:
        """
        Set reward signal (no-op for Hebbian learning).
        
        This method exists for interface compatibility with other learning components.
        Hebbian learning doesn't use reward signals, so this method does nothing.
        
        Args:
            reward: Reward value (-1.0 to 1.0)
        """
        logger.debug("Reward signals not used in Hebbian learning")
        pass
    
    def set_target(self, target: np.ndarray) -> None:
        """
        Set target output (no-op for Hebbian learning).
        
        This method exists for interface compatibility with other learning components.
        Hebbian learning doesn't use target outputs, so this method does nothing.
        
        Args:
            target: Target output vector
        """
        logger.debug("Target outputs not used in Hebbian learning")
        pass
    
    def apply_to_hardware(self, hardware_interface) -> bool:
        """
        Apply updated weights to hardware.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
            
        Returns:
            bool: Success status
        """
        return self.hebbian.apply_to_hardware(hardware_interface, self.connections)


def create_hebbian_component(config: Optional[Dict[str, Any]] = None) -> HebbianLearningComponent:
    """
    Create a Hebbian learning component with the given configuration.
    
    Args:
        config: Configuration parameters
        
    Returns:
        HebbianLearningComponent: Initialized Hebbian component
    """
    return HebbianLearningComponent(config)
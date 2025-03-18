"""
Loss functions for Spiking Neural Networks.
"""

from typing import Dict, Any, Optional
import numpy as np

def spike_count_loss(output_spikes: np.ndarray, target_spikes: np.ndarray) -> float:
    """
    Calculate loss based on difference in spike counts.
    
    Args:
        output_spikes: Output spike train (neurons, time_steps)
        target_spikes: Target spike train (neurons, time_steps)
        
    Returns:
        Loss value
    """
    # Count spikes for each neuron
    output_counts = np.sum(output_spikes, axis=1)
    target_counts = np.sum(target_spikes, axis=1)
    
    # Calculate mean squared error of spike counts
    return np.mean((output_counts - target_counts) ** 2)

def van_rossum_distance(output_spikes: np.ndarray, target_spikes: np.ndarray, 
                       tau: float = 10.0, dt: float = 1.0) -> float:
    """
    Calculate van Rossum distance between spike trains.
    
    Args:
        output_spikes: Output spike train (neurons, time_steps)
        target_spikes: Target spike train (neurons, time_steps)
        tau: Time constant (ms)
        dt: Time step (ms)
        
    Returns:
        Loss value
    """
    neurons, time_steps = output_spikes.shape
    
    # Initialize filtered spike trains
    output_filtered = np.zeros((neurons, time_steps))
    target_filtered = np.zeros((neurons, time_steps))
    
    # Apply exponential filter
    for n in range(neurons):
        for t in range(time_steps):
            if output_spikes[n, t] > 0:
                # Add exponential decay for each spike
                for t_post in range(t, time_steps):
                    output_filtered[n, t_post] += np.exp(-(t_post - t) * dt / tau)
            
            if target_spikes[n, t] > 0:
                # Add exponential decay for each spike
                for t_post in range(t, time_steps):
                    target_filtered[n, t_post] += np.exp(-(t_post - t) * dt / tau)
    
    # Calculate squared difference
    squared_diff = (output_filtered - target_filtered) ** 2
    
    # Integrate over time
    return float(np.sum(squared_diff) * dt)

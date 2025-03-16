"""
Utility functions for neuromorphic operations.

This module provides common utility functions for working with spiking neural networks
and neuromorphic hardware.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional


def encode_poisson_spike_train(rate: float, duration: float, dt: float = 0.001) -> List[float]:
    """
    Generate a Poisson spike train based on a firing rate.
    
    Args:
        rate: Firing rate in Hz
        duration: Duration of the spike train in seconds
        dt: Time step in seconds
        
    Returns:
        List[float]: List of spike times in seconds
    """
    steps = int(duration / dt)
    spike_probability = rate * dt
    spikes = np.random.random(steps) < spike_probability
    spike_times = [i * dt for i, spike in enumerate(spikes) if spike]
    return spike_times


def encode_rate_to_spike(value: float, min_val: float, max_val: float, 
                         duration: float, max_rate: float = 100.0) -> List[float]:
    """
    Encode a continuous value as a spike train using rate coding.
    
    Args:
        value: Value to encode
        min_val: Minimum value in the range
        max_val: Maximum value in the range
        duration: Duration of the spike train in seconds
        max_rate: Maximum firing rate in Hz
        
    Returns:
        List[float]: List of spike times in seconds
    """
    # Normalize value to [0, 1]
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
    
    # Convert to rate
    rate = normalized * max_rate
    
    # Generate spike train
    return encode_poisson_spike_train(rate, duration)


def decode_spike_train_to_rate(spike_times: List[float], window_size: float, 
                               duration: float) -> float:
    """
    Decode a spike train to a firing rate.
    
    Args:
        spike_times: List of spike times in seconds
        window_size: Size of the counting window in seconds
        duration: Total duration of the recording in seconds
        
    Returns:
        float: Firing rate in Hz
    """
    if not spike_times:
        return 0.0
    
    # Count spikes in the window
    count = sum(1 for t in spike_times if t <= duration)
    
    # Calculate rate
    rate = count / window_size if window_size > 0 else 0.0
    return rate


def calculate_van_rossum_distance(spike_train1: List[float], spike_train2: List[float], 
                                  tau: float = 0.02) -> float:
    """
    Calculate the van Rossum distance between two spike trains.
    
    Args:
        spike_train1: First spike train (list of spike times)
        spike_train2: Second spike train (list of spike times)
        tau: Time constant for exponential kernel in seconds
        
    Returns:
        float: van Rossum distance between the spike trains
    """
    # Simple implementation for demonstration
    # A full implementation would convolve the spike trains with an exponential kernel
    # and calculate the L2 distance between the resulting functions
    
    # For simplicity, we'll use a discrete approximation
    max_time = max(spike_train1[-1] if spike_train1 else 0, 
                   spike_train2[-1] if spike_train2 else 0)
    dt = min(0.001, tau / 10)  # Time step for discretization
    steps = int(max_time / dt) + 1
    
    # Initialize discrete signals
    signal1 = np.zeros(steps)
    signal2 = np.zeros(steps)
    
    # Convolve spikes with exponential kernel
    for t in spike_train1:
        idx = int(t / dt)
        if idx < steps:
            times = np.arange(idx, steps)
            kernel = np.exp(-(times - idx) * dt / tau)
            signal1[idx:] += kernel[:len(signal1[idx:])]
    
    for t in spike_train2:
        idx = int(t / dt)
        if idx < steps:
            times = np.arange(idx, steps)
            kernel = np.exp(-(times - idx) * dt / tau)
            signal2[idx:] += kernel[:len(signal2[idx:])]
    
    # Calculate L2 distance
    distance = np.sqrt(np.sum((signal1 - signal2) ** 2) * dt)
    return distance


def convert_weights_to_fixed_point(weights: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    Convert floating-point weights to fixed-point representation for neuromorphic hardware.
    
    Args:
        weights: Array of floating-point weights
        bits: Number of bits for fixed-point representation
        
    Returns:
        np.ndarray: Array of fixed-point weights
    """
    # Determine range of fixed-point values
    max_val = 2 ** (bits - 1) - 1
    min_val = -2 ** (bits - 1)
    
    # Scale weights to use full range
    abs_max = np.max(np.abs(weights))
    if abs_max > 0:
        scale_factor = max_val / abs_max
        scaled_weights = weights * scale_factor
    else:
        scaled_weights = weights
    
    # Convert to integers
    fixed_weights = np.clip(np.round(scaled_weights), min_val, max_val).astype(np.int32)
    return fixed_weights


def generate_synfire_chain(layers: List[int], connection_probability: float = 1.0) -> List[Tuple[int, int, float]]:
    """
    Generate a synfire chain network topology.
    
    Args:
        layers: List of neuron counts for each layer
        connection_probability: Probability of connection between neurons in adjacent layers
        
    Returns:
        List[Tuple[int, int, float]]: List of connections (pre_id, post_id, weight)
    """
    connections = []
    neuron_id_offset = 0
    
    for i in range(len(layers) - 1):
        layer_size = layers[i]
        next_layer_size = layers[i + 1]
        
        for pre in range(neuron_id_offset, neuron_id_offset + layer_size):
            for post in range(neuron_id_offset + layer_size, 
                             neuron_id_offset + layer_size + next_layer_size):
                
                # Apply connection probability
                if np.random.random() < connection_probability:
                    # Use a fixed weight for simplicity
                    weight = 1.0
                    connections.append((pre, post, weight))
        
        neuron_id_offset += layer_size
    
    return connections


def calculate_neuron_parameters(neuron_type: str, time_constant_ms: float, 
                               threshold: float) -> Dict[str, Any]:
    """
    Calculate hardware-specific neuron parameters based on desired characteristics.
    
    Args:
        neuron_type: Type of neuron ('LIF', 'IF', 'Izhikevich')
        time_constant_ms: Membrane time constant in milliseconds
        threshold: Firing threshold
        
    Returns:
        Dict[str, Any]: Dictionary of hardware-specific neuron parameters
    """
    params = {
        'threshold': threshold,
    }
    
    if neuron_type == 'LIF':
        # For Leaky Integrate-and-Fire neurons
        decay = np.exp(-1.0 / time_constant_ms)  # Decay factor for discrete time steps
        params['decay'] = decay
        params['reset_potential'] = 0.0
        
    elif neuron_type == 'IF':
        # For Integrate-and-Fire neurons
        params['decay'] = 1.0  # No leak
        params['reset_potential'] = 0.0
        
    elif neuron_type == 'Izhikevich':
        # Simplified Izhikevich model parameters
        params['a'] = 0.02
        params['b'] = 0.2
        params['c'] = -65.0
        params['d'] = 8.0
        
    return params
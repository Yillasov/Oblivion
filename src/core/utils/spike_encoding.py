"""
Spike Encoding/Decoding Utilities

Provides functions to convert between conventional data formats and spike-based representations
for neuromorphic computing.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import time

from src.core.utils.logging_framework import get_logger

logger = get_logger("spike_encoding")


def encode_array(data: np.ndarray, method: str = "rate", 
                duration: float = 100.0, **kwargs) -> Dict[int, List[float]]:
    """
    Encode a numpy array into spike times.
    
    Args:
        data: Input data array (1D or flattened)
        method: Encoding method ("rate", "ttfs", "temporal", "phase")
        duration: Simulation duration in ms
        **kwargs: Additional method-specific parameters
        
    Returns:
        Dict[int, List[float]]: Dictionary mapping neuron IDs to spike times
    """
    # Ensure data is 1D
    flat_data = data.flatten()
    
    # Normalize data to [0, 1] range
    if kwargs.get("normalize", True):
        min_val = kwargs.get("min_val", flat_data.min())
        max_val = kwargs.get("max_val", flat_data.max())
        if max_val > min_val:
            normalized = (flat_data - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(flat_data)
    else:
        normalized = flat_data
    
    # Apply encoding method
    if method == "rate":
        return _rate_coding(normalized, duration, **kwargs)
    elif method == "ttfs":
        return _time_to_first_spike(normalized, duration, **kwargs)
    elif method == "temporal":
        return _temporal_coding(normalized, duration, **kwargs)
    elif method == "phase":
        return _phase_coding(normalized, duration, **kwargs)
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def _rate_coding(data: np.ndarray, duration: float, 
                max_rate: float = 10.0, **kwargs) -> Dict[int, List[float]]:
    """Rate coding: higher values = more spikes."""
    spikes = {}
    
    for i, value in enumerate(data):
        if value > 0:
            # Number of spikes proportional to value
            spike_count = int(value * max_rate)
            if spike_count > 0:
                # Distribute spikes evenly across duration
                spikes[i] = [j * (duration / spike_count) for j in range(spike_count)]
    
    return spikes


def _time_to_first_spike(data: np.ndarray, duration: float, **kwargs) -> Dict[int, List[float]]:
    """Time-to-first-spike: higher values = earlier spikes."""
    spikes = {}
    
    for i, value in enumerate(data):
        if value > 0:
            # Normalize to get spike time (higher values spike earlier)
            spike_time = (1.0 - value) * duration
            spikes[i] = [spike_time]
    
    return spikes


def _temporal_coding(data: np.ndarray, duration: float, 
                    pattern_length: int = 3, **kwargs) -> Dict[int, List[float]]:
    """Temporal coding: distribute spikes with specific patterns."""
    spikes = {}
    
    for i, value in enumerate(data):
        if value > 0:
            # Create multiple spikes with temporal pattern
            base_time = (1.0 - value) * (duration * 0.5)
            pattern = [base_time + j * (duration / (pattern_length * 2)) for j in range(pattern_length)]
            spikes[i] = pattern
    
    return spikes


def _phase_coding(data: np.ndarray, duration: float, 
                 frequency: float = 20.0, **kwargs) -> Dict[int, List[float]]:
    """Phase coding: encode values as phase shifts of periodic spikes."""
    spikes = {}
    period = 1000.0 / frequency  # Period in ms
    
    for i, value in enumerate(data):
        if value > 0:
            # Phase shift based on value
            phase_shift = value * period
            # Generate multiple spikes at regular intervals with phase shift
            spike_times = [phase_shift + j * period for j in range(int(duration / period))]
            spikes[i] = spike_times
    
    return spikes


def decode_spikes(spike_data: Dict[int, List[float]], method: str = "rate",
                 duration: float = 100.0, output_size: Optional[int] = None,
                 **kwargs) -> np.ndarray:
    """
    Decode spike times back to continuous values.
    
    Args:
        spike_data: Dictionary mapping neuron IDs to spike times
        method: Decoding method ("rate", "ttfs", "temporal", "phase")
        duration: Simulation duration in ms
        output_size: Size of output array (defaults to max neuron ID + 1)
        **kwargs: Additional method-specific parameters
        
    Returns:
        np.ndarray: Decoded values
    """
    if not spike_data:
        return np.array([])
    
    # Determine output size
    if output_size is None:
        output_size = max(spike_data.keys()) + 1
    
    # Initialize output array
    output = np.zeros(output_size)
    
    # Apply decoding method
    if method == "rate":
        return _decode_rate(spike_data, output, duration, **kwargs)
    elif method == "ttfs":
        return _decode_ttfs(spike_data, output, duration, **kwargs)
    elif method == "temporal":
        return _decode_temporal(spike_data, output, duration, **kwargs)
    elif method == "phase":
        return _decode_phase(spike_data, output, duration, **kwargs)
    else:
        raise ValueError(f"Unknown decoding method: {method}")


def _decode_rate(spike_data: Dict[int, List[float]], output: np.ndarray, 
                duration: float, **kwargs) -> np.ndarray:
    """Decode rate-coded spikes: count spikes per neuron."""
    for neuron_id, spike_times in spike_data.items():
        if neuron_id < len(output):
            # Normalize by maximum possible spike count
            max_rate = kwargs.get("max_rate", 10.0)
            output[neuron_id] = len(spike_times) / max_rate
    
    return output


def _decode_ttfs(spike_data: Dict[int, List[float]], output: np.ndarray, 
                duration: float, **kwargs) -> np.ndarray:
    """Decode time-to-first-spike: earlier spikes = higher values."""
    for neuron_id, spike_times in spike_data.items():
        if neuron_id < len(output) and spike_times:
            # First spike time determines the value
            first_spike = min(spike_times)
            # Normalize: earlier spikes (smaller times) = higher values
            output[neuron_id] = 1.0 - (first_spike / duration)
    
    return output


def _decode_temporal(spike_data: Dict[int, List[float]], output: np.ndarray, 
                    duration: float, **kwargs) -> np.ndarray:
    """Decode temporal patterns: match specific spike patterns."""
    for neuron_id, spike_times in spike_data.items():
        if neuron_id < len(output) and spike_times:
            # Use first spike time as indicator
            first_spike = min(spike_times)
            # Normalize based on when pattern starts
            output[neuron_id] = 1.0 - (first_spike / (duration * 0.5))
    
    return output


def _decode_phase(spike_data: Dict[int, List[float]], output: np.ndarray, 
                 duration: float, **kwargs) -> np.ndarray:
    """Decode phase-coded spikes: extract phase information."""
    frequency = kwargs.get("frequency", 20.0)
    period = 1000.0 / frequency
    
    for neuron_id, spike_times in spike_data.items():
        if neuron_id < len(output) and spike_times:
            # Extract phase from first spike
            first_spike = min(spike_times)
            # Normalize phase to [0, 1]
            output[neuron_id] = (first_spike % period) / period
    
    return output


def encode_dataset(data: np.ndarray, method: str = "rate", 
                  duration: float = 100.0, **kwargs) -> List[Dict[int, List[float]]]:
    """
    Encode a batch of data samples into spike times.
    
    Args:
        data: Input data array (batch_size x features)
        method: Encoding method
        duration: Simulation duration in ms
        **kwargs: Additional method-specific parameters
        
    Returns:
        List[Dict[int, List[float]]]: List of spike dictionaries, one per sample
    """
    return [encode_array(sample, method, duration, **kwargs) for sample in data]


def decode_dataset(spike_data_list: List[Dict[int, List[float]]], method: str = "rate",
                  duration: float = 100.0, output_size: Optional[int] = None,
                  **kwargs) -> np.ndarray:
    """
    Decode a batch of spike data back to continuous values.
    
    Args:
        spike_data_list: List of spike dictionaries, one per sample
        method: Decoding method
        duration: Simulation duration in ms
        output_size: Size of each output sample
        **kwargs: Additional method-specific parameters
        
    Returns:
        np.ndarray: Decoded values (batch_size x features)
    """
    decoded = [decode_spikes(spikes, method, duration, output_size, **kwargs) 
              for spikes in spike_data_list]
    return np.array(decoded)
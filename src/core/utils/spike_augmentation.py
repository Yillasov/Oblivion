#!/usr/bin/env python3
"""
Spike Data Augmentation

Provides functions to augment spike-based data for improved SNN training.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Optional, Tuple
import random

from src.core.utils.logging_framework import get_logger

logger = get_logger("spike_augmentation")


def jitter_spikes(spike_data: Dict[int, List[float]], 
                 jitter_range: float = 2.0) -> Dict[int, List[float]]:
    """
    Add temporal jitter to spike times.
    
    Args:
        spike_data: Dictionary mapping neuron IDs to spike times
        jitter_range: Maximum jitter in ms (both positive and negative)
        
    Returns:
        Dict[int, List[float]]: Augmented spike data
    """
    augmented = {}
    
    for neuron_id, spike_times in spike_data.items():
        if not spike_times:
            continue
            
        # Add random jitter to each spike time
        jittered_times = [
            max(0, t + random.uniform(-jitter_range, jitter_range)) 
            for t in spike_times
        ]
        
        augmented[neuron_id] = sorted(jittered_times)
    
    return augmented


def drop_spikes(spike_data: Dict[int, List[float]], 
               drop_prob: float = 0.1) -> Dict[int, List[float]]:
    """
    Randomly drop spikes to simulate noise/unreliability.
    
    Args:
        spike_data: Dictionary mapping neuron IDs to spike times
        drop_prob: Probability of dropping each spike
        
    Returns:
        Dict[int, List[float]]: Augmented spike data
    """
    augmented = {}
    
    for neuron_id, spike_times in spike_data.items():
        if not spike_times:
            continue
            
        # Randomly drop spikes
        kept_times = [t for t in spike_times if random.random() > drop_prob]
        
        if kept_times:  # Only add if there are remaining spikes
            augmented[neuron_id] = kept_times
    
    return augmented


def add_noise_spikes(spike_data: Dict[int, List[float]], 
                    noise_prob: float = 0.05,
                    duration: float = 100.0) -> Dict[int, List[float]]:
    """
    Add random noise spikes.
    
    Args:
        spike_data: Dictionary mapping neuron IDs to spike times
        noise_prob: Probability of adding a noise spike per neuron
        duration: Simulation duration in ms
        
    Returns:
        Dict[int, List[float]]: Augmented spike data
    """
    augmented = {k: v.copy() for k, v in spike_data.items()}
    
    # Add random noise spikes
    for neuron_id in augmented:
        if random.random() < noise_prob:
            noise_time = random.uniform(0, duration)
            augmented[neuron_id].append(noise_time)
            augmented[neuron_id].sort()
    
    return augmented


def shift_spikes(spike_data: Dict[int, List[float]], 
                shift_range: float = 5.0) -> Dict[int, List[float]]:
    """
    Shift all spikes by a random amount.
    
    Args:
        spike_data: Dictionary mapping neuron IDs to spike times
        shift_range: Maximum shift in ms (both positive and negative)
        
    Returns:
        Dict[int, List[float]]: Augmented spike data
    """
    shift = random.uniform(-shift_range, shift_range)
    augmented = {}
    
    for neuron_id, spike_times in spike_data.items():
        if not spike_times:
            continue
            
        # Shift all spike times by the same amount
        shifted_times = [max(0, t + shift) for t in spike_times]
        
        augmented[neuron_id] = shifted_times
    
    return augmented


def augment_spike_dataset(spike_data_list: List[Dict[int, List[float]]],
                         methods: Optional[List[str]] = None,
                         **kwargs) -> List[Dict[int, List[float]]]:
    """
    Apply multiple augmentation methods to a dataset of spike patterns.
    
    Args:
        spike_data_list: List of spike dictionaries
        methods: List of augmentation methods to apply
        **kwargs: Parameters for augmentation methods
        
    Returns:
        List[Dict[int, List[float]]]: Augmented dataset
    """
    if methods is None:
        methods = ["jitter", "drop", "noise", "shift"]
    
    augmented_dataset = []
    
    for spike_data in spike_data_list:
        # Apply a randomly selected augmentation method
        method = random.choice(methods)
        
        if method == "jitter":
            jitter_range = kwargs.get("jitter_range", 2.0)
            augmented = jitter_spikes(spike_data, jitter_range)
        elif method == "drop":
            drop_prob = kwargs.get("drop_prob", 0.1)
            augmented = drop_spikes(spike_data, drop_prob)
        elif method == "noise":
            noise_prob = kwargs.get("noise_prob", 0.05)
            duration = kwargs.get("duration", 100.0)
            augmented = add_noise_spikes(spike_data, noise_prob, duration)
        elif method == "shift":
            shift_range = kwargs.get("shift_range", 5.0)
            augmented = shift_spikes(spike_data, shift_range)
        else:
            augmented = spike_data
        
        augmented_dataset.append(augmented)
    
    return augmented_dataset
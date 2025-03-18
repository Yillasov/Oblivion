"""
Learning rules for Spiking Neural Networks.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class STDPLearningRule:
    """
    Spike-Timing-Dependent Plasticity (STDP) learning rule.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize STDP learning rule."""
        self.config = config or {}
        self.a_plus = self.config.get("a_plus", 0.01)  # LTP factor
        self.a_minus = self.config.get("a_minus", 0.0105)  # LTD factor
        self.tau_plus = self.config.get("tau_plus", 20.0)  # LTP time constant (ms)
        self.tau_minus = self.config.get("tau_minus", 20.0)  # LTD time constant (ms)
        self.w_min = self.config.get("w_min", 0.0)  # Minimum weight
        self.w_max = self.config.get("w_max", 1.0)  # Maximum weight
    
    def update_weights(self, weights: np.ndarray, pre_spikes: np.ndarray, 
                      post_spikes: np.ndarray, dt: float) -> np.ndarray:
        """
        Update weights based on STDP rule.
        
        Args:
            weights: Current weight matrix (pre_neurons, post_neurons)
            pre_spikes: Pre-synaptic spike times (pre_neurons, time_steps)
            post_spikes: Post-synaptic spike times (post_neurons, time_steps)
            dt: Time step (ms)
            
        Returns:
            Updated weight matrix
        """
        pre_neurons, time_steps = pre_spikes.shape
        post_neurons = post_spikes.shape[0]
        
        # For each pair of pre and post neurons
        for i in range(pre_neurons):
            for j in range(post_neurons):
                # Get spike times
                pre_spike_times = np.where(pre_spikes[i] > 0)[0] * dt
                post_spike_times = np.where(post_spikes[j] > 0)[0] * dt
                
                if len(pre_spike_times) == 0 or len(post_spike_times) == 0:
                    continue
                
                # Calculate weight change
                dw = 0.0
                
                # For each post-synaptic spike
                for t_post in post_spike_times:
                    # LTP: pre before post
                    for t_pre in pre_spike_times:
                        if t_pre < t_post:
                            dw += self.a_plus * np.exp(-(t_post - t_pre) / self.tau_plus)
                
                # For each pre-synaptic spike
                for t_pre in pre_spike_times:
                    # LTD: post before pre
                    for t_post in post_spike_times:
                        if t_post < t_pre:
                            dw -= self.a_minus * np.exp(-(t_pre - t_post) / self.tau_minus)
                
                # Update weight
                weights[i, j] += dw
                
                # Clip weight
                weights[i, j] = np.clip(weights[i, j], self.w_min, self.w_max)
        
        return weights


class RSTDPLearningRule:
    """
    Reward-modulated STDP learning rule.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize R-STDP learning rule."""
        self.config = config or {}
        self.stdp = STDPLearningRule(self.config)
        self.reward_factor = self.config.get("reward_factor", 1.0)
    
    def update_weights(self, weights: np.ndarray, pre_spikes: np.ndarray, 
                      post_spikes: np.ndarray, reward: float, dt: float) -> np.ndarray:
        """
        Update weights based on R-STDP rule.
        
        Args:
            weights: Current weight matrix
            pre_spikes: Pre-synaptic spike times
            post_spikes: Post-synaptic spike times
            reward: Reward signal (-1 to 1)
            dt: Time step (ms)
            
        Returns:
            Updated weight matrix
        """
        # Calculate STDP weight changes
        new_weights = self.stdp.update_weights(weights, pre_spikes, post_spikes, dt)
        
        # Modulate by reward
        weight_changes = new_weights - weights
        modulated_changes = weight_changes * reward * self.reward_factor
        
        return weights + modulated_changes
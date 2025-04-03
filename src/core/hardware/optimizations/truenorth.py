#!/usr/bin/env python3
"""
TrueNorth-specific optimization strategies.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List
from src.core.hardware.optimizations.base import HardwareOptimizer, OptimizationRegistry


class TrueNorthOptimizer(HardwareOptimizer):
    """Optimizer for IBM TrueNorth neuromorphic hardware."""
    
    def __init__(self):
        """Initialize the TrueNorth optimizer."""
        OptimizationRegistry.register("truenorth", self)
    
    def optimize_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a neural network configuration for TrueNorth hardware.
        
        Args:
            network_config: Neural network configuration
            
        Returns:
            Dict[str, Any]: Optimized network configuration
        """
        # Apply TrueNorth-specific network optimizations
        optimized_config = network_config.copy()
        
        # TrueNorth only supports binary weights
        if "connections" in optimized_config:
            for conn in optimized_config["connections"]:
                # Convert weights to binary (0 or 1)
                if "weight" in conn:
                    conn["weight"] = 1 if conn["weight"] > 0 else 0
        
        # TrueNorth only supports LIF neurons
        if "neurons" in optimized_config:
            for neuron in optimized_config["neurons"]:
                neuron["type"] = "LIF"
                # Remove unsupported parameters
                for param in list(neuron.keys()):
                    if param not in ["type", "threshold", "leak"]:
                        del neuron[param]
        
        return optimized_config
    
    def optimize_resource_allocation(self, resource_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize resource allocation for TrueNorth hardware.
        
        Args:
            resource_request: Resource allocation request
            
        Returns:
            Dict[str, Any]: Optimized resource request
        """
        optimized_request = resource_request.copy()
        
        # TrueNorth has 256 neurons per core
        if "neuron_count" in optimized_request:
            # Round up to multiple of 256 for efficient core utilization
            neuron_count = optimized_request["neuron_count"]
            optimized_request["neuron_count"] = ((neuron_count + 255) // 256) * 256
        
        # Ensure neuron parameters are compatible with TrueNorth
        if "neuron_params" in optimized_request:
            params = optimized_request["neuron_params"]
            # TrueNorth only supports LIF neurons
            params["neuron_type"] = "LIF"
            # Ensure binary weights
            params["binary_weights"] = True
            
            optimized_request["neuron_params"] = params
        
        return optimized_request
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get TrueNorth-specific optimization recommendations.
        
        Returns:
            List[str]: Optimization recommendations
        """
        return [
            "Use only LIF neuron models (TrueNorth limitation)",
            "Design networks with binary weights (0 or 1)",
            "Limit fan-in to 256 synapses per neuron",
            "Organize neurons in groups of 256 for optimal core utilization",
            "Consider stochastic firing patterns for approximating analog values"
        ]


# Create singleton instance
truenorth_optimizer = TrueNorthOptimizer()
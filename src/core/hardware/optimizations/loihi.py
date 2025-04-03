#!/usr/bin/env python3
"""
Loihi-specific optimization strategies.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List
from src.core.hardware.optimizations.base import HardwareOptimizer, OptimizationRegistry


class LoihiOptimizer(HardwareOptimizer):
    """Optimizer for Intel Loihi neuromorphic hardware."""
    
    def __init__(self):
        """Initialize the Loihi optimizer."""
        OptimizationRegistry.register("loihi", self)
    
    def optimize_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a neural network configuration for Loihi hardware.
        
        Args:
            network_config: Neural network configuration
            
        Returns:
            Dict[str, Any]: Optimized network configuration
        """
        # Apply Loihi-specific network optimizations
        optimized_config = network_config.copy()
        
        # Optimize neuron models for Loihi
        if "neurons" in optimized_config:
            for neuron in optimized_config["neurons"]:
                # Ensure neuron parameters are within Loihi's supported ranges
                if "threshold" in neuron and neuron["threshold"] < 1:
                    neuron["threshold"] = 1  # Loihi requires integer thresholds
                
                # Convert to compartment-based representation for Loihi
                neuron["compartment_type"] = self._get_compartment_type(neuron.get("type", "LIF"))
        
        return optimized_config
    
    def optimize_resource_allocation(self, resource_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize resource allocation for Loihi hardware.
        
        Args:
            resource_request: Resource allocation request
            
        Returns:
            Dict[str, Any]: Optimized resource request
        """
        optimized_request = resource_request.copy()
        
        # Optimize neuron allocation
        if "neuron_count" in optimized_request:
            # Loihi works best with neuron counts that are multiples of 4
            neuron_count = optimized_request["neuron_count"]
            optimized_request["neuron_count"] = ((neuron_count + 3) // 4) * 4
        
        # Optimize synapse allocation
        if "connections" in optimized_request:
            # Group connections by source to optimize for Loihi's axon-based routing
            connections = optimized_request["connections"]
            source_groups = {}
            
            for conn in connections:
                pre_id = conn.get("pre_id")
                if pre_id not in source_groups:
                    source_groups[pre_id] = []
                source_groups[pre_id].append(conn)
            
            # Reorder connections to group by source
            optimized_connections = []
            for pre_id, conns in source_groups.items():
                optimized_connections.extend(conns)
            
            optimized_request["connections"] = optimized_connections
        
        return optimized_request
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get Loihi-specific optimization recommendations.
        
        Returns:
            List[str]: Optimization recommendations
        """
        return [
            "Group neurons with similar parameters together for better core utilization",
            "Use integer thresholds for neurons (Loihi requirement)",
            "Limit fan-in to 4096 synapses per neuron for optimal performance",
            "Prefer sparse connectivity patterns for better resource utilization",
            "Consider using on-chip learning for adaptive applications"
        ]
    
    def _get_compartment_type(self, neuron_type: str) -> int:
        """Map neuron type to Loihi compartment type."""
        if neuron_type == "LIF":
            return 1
        elif neuron_type == "AdEx":
            return 2
        else:
            return 0  # Default compartment


# Create singleton instance
loihi_optimizer = LoihiOptimizer()
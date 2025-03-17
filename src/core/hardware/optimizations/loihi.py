"""
Loihi-specific hardware optimizations.
"""

from typing import Dict, Any, List
from src.core.hardware.optimizations.base import HardwareOptimizer, OptimizationRegistry


class LoihiOptimizer(HardwareOptimizer):
    """Optimizer for Intel Loihi neuromorphic hardware."""
    
    def optimize_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a neural network configuration for Loihi."""
        # Simple Loihi-specific optimizations
        optimized = network_config.copy()
        
        # Adjust neuron parameters for Loihi
        if "neurons" in optimized:
            for neuron in optimized["neurons"]:
                # Quantize weights to Loihi's supported precision
                if "weights" in neuron:
                    neuron["weights"] = [round(w * 256) / 256 for w in neuron["weights"]]
        
        return optimized
    
    def optimize_resource_allocation(self, resource_request: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation for Loihi."""
        optimized = resource_request.copy()
        
        # Optimize core allocation based on Loihi architecture
        if "neuron_count" in optimized:
            # Loihi has 128 neurons per core, so align to core boundaries
            cores_needed = (optimized["neuron_count"] + 127) // 128
            optimized["core_allocation"] = cores_needed
            
        return optimized
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get Loihi-specific optimization recommendations."""
        return [
            "Use sparse connectivity to maximize Loihi efficiency",
            "Group neurons by connectivity to minimize core-to-core communication",
            "Utilize Loihi's on-chip learning capabilities for adaptive networks"
        ]


# Register the optimizer
OptimizationRegistry.register("loihi", LoihiOptimizer())
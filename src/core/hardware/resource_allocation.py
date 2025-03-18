"""
Hardware-Specific Resource Allocation Strategies

Provides optimized resource allocation strategies for different neuromorphic hardware types.
"""

from typing import Dict, Any, List, Tuple, Optional
import math

from src.core.utils.logging_framework import get_logger

logger = get_logger("resource_allocation")


class ResourceAllocationStrategy:
    """Base class for hardware-specific resource allocation strategies."""
    
    def __init__(self, hardware_capabilities: Dict[str, Any]):
        """
        Initialize the resource allocation strategy.
        
        Args:
            hardware_capabilities: Hardware capabilities dictionary
        """
        self.capabilities = hardware_capabilities
        self.hardware_type = hardware_capabilities.get("hardware_type", "unknown")
    
    def allocate_neurons(self, count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate neurons based on hardware-specific strategy.
        
        Args:
            count: Number of neurons to allocate
            params: Neuron parameters
            
        Returns:
            Dict[str, Any]: Allocation result
        """
        raise NotImplementedError("Subclasses must implement allocate_neurons")
    
    def allocate_synapses(self, connections: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        """
        Allocate synapses based on hardware-specific strategy.
        
        Args:
            connections: List of (pre_id, post_id, weight) tuples
            
        Returns:
            Dict[str, Any]: Allocation result
        """
        raise NotImplementedError("Subclasses must implement allocate_synapses")
    
    def optimize_placement(self, neuron_groups: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Optimize neuron placement based on hardware-specific strategy.
        
        Args:
            neuron_groups: Dictionary of neuron group name to list of neuron IDs
            
        Returns:
            Dict[str, Any]: Optimized placement
        """
        raise NotImplementedError("Subclasses must implement optimize_placement")


class LoihiAllocationStrategy(ResourceAllocationStrategy):
    """Resource allocation strategy for Intel Loihi hardware."""
    
    def allocate_neurons(self, count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate neurons optimized for Loihi architecture."""
        neurons_per_core = self.capabilities.get("neurons_per_core", 1024)
        cores_per_chip = self.capabilities.get("cores_per_chip", 128)
        
        # Loihi works best when neurons with similar parameters are grouped together
        neuron_type = params.get("neuron_type", "LIF")
        
        # Calculate cores needed
        cores_needed = math.ceil(count / neurons_per_core)
        chips_needed = math.ceil(cores_needed / cores_per_chip)
        
        logger.info(f"Allocating {count} {neuron_type} neurons on Loihi ({cores_needed} cores, {chips_needed} chips)")
        
        # Loihi-specific optimization: group by compartment type
        compartment_type = 0  # Default compartment
        if neuron_type == "LIF":
            compartment_type = 1
        elif neuron_type == "AdEx":
            compartment_type = 2
        
        return {
            "allocated_neurons": count,
            "cores_used": cores_needed,
            "chips_used": chips_needed,
            "compartment_type": compartment_type,
            "neuron_params": params,
            "allocation_strategy": "loihi_optimized"
        }
    
    def allocate_synapses(self, connections: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        """Allocate synapses optimized for Loihi architecture."""
        # Loihi has specific synapse formats and weight representations
        # Group connections by source neuron to optimize for Loihi's axon-based routing
        source_groups = {}
        for pre_id, post_id, weight in connections:
            if pre_id not in source_groups:
                source_groups[pre_id] = []
            source_groups[pre_id].append((post_id, weight))
        
        # Loihi uses 8-bit weights, so we need to scale and quantize
        weight_scale = 1.0
        if connections:
            max_weight = max(abs(w) for _, _, w in connections)
            if max_weight > 0:
                weight_scale = 127.0 / max_weight
        
        return {
            "allocated_synapses": len(connections),
            "source_groups": len(source_groups),
            "weight_scale": weight_scale,
            "weight_precision": "8-bit",
            "allocation_strategy": "loihi_optimized"
        }
    
    def optimize_placement(self, neuron_groups: Dict[str, List[int]]) -> Dict[str, Any]:
        """Optimize neuron placement for Loihi architecture."""
        # Loihi works best when connected neurons are placed on the same or adjacent cores
        cores_per_chip = self.capabilities.get("cores_per_chip", 128)
        neurons_per_core = self.capabilities.get("neurons_per_core", 1024)
        
        # Simple placement strategy: try to keep neuron groups on the same chip
        placement = {}
        current_chip = 0
        current_core = 0
        neurons_in_core = 0
        
        for group_name, neuron_ids in neuron_groups.items():
            group_placement = []
            
            for neuron_id in neuron_ids:
                if neurons_in_core >= neurons_per_core:
                    # Move to next core
                    current_core += 1
                    neurons_in_core = 0
                    
                    if current_core >= cores_per_chip:
                        # Move to next chip
                        current_chip += 1
                        current_core = 0
                
                # Place neuron
                placement[neuron_id] = (current_chip, current_core, neurons_in_core)
                group_placement.append((current_chip, current_core, neurons_in_core))
                neurons_in_core += 1
            
            logger.info(f"Placed neuron group '{group_name}' with {len(neuron_ids)} neurons")
        
        return {
            "placement": placement,
            "chips_used": current_chip + 1,
            "cores_used": current_core + 1 if current_chip == 0 else cores_per_chip,
            "allocation_strategy": "loihi_optimized"
        }


class TrueNorthAllocationStrategy(ResourceAllocationStrategy):
    """Resource allocation strategy for IBM TrueNorth hardware."""
    
    def allocate_neurons(self, count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate neurons optimized for TrueNorth architecture."""
        neurons_per_core = self.capabilities.get("neurons_per_core", 256)
        cores_per_chip = self.capabilities.get("cores_per_chip", 4096)
        
        # TrueNorth has fixed 256 neurons per core
        cores_needed = math.ceil(count / neurons_per_core)
        chips_needed = math.ceil(cores_needed / cores_per_chip)
        
        logger.info(f"Allocating {count} neurons on TrueNorth ({cores_needed} cores, {chips_needed} chips)")
        
        # TrueNorth only supports simple LIF neurons with binary weights
        return {
            "allocated_neurons": count,
            "cores_used": cores_needed,
            "chips_used": chips_needed,
            "neuron_params": {"type": "LIF", "binary_weights": True},
            "allocation_strategy": "truenorth_optimized"
        }
    
    def allocate_synapses(self, connections: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        """Allocate synapses optimized for TrueNorth architecture."""
        # TrueNorth uses binary weights (0 or 1)
        binary_connections = [(pre, post, 1 if weight > 0 else 0) for pre, post, weight in connections]
        
        # Count active connections (weight > 0)
        active_connections = sum(1 for _, _, w in binary_connections if w > 0)
        
        return {
            "allocated_synapses": len(connections),
            "active_synapses": active_connections,
            "weight_precision": "1-bit",
            "allocation_strategy": "truenorth_optimized"
        }
    
    def optimize_placement(self, neuron_groups: Dict[str, List[int]]) -> Dict[str, Any]:
        """Optimize neuron placement for TrueNorth architecture."""
        # TrueNorth has a fixed crossbar architecture with 256 neurons per core
        neurons_per_core = self.capabilities.get("neurons_per_core", 256)
        
        # Simple placement strategy: fill cores sequentially
        placement = {}
        current_core = 0
        neurons_in_core = 0
        
        for group_name, neuron_ids in neuron_groups.items():
            for neuron_id in neuron_ids:
                if neurons_in_core >= neurons_per_core:
                    # Move to next core
                    current_core += 1
                    neurons_in_core = 0
                
                # Place neuron
                placement[neuron_id] = (current_core, neurons_in_core)
                neurons_in_core += 1
        
        return {
            "placement": placement,
            "cores_used": current_core + 1,
            "allocation_strategy": "truenorth_optimized"
        }


class SpiNNakerAllocationStrategy(ResourceAllocationStrategy):
    """Resource allocation strategy for SpiNNaker hardware."""
    
    def allocate_neurons(self, count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate neurons optimized for SpiNNaker architecture."""
        neurons_per_core = self.capabilities.get("neurons_per_core", 1000)
        cores_per_chip = self.capabilities.get("cores_per_chip", 18)
        
        # SpiNNaker allocation depends on neuron type and complexity
        neuron_type = params.get("neuron_type", "LIF")
        
        # Adjust neurons per core based on model complexity
        if neuron_type == "Izhikevich":
            neurons_per_core = neurons_per_core // 2  # More complex model
        elif neuron_type == "Custom":
            neurons_per_core = neurons_per_core // 3  # Custom models are more resource-intensive
        
        # Calculate cores needed
        cores_needed = math.ceil(count / neurons_per_core)
        chips_needed = math.ceil(cores_needed / cores_per_chip)
        
        logger.info(f"Allocating {count} {neuron_type} neurons on SpiNNaker ({cores_needed} cores, {chips_needed} chips)")
        
        return {
            "allocated_neurons": count,
            "cores_used": cores_needed,
            "chips_used": chips_needed,
            "neurons_per_core": neurons_per_core,
            "neuron_params": params,
            "allocation_strategy": "spinnaker_optimized"
        }
    
    def allocate_synapses(self, connections: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        """Allocate synapses optimized for SpiNNaker architecture."""
        # SpiNNaker uses SDRAM to store synaptic connectivity
        # Group connections by target neuron for SpiNNaker's routing system
        target_groups = {}
        for pre_id, post_id, weight in connections:
            if post_id not in target_groups:
                target_groups[post_id] = []
            target_groups[post_id].append((pre_id, weight))
        
        # Estimate memory usage (16 bytes per synapse is a rough estimate)
        memory_usage = len(connections) * 16
        
        return {
            "allocated_synapses": len(connections),
            "target_groups": len(target_groups),
            "estimated_memory_bytes": memory_usage,
            "weight_precision": "16-bit",
            "allocation_strategy": "spinnaker_optimized"
        }
    
    def optimize_placement(self, neuron_groups: Dict[str, List[int]]) -> Dict[str, Any]:
        """Optimize neuron placement for SpiNNaker architecture."""
        # SpiNNaker benefits from placing highly connected neurons on the same chip
        cores_per_chip = self.capabilities.get("cores_per_chip", 18)
        neurons_per_core = self.capabilities.get("neurons_per_core", 1000)
        
        # Simple placement strategy: try to keep neuron groups on the same chip
        placement = {}
        current_chip = 0
        current_core = 0
        neurons_in_core = 0
        
        for group_name, neuron_ids in neuron_groups.items():
            # Try to keep each group on a single chip if possible
            if current_core >= cores_per_chip:
                current_chip += 1
                current_core = 0
                neurons_in_core = 0
            
            for neuron_id in neuron_ids:
                if neurons_in_core >= neurons_per_core:
                    # Move to next core
                    current_core += 1
                    neurons_in_core = 0
                    
                    if current_core >= cores_per_chip:
                        # Move to next chip
                        current_chip += 1
                        current_core = 0
                
                # Place neuron
                placement[neuron_id] = (current_chip, current_core, neurons_in_core)
                neurons_in_core += 1
        
        return {
            "placement": placement,
            "chips_used": current_chip + 1,
            "cores_used": current_core + 1 if current_chip == 0 else cores_per_chip,
            "allocation_strategy": "spinnaker_optimized"
        }


class ResourceAllocator:
    """Factory class for creating hardware-specific allocation strategies."""
    
    @staticmethod
    def create_strategy(hardware_capabilities: Dict[str, Any]) -> ResourceAllocationStrategy:
        """
        Create a hardware-specific allocation strategy.
        
        Args:
            hardware_capabilities: Hardware capabilities dictionary
            
        Returns:
            ResourceAllocationStrategy: Hardware-specific allocation strategy
        """
        hw_type = hardware_capabilities.get("hardware_type", "unknown").lower()
        
        if "loihi" in hw_type:
            return LoihiAllocationStrategy(hardware_capabilities)
        elif "truenorth" in hw_type:
            return TrueNorthAllocationStrategy(hardware_capabilities)
        elif "spinnaker" in hw_type:
            return SpiNNakerAllocationStrategy(hardware_capabilities)
        else:
            # Default to Loihi strategy as fallback
            logger.warning(f"No specific allocation strategy for {hw_type}, using default")
            return LoihiAllocationStrategy(hardware_capabilities)
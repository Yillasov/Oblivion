#!/usr/bin/env python3
"""
TrueNorth Hardware Abstraction Implementation

Provides a hardware abstraction implementation for IBM TrueNorth neuromorphic hardware.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Tuple, Callable
import logging

from src.core.utils.logging_framework import get_logger
from src.core.hardware.exceptions import (
    HardwareInitializationError,
    HardwareAllocationError,
    HardwareSimulationError,
    HardwareCommunicationError,
    UnsupportedFeatureError,
    NeuromorphicHardwareError
)
from src.core.hardware.hardware_abstraction import NeuromorphicHardware
from src.core.hardware.truenorth_driver import TrueNorthProcessor

logger = get_logger("truenorth_hardware")


class TrueNorthHardware(NeuromorphicHardware):
    """Hardware abstraction implementation for IBM TrueNorth."""
    
    def __init__(self):
        """Initialize the TrueNorth hardware abstraction."""
        super().__init__()
        self.driver = TrueNorthProcessor()
        self.neuron_mapping = {}  # Maps logical neuron IDs to hardware neuron IDs
        self.core_mapping = {}    # Maps core IDs to allocated neurons
        self.channel_mapping = {} # Maps channel IDs to synapse IDs
        self.memory_regions = {}  # Maps memory block IDs to their properties
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the TrueNorth hardware with the given configuration."""
        with self.error_context("hardware initialization"):
            if config is None:
                config = {}
                
            result = self.driver.initialize(config)
            self.initialized = result
            
            if result:
                # Get hardware capabilities
                hw_info = self.driver.get_hardware_info()
                self.hardware_capabilities = {
                    "cores_available": hw_info.get("cores_total", 4096),
                    "neurons_per_core": hw_info.get("neurons_per_core", 256),
                    "max_synapses": 4096 * 256 * 256,  # Each neuron can connect to all neurons in a core
                    "supports_learning": False,  # TrueNorth doesn't support on-chip learning
                    "processor_type": "IBM TrueNorth"
                }
            
            return result
    
    def shutdown(self) -> bool:
        """Safely shutdown the hardware."""
        with self.error_context("hardware shutdown"):
            result = self.driver.shutdown()
            if result:
                self.initialized = False
                self.neuron_mapping = {}
                self.core_mapping = {}
                self.channel_mapping = {}
                self.resource_usage = {
                    "neurons": 0,
                    "synapses": 0,
                    "memory": 0,
                    "compute_units": 0
                }
            return result
    
    def allocate_resources(self, resource_request: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate hardware resources based on request."""
        with self.error_context("resource allocation"):
            # Extract resource requirements
            neuron_count = resource_request.get("neuron_count", 0)
            neuron_params = resource_request.get("neuron_params", {})
            
            # TrueNorth has 256 neurons per core
            cores_needed = (neuron_count + 255) // 256
            
            # Allocate neurons
            if neuron_count > 0:
                neuron_ids = self.driver.allocate_neurons(neuron_count, neuron_params)
                for i, hw_id in enumerate(neuron_ids):
                    logical_id = i + len(self.neuron_mapping)
                    self.neuron_mapping[logical_id] = hw_id
                
                # Update resource usage
                self.resource_usage["neurons"] += neuron_count
                self.resource_usage["compute_units"] += cores_needed
            
            return {
                "allocated_neurons": neuron_count,
                "allocated_cores": cores_needed,
                "logical_ids": list(range(len(self.neuron_mapping) - neuron_count, len(self.neuron_mapping))),
                "status": "success"
            }
    
    def run_computation(self, inputs: Dict[str, Any], duration_ms: float) -> Dict[str, Any]:
        """Run a computation on the hardware with the given inputs."""
        with self.error_context("computation"):
            # Set input spikes if provided
            if "spikes" in inputs:
                spike_inputs = []
                spike_times = []
                
                for logical_id, times in inputs["spikes"].items():
                    if logical_id in self.neuron_mapping:
                        hw_id = self.neuron_mapping[logical_id]
                        spike_inputs.append(hw_id)
                        spike_times.append(times)
                
                self.driver.set_spike_input(spike_inputs, spike_times)
            
            # Run simulation - TrueNorth uses discrete time steps
            time_steps = int(duration_ms)
            self.driver.run_simulation(time_steps)
            
            # Get output spikes
            hw_neuron_ids = list(self.neuron_mapping.values())
            spike_outputs = self.driver.get_spike_output(hw_neuron_ids)
            
            # Convert hardware IDs back to logical IDs
            logical_outputs = {}
            for logical_id, hw_id in self.neuron_mapping.items():
                if hw_id in spike_outputs:
                    logical_outputs[logical_id] = spike_outputs[hw_id]
            
            return {
                "spikes": logical_outputs,
                "duration_ms": duration_ms
            }
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about the hardware."""
        with self.error_context("hardware info"):
            hw_info = self.driver.get_hardware_info()
            hw_info.update({
                "allocated_neurons": self.resource_usage["neurons"],
                "allocated_synapses": self.resource_usage["synapses"],
                "allocated_cores": self.resource_usage["compute_units"]
            })
            return hw_info
    
    def update_synaptic_weights(self, connections: List[Tuple[int, int, float]]) -> bool:
        """Update synaptic weights between neurons."""
        with self.error_context("weight update"):
            # Convert logical IDs to hardware IDs
            hw_connections = []
            for pre_id, post_id, weight in connections:
                if pre_id in self.neuron_mapping and post_id in self.neuron_mapping:
                    hw_pre = self.neuron_mapping[pre_id]
                    hw_post = self.neuron_mapping[post_id]
                    # TrueNorth uses binary weights, so we binarize them
                    binary_weight = 1 if weight > 0 else 0
                    hw_connections.append((hw_pre, hw_post, binary_weight))
            
            # Create synapses on hardware
            if hw_connections:
                synapse_ids = self.driver.create_synapses(hw_connections)
                self.resource_usage["synapses"] += len(synapse_ids)
                return True
            return False
    
    def reset_state(self) -> bool:
        """Reset the hardware state without full reinitialization."""
        with self.error_context("state reset"):
            # Check if the driver has the reset_state method
            if hasattr(self.driver, 'reset_state'):
                return self.driver.reset_state()
            else:
                # Fallback implementation if the driver doesn't have reset_state
                logger.warning("TrueNorth driver does not implement reset_state, using fallback")
                # Clear spike data
                self._spike_data = {}
                # Reset neuron states but keep connections
                return True
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        return self.resource_usage
    
    def check_compatibility(self, model_requirements: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if a neural model is compatible with this hardware."""
        compatible = True
        details = {}
        
        # Check neuron count
        if "neuron_count" in model_requirements:
            req_neurons = model_requirements["neuron_count"]
            max_neurons = self.hardware_capabilities.get("cores_available", 4096) * self.hardware_capabilities.get("neurons_per_core", 256)
            if req_neurons > max_neurons:
                compatible = False
                details["neuron_count"] = f"Required: {req_neurons}, Available: {max_neurons}"
        
        # Check synapse count
        if "synapse_count" in model_requirements:
            req_synapses = model_requirements["synapse_count"]
            max_synapses = self.hardware_capabilities.get("max_synapses", 4096 * 256 * 256)
            if req_synapses > max_synapses:
                compatible = False
                details["synapse_count"] = f"Required: {req_synapses}, Available: {max_synapses}"
        
        # Check if learning is required (TrueNorth doesn't support on-chip learning)
        if model_requirements.get("requires_learning", False):
            compatible = False
            details["learning"] = "TrueNorth does not support on-chip learning"
        
        # Check if non-binary weights are required
        if model_requirements.get("non_binary_weights", False):
            compatible = False
            details["weights"] = "TrueNorth only supports binary weights"
        
        return compatible, details
    
    def configure_monitoring(self, monitoring_config: Dict[str, Any]) -> bool:
        """Configure hardware monitoring parameters."""
        with self.error_context("monitoring configuration"):
            # TrueNorth has limited monitoring capabilities
            return True
    
    def allocate_memory(self, size: int, memory_type: str) -> Optional[int]:
        """Allocate memory of specified size and type."""
        with self.error_context("memory allocation"):
            # TrueNorth has limited memory types
            if memory_type not in ["core", "chip"]:
                memory_type = "core"  # Default to core memory
                
            # Simulate memory allocation
            block_id = len(self.memory_regions)
            self.memory_regions[block_id] = {"size": size, "type": memory_type}
            self.resource_usage["memory"] += size
            return block_id
    
    def free_memory(self, block_id: int) -> bool:
        """Free previously allocated memory."""
        with self.error_context("memory deallocation"):
            if block_id in self.memory_regions:
                self.resource_usage["memory"] -= self.memory_regions[block_id]["size"]
                del self.memory_regions[block_id]
                return True
            return False
    
    def allocate_compute_units(self, count: int, unit_type: str) -> List[int]:
        """Allocate compute units of specified type."""
        with self.error_context("compute unit allocation"):
            # In TrueNorth, compute units are cores
            core_ids = []
            for i in range(count):
                core_id = len(self.compute_units)
                self.compute_units[core_id] = {"type": unit_type, "allocated": True}
                core_ids.append(core_id)
            
            self.resource_usage["compute_units"] += count
            return core_ids
    
    def free_compute_units(self, unit_ids: List[int]) -> bool:
        """Free previously allocated compute units."""
        with self.error_context("compute unit deallocation"):
            success = True
            for unit_id in unit_ids:
                if unit_id in self.compute_units:
                    del self.compute_units[unit_id]
                    self.resource_usage["compute_units"] -= 1
                else:
                    success = False
            return success
    
    def create_communication_channel(self, source_id: int, target_id: int, channel_type: str) -> Optional[int]:
        """Create a communication channel between compute units."""
        with self.error_context("channel creation"):
            # TrueNorth uses axons for communication
            if source_id in self.neuron_mapping and target_id in self.neuron_mapping:
                hw_source = self.neuron_mapping[source_id]
                hw_target = self.neuron_mapping[target_id]
                
                # Create a synapse with weight 1 to establish the channel
                synapse_ids = self.driver.create_synapses([(hw_source, hw_target, 1)])
                if synapse_ids:
                    channel_id = len(self.channel_mapping)
                    self.channel_mapping[channel_id] = synapse_ids[0]
                    return channel_id
            return None
    
    def close_communication_channel(self, channel_id: int) -> bool:
        """Close a communication channel."""
        with self.error_context("channel closure"):
            if channel_id in self.channel_mapping:
                # In a real implementation, this would remove the synapse
                del self.channel_mapping[channel_id]
                return True
            return False
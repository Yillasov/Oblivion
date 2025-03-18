"""
Neuromorphic Hardware Abstraction Layer

Provides a unified interface for interacting with different 
neuromorphic hardware platforms, abstracting away hardware-specific details.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable, ContextManager
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager

# Import the correct logger types
from src.core.utils.logging_framework import get_logger, ErrorHandler, neuromorphic_logger
from src.core.hardware.exceptions import (
    HardwareInitializationError,
    HardwareAllocationError,
    HardwareSimulationError,
    HardwareCommunicationError,
    UnsupportedFeatureError,
    NeuromorphicHardwareError
)
# Import the LoihiProcessor class
from src.core.hardware.loihi_driver import LoihiProcessor
# Import the SpiNNakerProcessor class
from src.core.hardware.spinnaker_driver import SpiNNakerProcessor
# Add this import at the top of the file
from src.core.hardware.truenorth_hardware import TrueNorthHardware
# Add this import at the top of the file
from src.core.hardware.simulated_hardware import SimulatedHardware
# Add this import at the top of the file
from src.core.hardware.communication_protocols import ProtocolFactory
# Add this import at the top of the file
from src.core.hardware.data_converters import FormatConverterFactory, DataFormatConverter
# Add this import at the top of the file
from src.core.hardware.data_transfer import DataTransfer, TransferMode
# Add this import at the top of the file
from src.core.hardware.resource_sharing import (
    get_resource_manager, ResourceType, ResourceSharingError
)

logger = get_logger("hardware_abstraction")


class NeuromorphicHardware(ABC):
    """
    Abstract base class defining the hardware abstraction interface.
    All hardware-specific implementations should inherit from this class.
    """
    
    def __init__(self):
        """Initialize the hardware abstraction."""
        self.error_handler = ErrorHandler(neuromorphic_logger)
        self.initialized = False
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.hardware_capabilities = {}
        self.resource_usage = {
            "neurons": 0,
            "synapses": 0,
            "memory": 0,
            "compute_units": 0
        }
        # Resource management tracking
        self.memory_regions = {}
        self.compute_units = {}
        self.communication_channels = {}
    
    # Standard error handling methods
    def handle_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Any:
        """
        Standard method to handle any hardware operation with consistent error handling.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the operation function
            
        Returns:
            Any: Result of the operation function
            
        Raises:
            NeuromorphicHardwareError: If operation fails
        """
        with self.error_context(operation_name):
            return operation_func(*args, **kwargs)
    
    # Memory management interface
    @abstractmethod
    def allocate_memory(self, size: int, memory_type: str) -> Optional[int]:
        """
        Allocate memory of specified size and type.
        
        Args:
            size: Size in bytes
            memory_type: Type of memory ('core', 'shared', etc.)
            
        Returns:
            Optional[int]: Memory block ID or None if allocation failed
        """
        pass
    
    @abstractmethod
    def free_memory(self, block_id: int) -> bool:
        """
        Free previously allocated memory.
        
        Args:
            block_id: Memory block ID to free
            
        Returns:
            bool: True if memory was freed successfully
        """
        pass
    
    # Compute unit management interface
    @abstractmethod
    def allocate_compute_units(self, count: int, unit_type: str) -> List[int]:
        """
        Allocate compute units of specified type.
        
        Args:
            count: Number of compute units to allocate
            unit_type: Type of compute units ('neuron', 'core', etc.)
            
        Returns:
            List[int]: IDs of allocated compute units
        """
        pass
    
    @abstractmethod
    def free_compute_units(self, unit_ids: List[int]) -> bool:
        """
        Free previously allocated compute units.
        
        Args:
            unit_ids: List of compute unit IDs to free
            
        Returns:
            bool: True if all units were freed successfully
        """
        pass
    
    # Communication channel management interface
    @abstractmethod
    def create_communication_channel(self, source_id: int, target_id: int, 
                                    channel_type: str) -> Optional[int]:
        """
        Create a communication channel between compute units.
        
        Args:
            source_id: Source compute unit ID
            target_id: Target compute unit ID
            channel_type: Type of channel ('spike', 'rate', etc.)
            
        Returns:
            Optional[int]: Channel ID or None if creation failed
        """
        pass
    
    @abstractmethod
    def close_communication_channel(self, channel_id: int) -> bool:
        """
        Close a communication channel.
        
        Args:
            channel_id: Channel ID to close
            
        Returns:
            bool: True if channel was closed successfully
        """
        pass
    
    def handle_initialization_error(self, error: Exception) -> None:
        """Handle initialization errors consistently."""
        logger.error(f"Hardware initialization error: {str(error)}")
        self.initialized = False
        raise HardwareInitializationError(f"Failed to initialize hardware: {str(error)}")
    
    def handle_allocation_error(self, error: Exception, resource_type: str) -> None:
        """Handle resource allocation errors consistently."""
        logger.error(f"Hardware allocation error for {resource_type}: {str(error)}")
        raise HardwareAllocationError(f"Failed to allocate {resource_type}: {str(error)}")
    
    def handle_communication_error(self, error: Exception, operation: str) -> None:
        """Handle communication errors consistently."""
        logger.error(f"Hardware communication error during {operation}: {str(error)}")
        raise HardwareCommunicationError(f"Communication failed during {operation}: {str(error)}")
    
    def handle_simulation_error(self, error: Exception) -> None:
        """Handle simulation errors consistently."""
        logger.error(f"Hardware simulation error: {str(error)}")
        raise HardwareSimulationError(f"Simulation failed: {str(error)}")
    
    def handle_unsupported_feature(self, feature: str) -> None:
        """Handle unsupported feature errors consistently."""
        logger.error(f"Unsupported hardware feature: {feature}")
        raise UnsupportedFeatureError(f"Feature not supported: {feature}")
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the hardware with the given configuration."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Safely shutdown the hardware."""
        pass
    
    @abstractmethod
    def allocate_resources(self, resource_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate hardware resources based on request.
        
        Args:
            resource_request: Dictionary specifying required resources
            
        Returns:
            Dict[str, Any]: Allocated resource information
        """
        pass
    
    @abstractmethod
    def run_computation(self, inputs: Dict[str, Any], duration_ms: float) -> Dict[str, Any]:
        """
        Run a computation on the hardware with the given inputs.
        
        Args:
            inputs: Input data for computation
            duration_ms: Duration to run in milliseconds
            
        Returns:
            Dict[str, Any]: Computation results
        """
        pass
    
    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get information about the hardware.
        
        Returns:
            Dict[str, Any]: Hardware information
        """
        pass
    
    @abstractmethod
    def update_synaptic_weights(self, connections: List[Tuple[int, int, float]]) -> bool:
        """
        Update synaptic weights between neurons.
        
        Args:
            connections: List of (pre_neuron_id, post_neuron_id, weight) tuples
            
        Returns:
            bool: True if weights were updated successfully
        """
        pass
    
    @abstractmethod
    def reset_state(self) -> bool:
        """
        Reset the hardware state without full reinitialization.
        
        Returns:
            bool: True if reset was successful
        """
        pass
    
    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage statistics.
        
        Returns:
            Dict[str, Any]: Resource usage information
        """
        pass
    
    @abstractmethod
    def check_compatibility(self, model_requirements: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a neural model is compatible with this hardware.
        
        Args:
            model_requirements: Requirements of the neural model
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_compatible, compatibility_details)
        """
        pass
    
    @abstractmethod
    def configure_monitoring(self, monitoring_config: Dict[str, Any]) -> bool:
        """
        Configure hardware monitoring parameters.
        
        Args:
            monitoring_config: Monitoring configuration
            
        Returns:
            bool: True if monitoring was configured successfully
        """
        pass
    
    @contextmanager
    def error_context(self, operation_name: str, recovery_func: Optional[Callable] = None):
        """
        Context manager for hardware operations with error handling.
        
        Args:
            operation_name: Name of the operation being performed
            recovery_func: Optional function to call for recovery
        """
        try:
            yield
        except NeuromorphicHardwareError as e:
            # Log the specific hardware error
            logger.error(f"Hardware error during {operation_name}: {str(e)}")
            
            # Attempt recovery if function provided and not too many attempts
            if recovery_func and self.recovery_attempts < self.max_recovery_attempts:
                logger.info(f"Attempting recovery for {operation_name}")
                self.recovery_attempts += 1
                try:
                    recovery_func()
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {str(recovery_error)}")
            
            # Re-raise the original error
            raise
        except Exception as e:
            # Wrap unknown errors in a hardware error
            logger.error(f"Unexpected error during {operation_name}: {str(e)}")
            raise HardwareCommunicationError(f"Unexpected error: {str(e)}")
        finally:
            # Reset recovery counter on success
            if recovery_func:
                self.recovery_attempts = 0
    
    def is_initialized(self) -> bool:
        """
        Check if hardware is initialized.
        
        Returns:
            bool: True if hardware is initialized
        """
        return self.initialized
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get hardware capabilities.
        
        Returns:
            Dict[str, Any]: Hardware capabilities
        """
        return self.hardware_capabilities


class LoihiHardware(NeuromorphicHardware):
    """Hardware abstraction implementation for Intel Loihi."""
    
    def __init__(self):
        """Initialize the Loihi hardware abstraction."""
        super().__init__()
        self.driver = LoihiProcessor()
        self.protocol = ProtocolFactory.create_protocol("loihi")
        self.neuron_mapping = {}  # Maps logical neuron IDs to hardware neuron IDs
        self.core_mapping = {}    # Maps core IDs to allocated neurons
        self.channel_mapping = {} # Maps channel IDs to synapse IDs
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the Loihi hardware with the given configuration."""
        with self.error_context("hardware initialization"):
            if config is None:
                config = {}
                
            result = self.driver.initialize(config)
            self.initialized = result
            
            if result:
                # Get hardware capabilities
                hw_info = self.driver.get_hardware_info()
                self.hardware_capabilities = {
                    "cores_available": hw_info.get("cores_available", 128),
                    "neurons_per_core": hw_info.get("neurons_per_core", 1024),
                    "max_synapses": hw_info.get("max_synapses", 1000000),
                    "supports_learning": True,
                    "processor_type": "Intel Loihi"
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
            
            # Allocate neurons
            if neuron_count > 0:
                neuron_ids = self.driver.allocate_neurons(neuron_count, neuron_params)
                for i, hw_id in enumerate(neuron_ids):
                    logical_id = i + len(self.neuron_mapping)
                    self.neuron_mapping[logical_id] = hw_id
                
                # Update resource usage
                self.resource_usage["neurons"] += neuron_count
                
                # Calculate core usage based on Loihi's architecture (128 neurons per core)
                cores_used = (neuron_count + 127) // 128
                self.resource_usage["compute_units"] += cores_used
            
            return {
                "allocated_neurons": neuron_count,
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
            
            # Run simulation
            self.driver.run_simulation(duration_ms)
            
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
                    hw_connections.append((hw_pre, hw_post, weight))
            
            # Create synapses on hardware
            if hw_connections:
                synapse_ids = self.driver.create_synapses(hw_connections)
                self.resource_usage["synapses"] += len(synapse_ids)
                return True
            return False
    
    def reset_state(self) -> bool:
        """Reset the hardware state without full reinitialization."""
        with self.error_context("state reset"):
            # In a real implementation, this would reset neuron states without deallocating
            # For this simple implementation, we'll just clear spike data
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
            max_neurons = self.hardware_capabilities.get("cores_available", 128) * self.hardware_capabilities.get("neurons_per_core", 1024)
            if req_neurons > max_neurons:
                compatible = False
                details["neuron_count"] = f"Required: {req_neurons}, Available: {max_neurons}"
        
        # Check synapse count
        if "synapse_count" in model_requirements:
            req_synapses = model_requirements["synapse_count"]
            max_synapses = self.hardware_capabilities.get("max_synapses", 1000000)
            if req_synapses > max_synapses:
                compatible = False
                details["synapse_count"] = f"Required: {req_synapses}, Available: {max_synapses}"
        
        return compatible, details
    
    def configure_monitoring(self, monitoring_config: Dict[str, Any]) -> bool:
        """Configure hardware monitoring parameters."""
        # Simple implementation - in a real system, this would configure hardware monitoring
        return True
    
    def allocate_memory(self, size: int, memory_type: str) -> Optional[int]:
        """Allocate memory of specified size and type."""
        with self.error_context("memory allocation"):
            # Simple implementation - track memory usage
            self.resource_usage["memory"] += size
            return len(self.memory_regions)  # Return a simple block ID
    
    def free_memory(self, block_id: int) -> bool:
        """Free previously allocated memory."""
        with self.error_context("memory deallocation"):
            # Simple implementation
            return True
    
    def allocate_compute_units(self, count: int, unit_type: str) -> List[int]:
        """Allocate compute units of specified type."""
        with self.error_context("compute unit allocation"):
            # Simple implementation - allocate core IDs
            start_id = self.resource_usage["compute_units"]
            self.resource_usage["compute_units"] += count
            return list(range(start_id, start_id + count))
    
    def free_compute_units(self, unit_ids: List[int]) -> bool:
        """Free previously allocated compute units."""
        with self.error_context("compute unit deallocation"):
            # Simple implementation
            return True
    
    def create_communication_channel(self, source_id: int, target_id: int, channel_type: str) -> Optional[int]:
        """Create a communication channel between compute units."""
        with self.error_context("channel creation"):
            # In Loihi, communication channels are implemented as synapses
            if source_id in self.neuron_mapping and target_id in self.neuron_mapping:
                hw_source = self.neuron_mapping[source_id]
                hw_target = self.neuron_mapping[target_id]
                
                # Default weight for the channel
                weight = 1.0
                
                # Create a synapse for this channel
                synapse_ids = self.driver.create_synapses([(hw_source, hw_target, weight)])
                if synapse_ids:
                    channel_id = len(self.channel_mapping)
                    self.channel_mapping[channel_id] = synapse_ids[0]
                    return channel_id
            return None
    
    def close_communication_channel(self, channel_id: int) -> bool:
        """Close a communication channel."""
        with self.error_context("channel closure"):
            # Simple implementation
            if channel_id in self.channel_mapping:
                del self.channel_mapping[channel_id]
                return True
            return False

# Add this class after the LoihiHardware class

class SpiNNakerHardware(NeuromorphicHardware):
    """Hardware abstraction implementation for SpiNNaker."""
    
    def __init__(self):
        """Initialize the SpiNNaker hardware abstraction."""
        super().__init__()
        self.driver = SpiNNakerProcessor()
        self.neuron_mapping = {}  # Maps logical neuron IDs to hardware neuron IDs
        self.core_mapping = {}    # Maps core IDs to allocated neurons
        self.channel_mapping = {} # Maps channel IDs to connection IDs
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the SpiNNaker hardware with the given configuration."""
        with self.error_context("hardware initialization"):
            if config is None:
                config = {}
                
            result = self.driver.initialize(config)
            self.initialized = result
            
            if result:
                # Get hardware capabilities
                hw_info = self.driver.get_hardware_info()
                self.hardware_capabilities = {
                    "cores_available": hw_info.get("cores_available", 48),
                    "neurons_per_core": hw_info.get("neurons_per_core", 1000),
                    "max_synapses": hw_info.get("max_synapses", 16000000),
                    "supports_learning": True,
                    "processor_type": "SpiNNaker"
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
            
            # Apply SpiNNaker-specific optimizations
            # Each ARM core can handle about 1000 neurons efficiently
            cores_needed = (neuron_count + 999) // 1000
            
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
            
            # Run simulation - SpiNNaker uses 1ms timesteps by default
            timesteps = int(duration_ms)
            self.driver.run_simulation(timesteps)
            
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
                    hw_connections.append((hw_pre, hw_post, weight))
            
            # Create synapses on hardware
            if hw_connections:
                synapse_ids = self.driver.create_synapses(hw_connections)
                self.resource_usage["synapses"] += len(synapse_ids)
                return True
            return False
    
    def reset_state(self) -> bool:
        """Reset the hardware state without full reinitialization."""
        with self.error_context("state reset"):
            return self.driver.reset_state()
    
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
            max_neurons = self.hardware_capabilities.get("cores_available", 48) * self.hardware_capabilities.get("neurons_per_core", 1000)
            if req_neurons > max_neurons:
                compatible = False
                details["neuron_count"] = f"Required: {req_neurons}, Available: {max_neurons}"
        
        # Check synapse count
        if "synapse_count" in model_requirements:
            req_synapses = model_requirements["synapse_count"]
            max_synapses = self.hardware_capabilities.get("max_synapses", 16000000)
            if req_synapses > max_synapses:
                compatible = False
                details["synapse_count"] = f"Required: {req_synapses}, Available: {max_synapses}"
        
        return compatible, details
    
    def configure_monitoring(self, monitoring_config: Dict[str, Any]) -> bool:
        """Configure hardware monitoring parameters."""
        with self.error_context("monitoring configuration"):
            return self.driver.configure_monitoring(monitoring_config)
    
    def allocate_memory(self, size: int, memory_type: str) -> Optional[int]:
        """Allocate memory of specified size and type."""
        with self.error_context("memory allocation"):
            # SpiNNaker has SDRAM and DTCM memory types
            if memory_type not in ["sdram", "dtcm"]:
                memory_type = "sdram"  # Default to SDRAM
                
            block_id = self.driver.allocate_memory(size, memory_type)
            if block_id is not None:
                self.memory_regions[block_id] = {"size": size, "type": memory_type}
                self.resource_usage["memory"] += size
            return block_id
    
    def free_memory(self, block_id: int) -> bool:
        """Free previously allocated memory."""
        with self.error_context("memory deallocation"):
            if block_id in self.memory_regions:
                result = self.driver.free_memory(block_id)
                if result:
                    self.resource_usage["memory"] -= self.memory_regions[block_id]["size"]
                    del self.memory_regions[block_id]
                return result
            return False
    
    def allocate_compute_units(self, count: int, unit_type: str) -> List[int]:
        """Allocate compute units of specified type."""
        with self.error_context("compute unit allocation"):
            # In SpiNNaker, compute units are ARM cores
            core_ids = self.driver.allocate_cores(count)
            if core_ids:
                for core_id in core_ids:
                    self.compute_units[core_id] = {"type": unit_type, "allocated": True}
                self.resource_usage["compute_units"] += len(core_ids)
            return core_ids
    
    def free_compute_units(self, unit_ids: List[int]) -> bool:
        """Free previously allocated compute units."""
        with self.error_context("compute unit deallocation"):
            success = True
            for unit_id in unit_ids:
                if unit_id in self.compute_units:
                    result = self.driver.free_core(unit_id)
                    if result:
                        del self.compute_units[unit_id]
                        self.resource_usage["compute_units"] -= 1
                    else:
                        success = False
            return success
    
    def create_communication_channel(self, source_id: int, target_id: int, channel_type: str) -> Optional[int]:
        """Create a communication channel between compute units."""
        with self.error_context("channel creation"):
            # SpiNNaker uses multicast routing for communication
            if source_id in self.neuron_mapping and target_id in self.neuron_mapping:
                hw_source = self.neuron_mapping[source_id]
                hw_target = self.neuron_mapping[target_id]
                
                channel_id = self.driver.create_multicast_route(hw_source, hw_target, channel_type)
                if channel_id is not None:
                    self.channel_mapping[channel_id] = {"source": hw_source, "target": hw_target}
                    return channel_id
            return None
    
    def close_communication_channel(self, channel_id: int) -> bool:
        """Close a communication channel."""
        with self.error_context("channel closure"):
            if channel_id in self.channel_mapping:
                result = self.driver.remove_multicast_route(channel_id)
                if result:
                    del self.channel_mapping[channel_id]
                return result
            return False


class HardwareFactory:
    """Factory for creating hardware instances."""
    
    @staticmethod
    def create_hardware(hardware_type: str) -> Optional[NeuromorphicHardware]:
        """
        Create a hardware instance based on the specified type.
        
        Args:
            hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth', 'simulated')
            
        Returns:
            Optional[NeuromorphicHardware]: Hardware instance or None if type is unsupported
        """
        hardware_type = hardware_type.lower()
        
        if hardware_type == "loihi":
            return LoihiHardware()
        elif hardware_type == "spinnaker":
            return SpiNNakerHardware()
        elif hardware_type == "truenorth":
            return TrueNorthHardware()
        elif hardware_type == "simulated" or hardware_type == "simulation":
            return SimulatedHardware()
        elif hardware_type.startswith("simulated_"):
            # Allow simulating specific hardware types
            # e.g., "simulated_loihi" will create a simulation of Loihi
            specific_type = hardware_type.split("_", 1)[1]
            return SimulatedHardware(hardware_type=specific_type)
        else:
            logger.warning(f"Unsupported hardware type: {hardware_type}")
            return None

# Add this method to the `NeuromorphicHardware` class to use this data transfer mechanism:
def transfer_data(self, data: Any, target_hardware: 'NeuromorphicHardware', 
                 mode: TransferMode = TransferMode.STANDARD) -> Any:
    """
    Transfer data to another hardware platform efficiently.
    
    Args:
        data: Data to transfer
        target_hardware: Target hardware instance
        mode: Transfer mode (STANDARD, FAST, DIRECT, BATCH)
        
    Returns:
        Any: Transferred data in target format
    """
    with self.error_context("data transfer"):
        source_type = self.__class__.__name__.lower().replace('hardware', '')
        target_type = target_hardware.__class__.__name__.lower().replace('hardware', '')
        
        return DataTransfer.transfer(data, source_type, target_type, mode)

# Add this method to the hardware abstraction classes

def share_resources(self, resource_type: str, quantity: int, 
                   process_id: str = "", timeout: float = 0.0) -> str:
    """
    Share hardware resources with other processes.
    
    Args:
        resource_type: Type of resource to share
        quantity: Amount to share
        process_id: Process ID (defaults to current process)
        timeout: Optional timeout in seconds
        
    Returns:
        str: Resource allocation ID
    """
    with self.error_context("resource sharing"):
        # Get process ID if not provided
        if not process_id:
            import os
            process_id = str(os.getpid())
        
        # Convert string resource type to enum
        try:
            res_type = ResourceType(resource_type)
        except ValueError:
            raise ValueError(f"Invalid resource type: {resource_type}")
        
        # Get resource manager
        manager = get_resource_manager()
        
        # Allocate resource
        return manager.allocate_resource(
            process_id, self.hardware_type, res_type, quantity, timeout)


def reserve_hardware(self, user_id: str, resources: Dict[str, int], 
                    duration_minutes: int, start_time: Optional[float] = None) -> str:
    """
    Reserve hardware for a specific time period.
    
    Args:
        user_id: User ID
        resources: Resource requirements (type -> quantity)
        duration_minutes: Duration in minutes
        start_time: Optional start time (timestamp), defaults to now
        
    Returns:
        str: Reservation ID
    """
    with self.error_context("hardware reservation"):
        from src.core.hardware.reservation_system import get_reservation_system
        
        # Get reservation system
        reservation_system = get_reservation_system()
        
        # Create reservation
        return reservation_system.reserve_hardware(
            user_id, 
            self.hardware_type, 
            resources, 
            duration_minutes, 
            start_time
        )

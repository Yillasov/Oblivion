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

# Import hardware drivers
from src.core.hardware.loihi_driver import LoihiProcessor
# Import other hardware drivers as needed

logger = get_logger("hardware_abstraction")


class NeuromorphicHardware(ABC):
    """
    Abstract base class defining the hardware abstraction interface.
    All hardware-specific implementations should inherit from this class.
    """
    
    def __init__(self):
        """Initialize the hardware abstraction."""
        # Fix: Use neuromorphic_logger instead of the logger instance
        self.error_handler = ErrorHandler(neuromorphic_logger)
        self.initialized = False
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
    
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
        """Allocate hardware resources based on request."""
        pass
    
    @abstractmethod
    def run_computation(self, inputs: Dict[str, Any], duration_ms: float) -> Dict[str, Any]:
        """Run a computation on the hardware with the given inputs."""
        pass
    
    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about the hardware."""
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


class LoihiHardware(NeuromorphicHardware):
    """Hardware abstraction implementation for Intel Loihi."""
    
    def __init__(self):
        """Initialize the Loihi hardware abstraction."""
        super().__init__()
        self.driver = LoihiProcessor()
        self.neuron_mapping = {}  # Maps logical neuron IDs to hardware neuron IDs
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the Loihi hardware with the given configuration."""
        with self.error_context("hardware initialization"):
            if config is None:
                config = {}
                
            result = self.driver.initialize(config)
            self.initialized = result
            return result
    
    def shutdown(self) -> bool:
        """Safely shutdown the Loihi hardware."""
        if not self.initialized:
            return True
            
        with self.error_context("hardware shutdown"):
            result = self.driver.shutdown()
            if result:
                self.initialized = False
            return result
    
    def allocate_resources(self, resource_request: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources on Loihi hardware."""
        if not self.initialized:
            raise HardwareInitializationError("Hardware not initialized")
        
        with self.error_context("resource allocation", self._reset_allocation):
            result = {"success": True, "resources": {}}
            
            # Allocate neurons
            if "neuron_count" in resource_request:
                count = resource_request["neuron_count"]
                params = resource_request.get("neuron_params", {})
                
                neuron_ids = self.driver.allocate_neurons(count, params)
                result["resources"]["neuron_ids"] = neuron_ids
            
            # Create synapses
            if "connections" in resource_request:
                connections = resource_request["connections"]
                synapse_ids = self.driver.create_synapses(connections)
                result["resources"]["synapse_ids"] = synapse_ids
            
            return result
    
    def run_computation(self, inputs: Dict[str, Any], duration_ms: float) -> Dict[str, Any]:
        """Run a computation on Loihi hardware."""
        if not self.initialized:
            raise HardwareInitializationError("Hardware not initialized")
        
        with self.error_context("computation execution", self._reset_computation):
            # Set spike inputs
            if "spike_inputs" in inputs:
                spike_inputs = inputs["spike_inputs"]
                neuron_ids = list(spike_inputs.keys())
                spike_times = [spike_inputs[nid] for nid in neuron_ids]
                
                self.driver.set_spike_input(neuron_ids, spike_times)
            
            # Run simulation
            success = self.driver.run_simulation(duration_ms)
            
            if not success:
                raise HardwareSimulationError("Simulation failed")
            
            # Get outputs
            output_neuron_ids = inputs.get("output_neurons", list(self.driver._neurons.keys()))
            spike_outputs = self.driver.get_spike_output(output_neuron_ids)
            
            return {
                "success": True,
                "spike_outputs": spike_outputs,
                "duration_ms": duration_ms
            }
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about the Loihi hardware."""
        with self.error_context("hardware info retrieval"):
            return self.driver.get_hardware_info()
    
    def _reset_allocation(self):
        """Reset resource allocation state after failure."""
        logger.info("Resetting resource allocation state")
        # Implementation would depend on specific hardware
        # For example, might need to free allocated resources
    
    def _reset_computation(self):
        """Reset computation state after failure."""
        logger.info("Resetting computation state")
        # Implementation would depend on specific hardware
        # For example, might need to clear input buffers


class HardwareFactory:
    """Factory for creating hardware abstraction instances."""
    
    @staticmethod
    def create_hardware(hardware_type: str) -> Optional[NeuromorphicHardware]:
        """
        Create a hardware abstraction instance for the specified hardware type.
        
        Args:
            hardware_type: Type of neuromorphic hardware ('loihi', 'spinnaker', etc.)
            
        Returns:
            NeuromorphicHardware: Hardware abstraction instance or None if not supported
        """
        try:
            hardware_map = {
                'loihi': LoihiHardware,
                # Add other hardware types as they become available:
                # 'spinnaker': SpiNNakerHardware,
                # 'truenorth': TrueNorthHardware,
            }
            
            hardware_class = hardware_map.get(hardware_type.lower())
            if hardware_class:
                return hardware_class()
            
            logger.error(f"Unsupported hardware type: {hardware_type}")
            raise UnsupportedFeatureError(f"Unsupported hardware type: {hardware_type}")
            
        except Exception as e:
            logger.error(f"Failed to create hardware instance: {str(e)}")
            if isinstance(e, NeuromorphicHardwareError):
                raise
            raise HardwareInitializationError(f"Failed to create hardware: {str(e)}")
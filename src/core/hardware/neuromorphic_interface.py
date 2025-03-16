"""
Neuromorphic Hardware Abstraction Interface

This module defines the base interfaces for interacting with neuromorphic hardware
in a hardware-agnostic manner, allowing the SDK to support multiple neuromorphic
processor architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class NeuromorphicProcessor(ABC):
    """
    Abstract base class defining the interface for neuromorphic processors.
    All specific hardware implementations must inherit from this class.
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the neuromorphic processor with the given configuration.
        
        Args:
            config: Dictionary containing hardware-specific configuration parameters
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        Safely shutdown the neuromorphic processor.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def allocate_neurons(self, count: int, neuron_params: Dict[str, Any]) -> List[int]:
        """
        Allocate a specified number of neurons on the hardware.
        
        Args:
            count: Number of neurons to allocate
            neuron_params: Dictionary containing neuron configuration parameters
            
        Returns:
            List[int]: List of neuron IDs that were allocated
        """
        pass
    
    @abstractmethod
    def create_synapses(self, connections: List[Tuple[int, int, float]]) -> List[int]:
        """
        Create synaptic connections between neurons.
        
        Args:
            connections: List of tuples (pre_neuron_id, post_neuron_id, weight)
            
        Returns:
            List[int]: List of synapse IDs that were created
        """
        pass
    
    @abstractmethod
    def set_spike_input(self, neuron_ids: List[int], spike_times: List[List[float]]) -> bool:
        """
        Set input spike trains for specified neurons.
        
        Args:
            neuron_ids: List of neuron IDs to receive input spikes
            spike_times: List of spike time lists, one per neuron
            
        Returns:
            bool: True if input was set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def run_simulation(self, duration_ms: float) -> bool:
        """
        Run the simulation for the specified duration.
        
        Args:
            duration_ms: Duration of simulation in milliseconds
            
        Returns:
            bool: True if simulation completed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_spike_output(self, neuron_ids: List[int]) -> Dict[int, List[float]]:
        """
        Get output spike times for specified neurons.
        
        Args:
            neuron_ids: List of neuron IDs to retrieve output spikes from
            
        Returns:
            Dict[int, List[float]]: Dictionary mapping neuron IDs to lists of spike times
        """
        pass
    
    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get information about the neuromorphic hardware.
        
        Returns:
            Dict[str, Any]: Dictionary containing hardware information
        """
        pass


class NeuromorphicProcessorManager:
    """
    Manager class for neuromorphic processors, providing a unified interface
    for processor discovery, initialization, and management.
    """
    
    def __init__(self):
        self._available_processors = {}
        self._active_processors = {}
    
    def discover_processors(self) -> List[str]:
        """
        Discover available neuromorphic processors in the system.
        
        Returns:
            List[str]: List of processor identifiers
        """
        # Implementation will scan for available hardware
        # This is a placeholder for hardware discovery logic
        return list(self._available_processors.keys())
    
    def register_processor(self, processor_id: str, processor_class: type) -> None:
        """
        Register a neuromorphic processor implementation.
        
        Args:
            processor_id: Unique identifier for the processor
            processor_class: Class implementing NeuromorphicProcessor interface
        """
        if not issubclass(processor_class, NeuromorphicProcessor):
            raise TypeError("Processor class must implement NeuromorphicProcessor interface")
        self._available_processors[processor_id] = processor_class
    
    def initialize_processor(self, processor_id: str, config: Dict[str, Any]) -> bool:
        """
        Initialize a specific neuromorphic processor.
        
        Args:
            processor_id: Identifier of the processor to initialize
            config: Configuration parameters for the processor
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if processor_id not in self._available_processors:
            return False
        
        processor_class = self._available_processors[processor_id]
        processor = processor_class()
        
        if processor.initialize(config):
            self._active_processors[processor_id] = processor
            return True
        return False
    
    def get_processor(self, processor_id: str) -> Optional[NeuromorphicProcessor]:
        """
        Get an initialized processor instance.
        
        Args:
            processor_id: Identifier of the processor to retrieve
            
        Returns:
            Optional[NeuromorphicProcessor]: The processor instance, or None if not found
        """
        return self._active_processors.get(processor_id)
    
    def shutdown_processor(self, processor_id: str) -> bool:
        """
        Shutdown a specific neuromorphic processor.
        
        Args:
            processor_id: Identifier of the processor to shutdown
            
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        if processor_id not in self._active_processors:
            return False
        
        processor = self._active_processors[processor_id]
        if processor.shutdown():
            del self._active_processors[processor_id]
            return True
        return False
    
    def shutdown_all(self) -> bool:
        """
        Shutdown all active neuromorphic processors.
        
        Returns:
            bool: True if all processors were shutdown successfully, False otherwise
        """
        success = True
        processor_ids = list(self._active_processors.keys())
        
        for processor_id in processor_ids:
            if not self.shutdown_processor(processor_id):
                success = False
        
        return success
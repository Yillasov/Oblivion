#!/usr/bin/env python3
"""
Unified Hardware Interface

Provides a standardized interface for interacting with different neuromorphic 
hardware platforms (Loihi, SpiNNaker, and TrueNorth).
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging

from src.core.utils.logging_framework import get_logger
from src.core.hardware.exceptions import (
    HardwareInitializationError,
    HardwareAllocationError,
    HardwareSimulationError
)
from src.core.hardware.optimizations import get_optimizer
from src.core.hardware.hardware_detection import hardware_detector

logger = get_logger("hardware_interface")


class NeuromorphicHardwareInterface:
    """
    Unified interface for all neuromorphic hardware platforms.
    """
    
    def __init__(self, hardware_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hardware interface.
        
        Args:
            hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth'), or None for auto-detection
            config: Optional configuration parameters
        """
        # Auto-detect hardware if not specified
        if hardware_type is None:
            detected = hardware_detector.get_best_hardware()
            if detected:
                self.hardware_type, detected_config = detected
                logger.info(f"Auto-detected {self.hardware_type} hardware")
                
                # Merge detected config with provided config
                self.config = detected_config.copy()
                if config:
                    self.config.update(config)
            else:
                # Default to Loihi if nothing detected
                self.hardware_type = "loihi"
                self.config = config or {}
                logger.warning("No hardware detected, defaulting to Loihi")
        else:
            self.hardware_type = hardware_type.lower()
            self.config = config or {}
        
        self.hardware = self._create_hardware_instance()
        self.initialized = False
        
        # Get hardware-specific optimizer
        try:
            self.optimizer = get_optimizer(self.hardware_type)
        except ValueError:
            logger.warning(f"No optimizer available for {self.hardware_type}")
            self.optimizer = None
    
    def _create_hardware_instance(self):
        """Create the appropriate hardware instance based on type."""
        from src.core.hardware.hardware_abstraction import HardwareFactory
        return HardwareFactory.create_hardware(self.hardware_type)
    
    def initialize(self) -> bool:
        """Initialize the hardware with the current configuration."""
        if not self.hardware:
            logger.error(f"Unsupported hardware type: {self.hardware_type}")
            raise HardwareInitializationError(f"Unsupported hardware type: {self.hardware_type}")
        
        try:
            result = self.hardware.initialize(self.config)
            self.initialized = result
            return result
        except Exception as e:
            logger.error(f"Hardware initialization failed: {str(e)}")
            self.initialized = False
            raise HardwareInitializationError(f"Initialization failed: {str(e)}") from e
    
    def shutdown(self) -> bool:
        """Safely shutdown the hardware."""
        if not self.hardware:
            return False
        
        return self.hardware.shutdown()
    
    def allocate_neurons(self, count: int, params: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Allocate neurons on the hardware.
        
        Args:
            count: Number of neurons to allocate
            params: Optional neuron parameters
            
        Returns:
            List[int]: IDs of allocated neurons
        """
        if not self.initialized:
            raise HardwareInitializationError("Hardware not initialized")
        
        if not self.hardware:
            raise HardwareInitializationError(f"No hardware instance available for {self.hardware_type}")
        
        resource_request = {
            "neuron_count": count,
            "neuron_params": params or {}
        }
        
        result = self.hardware.allocate_resources(resource_request)
        return result.get("resources", {}).get("neuron_ids", [])
    
    def create_connections(self, connections: List[Tuple[int, int, float]]) -> List[int]:
        """
        Create connections between neurons.
        
        Args:
            connections: List of (source_id, target_id, weight) tuples
            
        Returns:
            List[int]: IDs of created connections
        """
        if not self.initialized:
            raise HardwareInitializationError("Hardware not initialized")
        
        if not self.hardware:
            raise HardwareInitializationError(f"No hardware instance available for {self.hardware_type}")
        
        resource_request = {
            "connections": connections
        }
        
        result = self.hardware.allocate_resources(resource_request)
        return result.get("resources", {}).get("synapse_ids", [])
    
    def run_simulation(self, inputs: Dict[int, List[float]], duration_ms: float, 
                      output_neurons: Optional[List[int]] = None) -> Dict[int, List[float]]:
        """
        Run a simulation on the hardware.
        
        Args:
            inputs: Dictionary mapping neuron IDs to spike times
            duration_ms: Duration of simulation in milliseconds
            output_neurons: Optional list of neuron IDs to record from
            
        Returns:
            Dict[int, List[float]]: Dictionary mapping neuron IDs to output spike times
        """
        if not self.initialized:
            raise HardwareInitializationError("Hardware not initialized")
        
        if not self.hardware:
            raise HardwareInitializationError(f"No hardware instance available for {self.hardware_type}")
        
        computation_inputs = {
            "spike_inputs": inputs,
            "output_neurons": output_neurons
        }
        
        result = self.hardware.run_computation(computation_inputs, duration_ms)
        return result.get("spike_outputs", {})
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about the hardware."""
        if not self.hardware:
            return {"error": f"Unsupported hardware type: {self.hardware_type}"}
        
        return self.hardware.get_hardware_info()
    
    def optimize_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply hardware-specific optimizations to a neural network configuration.
        
        Args:
            network_config: Neural network configuration
            
        Returns:
            Dict[str, Any]: Optimized configuration
        """
        if not self.optimizer:
            logger.warning(f"No optimizer available for {self.hardware_type}, returning unoptimized config")
            return network_config
        
        return self.optimizer.optimize_network(network_config)
    
    def optimize_resource_allocation(self, resource_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply hardware-specific optimizations to resource allocation.
        
        Args:
            resource_request: Resource allocation request
            
        Returns:
            Dict[str, Any]: Optimized resource request
        """
        if not self.optimizer:
            logger.warning(f"No optimizer available for {self.hardware_type}, returning unoptimized request")
            return resource_request
        
        return self.optimizer.optimize_resource_allocation(resource_request)
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get hardware-specific optimization recommendations.
        
        Returns:
            List[str]: List of optimization recommendations
        """
        if not self.optimizer:
            return [f"No specific optimization recommendations available for {self.hardware_type}"]
        
        return self.optimizer.get_optimization_recommendations()


# Factory function to create hardware interfaces
def create_hardware_interface(hardware_type: Optional[str] = None, 
                             config: Optional[Dict[str, Any]] = None) -> NeuromorphicHardwareInterface:
    """
    Create a hardware interface for the specified hardware type.
    
    Args:
        hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth'), or None for auto-detection
        config: Optional configuration parameters
        
    Returns:
        NeuromorphicHardwareInterface: Unified hardware interface
    """
    return NeuromorphicHardwareInterface(hardware_type, config)


# Remove this duplicate function declaration
# def create_hardware_interface(hardware_type: str, config: Optional[Dict[str, Any]] = None) -> NeuromorphicHardwareInterface:
#    """
#    Create a hardware interface for the specified hardware type.
#    
#    Args:
#        hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth')
#        config: Optional configuration parameters
#        
#    Returns:
#        NeuromorphicHardwareInterface: Unified hardware interface
#    """
#    return NeuromorphicHardwareInterface(hardware_type, config)

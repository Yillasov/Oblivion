"""
Resource Allocation Manager

Provides unified interface for hardware-specific resource allocation.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import threading
import time

from src.core.hardware.resource_allocation import ResourceAllocator
from src.core.hardware.optimizations import get_optimizer
from src.core.hardware.exceptions import HardwareAllocationError
from src.core.utils.logging_framework import get_logger
from src.core.hardware.resource_mapping import global_usage_tracker
from src.core.hardware.hardware_config import config_store

logger = get_logger("resource_manager")


class ResourceManager:
    """Manages resource allocation across different hardware types."""
    
    def __init__(self, hardware_capabilities: Dict[str, Any]):
        """
        Initialize the resource manager.
        
        Args:
            hardware_capabilities: Hardware capabilities dictionary
        """
        self.capabilities = hardware_capabilities
        self.hardware_type = hardware_capabilities.get("hardware_type", "unknown")
        
        # Create hardware-specific allocation strategy
        self.allocation_strategy = ResourceAllocator.create_strategy(hardware_capabilities)
        
        # Get hardware-specific optimizer if available
        try:
            self.optimizer = get_optimizer(self.hardware_type)
        except ValueError:
            logger.warning(f"No optimizer available for {self.hardware_type}")
            self.optimizer = None
        
        # Track allocated resources
        self.allocated_resources = {
            "neurons": 0,
            "cores": 0,
            "chips": 0,
            "connections": 0
        }
    
    def allocate_neurons(self, count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate neurons with hardware-specific optimizations.
        
        Args:
            count: Number of neurons to allocate
            params: Neuron parameters
            
        Returns:
            Dict[str, Any]: Allocation result
        """
        # Apply optimizer if available
        resource_request = {
            "neuron_count": count,
            "neuron_params": params
        }
        
        if self.optimizer:
            resource_request = self.optimizer.optimize_resource_allocation(resource_request)
        
        # Allocate using hardware-specific strategy
        result = self.allocation_strategy.allocate_neurons(
            resource_request.get("neuron_count", count),
            resource_request.get("neuron_params", params)
        )
        
        # Update allocated resources
        self.allocated_resources["neurons"] += result.get("allocated_neurons", 0)
        self.allocated_resources["cores"] += result.get("cores_used", 0)
        self.allocated_resources["chips"] += result.get("chips_used", 0)
        
        return result
    
    def allocate_synapses(self, connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Allocate synapses with hardware-specific optimizations.
        
        Args:
            connections: List of connection dictionaries
            
        Returns:
            Dict[str, Any]: Allocation result
        """
        # Convert to format expected by allocation strategy
        conn_tuples = [(c.get("pre_id", 0), c.get("post_id", 0), c.get("weight", 0.0)) 
                      for c in connections]
        
        # Apply optimizer if available
        resource_request = {
            "connections": connections
        }
        
        if self.optimizer:
            resource_request = self.optimizer.optimize_resource_allocation(resource_request)
        
        # Allocate using hardware-specific strategy
        result = self.allocation_strategy.allocate_synapses(conn_tuples)
        
        # Update allocated resources
        self.allocated_resources["connections"] += result.get("allocated_synapses", 0)
        
        return result
    
    def optimize_placement(self, neuron_groups: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Optimize neuron placement with hardware-specific strategy.
        
        Args:
            neuron_groups: Dictionary of neuron group name to list of neuron IDs
            
        Returns:
            Dict[str, Any]: Optimized placement
        """
        return self.allocation_strategy.optimize_placement(neuron_groups)
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get hardware-specific optimization recommendations.
        
        Returns:
            List[str]: Optimization recommendations
        """
        if self.optimizer:
            return self.optimizer.get_optimization_recommendations()
        return ["No hardware-specific optimizer available for recommendations"]
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage.
        
        Returns:
            Dict[str, Any]: Resource usage statistics
        """
        # Add hardware-specific capacity information
        usage = self.allocated_resources.copy()
        
        # Add capacity percentages
        if "neurons_per_core" in self.capabilities and "cores_per_chip" in self.capabilities:
            neurons_per_core = self.capabilities["neurons_per_core"]
            cores_per_chip = self.capabilities["cores_per_chip"]
            total_neurons = neurons_per_core * cores_per_chip * self.capabilities.get("chips_available", 1)
            
            usage["neuron_capacity_percent"] = (usage["neurons"] / total_neurons) * 100 if total_neurons > 0 else 0
            usage["core_capacity_percent"] = (usage["cores"] / (cores_per_chip * self.capabilities.get("chips_available", 1))) * 100 if cores_per_chip > 0 else 0
        
        return usage
    
    def reset_allocation(self):
        """Reset all resource allocations."""
        self.allocated_resources = {
            "neurons": 0,
            "cores": 0,
            "chips": 0,
            "connections": 0
        }
    
    # Add these methods to the ResourceManager class
    def load_hardware_config(self, config_name: str) -> bool:
        """
        Load hardware configuration.
        
        Args:
            config_name: Configuration name
            
        Returns:
            bool: Success status
        """
        config = config_store.load_config(self.hardware_type, config_name)
        if not config:
            logger.error(f"Failed to load configuration '{config_name}' for {self.hardware_type}")
            return False
            
        # Update capabilities with loaded config
        self.capabilities.update(config)
        
        # Set as active config
        config_store.set_active_config(self.hardware_type, config_name)
        
        logger.info(f"Loaded configuration '{config_name}' for {self.hardware_type}")
        return True

# Add these methods to the ResourceManager class
def validate_hardware_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate hardware configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    return config_store.validate_config(self.hardware_type, config)

def save_hardware_config(self, config_name: str) -> bool:
    """
    Save current hardware configuration.
    
    Args:
        config_name: Configuration name
        
    Returns:
        bool: Success status
    """
    # Validate current capabilities
    is_valid, errors = self.validate_hardware_config(self.capabilities)
    if not is_valid:
        logger.error(f"Cannot save invalid configuration: {', '.join(errors)}")
        return False
    
    # Save current capabilities
    result = config_store.save_config(self.hardware_type, config_name, self.capabilities)
    if result:
        logger.info(f"Saved current configuration as '{config_name}' for {self.hardware_type}")
    return result
    
    def list_hardware_configs(self) -> List[str]:
        """
        List available configurations for current hardware.
        
        Returns:
            List[str]: Available configuration names
        """
        configs = config_store.list_configs(self.hardware_type)
        return configs.get(self.hardware_type, [])
    
    def apply_monitoring_config(self, monitoring_config: Dict[str, Any]) -> bool:
        """
        Apply monitoring configuration.
        
        Args:
            monitoring_config: Monitoring configuration
            
        Returns:
            bool: Success status
        """
        # Update capabilities with monitoring config
        if "monitoring" not in self.capabilities:
            self.capabilities["monitoring"] = {}
            
        self.capabilities["monitoring"].update(monitoring_config)
        
        # Start/stop monitoring based on config
        if monitoring_config.get("enabled", False):
            interval = monitoring_config.get("interval_ms", 5000) / 1000.0
            return self.start_usage_monitoring(interval)
        else:
            return self.stop_usage_monitoring()
    
    def start_usage_monitoring(self, interval: float = 5.0) -> bool:
        """
        Start periodic resource usage monitoring.
        
        Args:
            interval: Monitoring interval in seconds
            
        Returns:
            bool: True if monitoring started, False otherwise
        """
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread is not None:
            logger.warning("Resource monitoring already started")
            return False
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started resource usage monitoring with interval {interval}s")
        return True
    
    def stop_usage_monitoring(self) -> bool:
        """
        Stop resource usage monitoring.
        
        Returns:
            bool: True if monitoring stopped, False otherwise
        """
        if not hasattr(self, 'monitoring_thread') or self.monitoring_thread is None:
            logger.warning("Resource monitoring not active")
            return False
        
        self.monitoring_active = False
        self.monitoring_thread.join(timeout=2.0)
        self.monitoring_thread = None
        logger.info("Stopped resource usage monitoring")
        return True
    
    def _monitoring_loop(self, interval: float) -> None:
        """
        Resource monitoring loop.
        
        Args:
            interval: Monitoring interval in seconds
        """
        while getattr(self, 'monitoring_active', False):
            try:
                # Get current usage and record it
                usage = self.get_resource_usage()
                global_usage_tracker.record_usage(usage)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
            
            # Sleep until next check
            time.sleep(interval)
    
    def get_usage_report(self) -> Dict[str, Any]:
        """
        Get resource usage report.
        
        Returns:
            Dict[str, Any]: Resource usage report
        """
        from src.core.hardware.resource_mapping import get_resource_report
        return get_resource_report()


# Factory function to create a resource manager for a specific hardware
def create_resource_manager(hardware_capabilities: Dict[str, Any]) -> ResourceManager:
    """
    Create a resource manager for specific hardware.
    
    Args:
        hardware_capabilities: Hardware capabilities dictionary
        
    Returns:
        ResourceManager: Hardware-specific resource manager
    """
    return ResourceManager(hardware_capabilities)
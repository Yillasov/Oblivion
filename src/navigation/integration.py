"""
Navigation integration framework for UCAV platforms.

This module provides integration capabilities for multiple navigation systems
with neuromorphic hardware support.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from typing import Dict, List, Any, Optional

from src.navigation.base import NavigationSystem, NavigationSpecs
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.core.hardware.neuromorphic_resource_manager import NeuromorphicResourceManager, ResourceType

# Configure logger
logger = logging.getLogger(__name__)


class NavigationIntegrator:
    """Framework for integrating navigation systems with UCAV platform."""
    
    def __init__(self, hardware_interface=None, config=None):
        """
        Initialize the navigation integrator.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
            config: Optional configuration parameters
        """
        from src.core.integration.system_factory import NeuromorphicSystemFactory
        
        self.system = NeuromorphicSystemFactory.create_system(hardware_interface, config)
        self.navigation_systems: Dict[str, NavigationSystem] = {}
        self.system_states: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[Dict[str, float]]] = {}
        self.process_id = f"navigation_{id(self)}"
        self.initialized = False
        
        # Get resource manager
        self.resource_manager = NeuromorphicResourceManager.get_instance()
        self.resource_manager.register_process(self.process_id)
        self.resource_allocations = {}
        
        logger.info("Navigation integrator initialized")
    
    def register_navigation_system(self, system_id: str, system: NavigationSystem) -> bool:
        """Register a navigation system with the integrator."""
        if system_id in self.navigation_systems:
            logger.warning(f"Navigation system {system_id} already registered")
            return False
            
        self.navigation_systems[system_id] = system
        self.system_states[system_id] = {
            "initialized": False,
            "active": False,
            "accuracy": 0.0,
            "health": 1.0
        }
        self.performance_history[system_id] = []
        logger.info(f"Registered navigation system: {system_id}")
        return True
    
    def initialize(self) -> bool:
        """Initialize the navigation integrator and all systems."""
        if self.initialized:
            return True
            
        try:
            # Allocate computational resources
            self._allocate_resources()
            
            # Initialize all registered navigation systems
            results = self.initialize_systems()
            
            # Check if all systems initialized successfully
            if all(results.values()):
                self.initialized = True
                logger.info("Navigation integrator initialized successfully")
                return True
            else:
                failed_systems = [sys_id for sys_id, success in results.items() if not success]
                logger.warning(f"Failed to initialize navigation systems: {failed_systems}")
                self._release_resources()
                return False
                
        except Exception as e:
            logger.error(f"Error during navigation integrator initialization: {str(e)}")
            self._release_resources()
            return False
    
    def _allocate_resources(self) -> None:
        """Allocate necessary resources for navigation systems."""
        try:
            # Calculate resource needs based on registered systems
            hardware_type = self.system.__class__.__name__
            neuron_count = len(self.navigation_systems) * 200  # Base allocation per system
            
            # Allocate neurons
            neuron_alloc = self.resource_manager.allocate(
                self.process_id, 
                hardware_type,
                ResourceType.NEURON,
                neuron_count
            )
            self.resource_allocations["neurons"] = neuron_alloc
            
            # Allocate memory
            memory_alloc = self.resource_manager.allocate(
                self.process_id,
                hardware_type,
                ResourceType.MEMORY,
                neuron_count * 10
            )
            self.resource_allocations["memory"] = memory_alloc
            
            logger.info(f"Allocated resources for navigation: {neuron_count} neurons")
            
        except Exception as e:
            logger.error(f"Failed to allocate resources: {str(e)}")
            raise
    
    def _release_resources(self) -> None:
        """Release all allocated resources."""
        for allocation_id in self.resource_allocations.values():
            try:
                self.resource_manager.release(allocation_id)
            except Exception as e:
                logger.error(f"Error releasing resource {allocation_id}: {str(e)}")
        self.resource_allocations.clear()
    
    def initialize_systems(self) -> Dict[str, bool]:
        """Initialize all registered navigation systems."""
        results = {}
        for system_id, system in self.navigation_systems.items():
            try:
                success = system.initialize()
                self.system_states[system_id]["initialized"] = success
                results[system_id] = success
            except Exception as e:
                logger.error(f"Error initializing navigation system {system_id}: {str(e)}")
                self.system_states[system_id]["initialized"] = False
                results[system_id] = False
        return results
    
    def activate_system(self, system_id: str) -> bool:
        """Activate a specific navigation system."""
        if system_id not in self.navigation_systems:
            logger.warning(f"Navigation system {system_id} not found")
            return False
            
        system = self.navigation_systems[system_id]
        if not self.system_states[system_id]["initialized"]:
            logger.warning(f"Cannot activate uninitialized system {system_id}")
            return False
            
        success = system.activate()
        self.system_states[system_id]["active"] = success
        return success
    
    def deactivate_system(self, system_id: str) -> bool:
        """Deactivate a specific navigation system."""
        if system_id not in self.navigation_systems:
            return False
            
        system = self.navigation_systems[system_id]
        success = system.deactivate()
        self.system_states[system_id]["active"] = not success
        return success
    
    def get_position(self, system_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get position from a specific system or best available.
        
        If system_id is None, returns position from the most accurate active system.
        """
        if system_id:
            if system_id in self.navigation_systems and self.system_states[system_id]["active"]:
                return self.navigation_systems[system_id].get_position()
            return {"x": float('nan'), "y": float('nan'), "z": float('nan')}
        
        # Find most accurate active system
        best_system = None
        best_accuracy = -1.0
        
        for sys_id, system in self.navigation_systems.items():
            if self.system_states[sys_id]["active"]:
                accuracy = self.system_states[sys_id]["accuracy"]
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_system = sys_id
        
        if best_system:
            return self.navigation_systems[best_system].get_position()
        return {"x": float('nan'), "y": float('nan'), "z": float('nan')}
    
    def update_system_status(self) -> None:
        """Update status information for all systems."""
        for system_id, system in self.navigation_systems.items():
            status = system.get_status()
            self.system_states[system_id].update(status)
            
            # Record performance metrics
            if self.system_states[system_id]["active"]:
                metrics = system.calculate_performance_metrics()
                self.performance_history[system_id].append(metrics)
                
                # Limit history size
                if len(self.performance_history[system_id]) > 1000:
                    self.performance_history[system_id].pop(0)
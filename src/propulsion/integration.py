"""
Propulsion system integration framework for UCAV platforms.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

from src.propulsion.base import PropulsionInterface, PropulsionSpecs, PropulsionType
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.core.integration.system_factory import NeuromorphicSystemFactory
from src.core.hardware.resource_sharing import ResourceSharingManager, ResourceType
# Add the correct import here
from src.core.hardware.neuromorphic_resource_manager import NeuromorphicResourceManager, ResourceType as NeuromorphicResourceType

logger = logging.getLogger(__name__)

@dataclass
class PropulsionIntegrationConfig:
    """Configuration for propulsion system integration."""
    max_power_draw: float  # Maximum power draw in kW
    thermal_threshold: float  # Maximum thermal load in kW
    response_time_limit: float  # Maximum allowed response time in seconds
    safety_margin: float  # Safety margin for critical operations
    redundancy_level: int  # Level of system redundancy


class PropulsionIntegrator:
    """Framework for integrating propulsion systems with UCAV platform."""
    
    # Remove this incorrect import from inside the class
    # from src.core.hardware.neuromorphic_resource_manager import NeuromorphicResourceManager, ResourceType
    
    def __init__(self, 
                 config: PropulsionIntegrationConfig,
                 neuromorphic_system: Optional[NeuromorphicSystem] = None,
                 hardware_interface=None):
        """Initialize the propulsion integrator."""
        self.config = config
        self.initialized = False
        self.process_id = f"propulsion_{id(self)}"
        
        # Get neuromorphic resource manager instead of generic resource manager
        self.resource_manager = NeuromorphicResourceManager.get_instance()
        self.resource_manager.register_process(self.process_id)
        
        # Track resource allocations
        self.resource_allocations = {}
        
        # Use provided system or create a new one
        if neuromorphic_system:
            self.neuromorphic_system = neuromorphic_system
        else:
            try:
                self.neuromorphic_system = NeuromorphicSystemFactory.create_system(
                    hardware_interface, 
                    {"subsystem_type": "propulsion"}
                )
            except Exception as e:
                logger.error(f"Failed to create neuromorphic system: {str(e)}")
                self.neuromorphic_system = None
            
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.system_states: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[Dict[str, float]]] = {}
        self.active_configurations: Dict[str, bool] = {}
        
        logger.info("Propulsion integrator initialized")
        
    def initialize(self) -> bool:
        """Initialize the propulsion integrator and all systems."""
        if self.initialized:
            return True
            
        if not self.neuromorphic_system:
            logger.error("Cannot initialize: neuromorphic system not available")
            return False
            
        try:
            # Allocate computational resources
            self._allocate_resources()
            
            # Initialize all registered propulsion systems
            results = self.initialize_systems()
            
            # Check if all systems initialized successfully
            if all(results.values()):
                self.initialized = True
                logger.info("Propulsion integrator initialized successfully")
                return True
            else:
                failed_systems = [sys_id for sys_id, success in results.items() if not success]
                logger.warning(f"Failed to initialize propulsion systems: {failed_systems}")
                self._release_resources()
                return False
                
        except Exception as e:
            logger.error(f"Error during propulsion integrator initialization: {str(e)}")
            self._release_resources()
            return False
    
    def _allocate_resources(self) -> None:
        """Allocate necessary resources for propulsion systems."""
        try:
            # Allocate computational resources based on system needs
            hardware_type = self.neuromorphic_system.__class__.__name__
            
            # Calculate neuron count based on system complexity
            neuron_count = 0
            for system in self.propulsion_systems.values():
                specs = system.get_specifications()
                # Use throttle range and efficiency curve points as complexity indicators
                control_complexity = len(specs.efficiency_curve.get("power_levels", []))
                neuron_count += control_complexity * 10  # Base neurons per control point
            
            # Ensure minimum allocation
            neuron_count = max(neuron_count, 100)  # Minimum allocation
            
            # Use the new allocate method with NeuromorphicResourceType
            neuron_alloc = self.resource_manager.allocate(
                self.process_id, 
                hardware_type,
                NeuromorphicResourceType.NEURON,  # Use the imported type
                neuron_count
            )
            self.resource_allocations["neurons"] = neuron_alloc
            
            # Allocate memory
            memory_alloc = self.resource_manager.allocate(
                self.process_id,
                hardware_type,
                NeuromorphicResourceType.MEMORY,  # Use the imported type
                neuron_count * 10  # Rough estimate of memory needs
            )
            self.resource_allocations["memory"] = memory_alloc
            
            logger.info(f"Allocated resources for propulsion: {neuron_count} neurons")
            
        except Exception as e:
            logger.error(f"Failed to allocate resources: {str(e)}")
            self._release_resources()
            raise
    
    # And update the _release_resources method
    def _release_resources(self) -> None:
        """Release all allocated resources."""
        for allocation_id in self.resource_allocations.values():
            try:
                self.resource_manager.release(allocation_id)
            except Exception as e:
                logger.error(f"Error releasing resource {allocation_id}: {str(e)}")
        self.resource_allocations.clear()
        
    def register_propulsion_system(self, 
                                 system_id: str, 
                                 system: PropulsionInterface) -> bool:
        """Register a propulsion system with the integrator."""
        if system_id in self.propulsion_systems:
            logger.warning(f"Propulsion system {system_id} already registered")
            return False
            
        self.propulsion_systems[system_id] = system
        self.system_states[system_id] = {
            "initialized": False,
            "active": False,
            "health": 1.0,
            "thermal_load": 0.0,
            "power_draw": 0.0
        }
        self.performance_history[system_id] = []
        logger.info(f"Registered propulsion system: {system_id}")
        return True
        
    def initialize_systems(self) -> Dict[str, bool]:
        """Initialize all registered propulsion systems."""
        results = {}
        for system_id, system in self.propulsion_systems.items():
            try:
                success = system.initialize()
                self.system_states[system_id]["initialized"] = success
                results[system_id] = success
            except Exception as e:
                logger.error(f"Error initializing propulsion system {system_id}: {str(e)}")
                self.system_states[system_id]["initialized"] = False
                results[system_id] = False
        return results
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage information."""
        if not self.initialized:
            return {"status": "uninitialized"}
            
        hardware_type = self.neuromorphic_system.__class__.__name__
        pool = self.resource_manager.get_pool(hardware_type)
        if not pool:
            # Try to register the hardware if it doesn't exist yet
            resources = {
                NeuromorphicResourceType.NEURON: 10000,
                NeuromorphicResourceType.MEMORY: 100000,
                NeuromorphicResourceType.SYNAPSE: 50000
            }
            self.resource_manager.register_hardware(hardware_type, resources)
            pool = self.resource_manager.get_pool(hardware_type)
            
            if not pool:
                return {"status": "no_resource_pool"}
            
        # Get usage for this process
        allocations = self.resource_manager.get_process_allocations(self.process_id)
        
        usage = {
            "process_id": self.process_id,
            "allocations": list(self.resource_allocations.values()),
            "systems": len(self.propulsion_systems),
            "active_systems": sum(1 for state in self.system_states.values() if state["active"])
        }
        
        # Add utilization information if available
        if pool:
            usage["utilization"] = self.resource_manager.get_utilization(hardware_type)
        
        return usage
        
    def cleanup(self) -> None:
        """Clean up resources when shutting down."""
        self._release_resources()
        for system_id, system in self.propulsion_systems.items():
            try:
                # Check if system has a shutdown method before calling it
                if hasattr(system, 'shutdown'):
                    system.shutdown()
                else:
                    # Alternative: use set_power_state if available
                    system.set_power_state({"state": "off"})
            except Exception as e:
                logger.error(f"Error shutting down system {system_id}: {str(e)}")
        self.initialized = False
        logger.info("Propulsion integrator cleaned up")
        
    def configure_neuromorphic_control(self, 
                                     system_id: str,
                                     control_params: Dict[str, Any]) -> bool:
        """Configure neuromorphic control for a specific system."""
        if not self.neuromorphic_system:
            logger.error("Cannot configure: neuromorphic system not available")
            return False
            
        if system_id not in self.propulsion_systems:
            logger.error(f"Unknown propulsion system: {system_id}")
            return False
            
        try:
            system = self.propulsion_systems[system_id]
            specs = system.get_specifications()
            
            # Configure neural network for propulsion control
            control_config = {
                "input_dimensions": len(control_params.get("input_mapping", [])),
                "output_dimensions": len(control_params.get("output_mapping", [])),
                "response_time": specs.thermal_response_time,
                "control_frequency": control_params.get("control_frequency", 100),
                "adaptation_rate": control_params.get("adaptation_rate", 0.1)
            }
            
            success = self.neuromorphic_system.add_component(
                f"propulsion_control_{system_id}",
                control_config
            )
            
            if success:
                logger.info(f"Configured neuromorphic control for {system_id}")
            else:
                logger.warning(f"Failed to configure neuromorphic control for {system_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error configuring neuromorphic control for {system_id}: {str(e)}")
            return False
        
    def monitor_thermal_conditions(self) -> Dict[str, Dict[str, float]]:
        """Monitor thermal conditions of all systems."""
        thermal_status = {}
        for system_id, system in self.propulsion_systems.items():
            try:
                status = system.get_status()
                specs = system.get_specifications()
                
                thermal_load = status.get("temperature", 0.0)
                thermal_limit = specs.thermal_limits.get("max_operating", float('inf'))
                
                # Avoid division by zero
                if thermal_limit > 0:
                    thermal_margin = (thermal_limit - thermal_load) / thermal_limit
                else:
                    thermal_margin = 0.0
                
                thermal_status[system_id] = {
                    "current_load": thermal_load,
                    "limit": thermal_limit,
                    "margin": thermal_margin,
                    "critical": thermal_margin < self.config.safety_margin
                }
                
                # Update system state
                self.system_states[system_id]["thermal_load"] = thermal_load
                
            except Exception as e:
                logger.error(f"Error monitoring thermal conditions for {system_id}: {str(e)}")
                thermal_status[system_id] = {
                    "error": str(e),
                    "critical": True
                }
                
        return thermal_status
        
    def optimize_power_distribution(self) -> Dict[str, float]:
        """Optimize power distribution across propulsion systems."""
        power_allocation = {}
        total_power_available = self.config.max_power_draw
        
        try:
            # Get power requirements and priorities
            power_requirements = {}
            for system_id, system in self.propulsion_systems.items():
                if self.system_states[system_id]["active"]:
                    specs = system.get_specifications()
                    power_requirements[system_id] = specs.power_rating
                    
            # Allocate power based on priorities and requirements
            remaining_power = total_power_available
            for system_id, required_power in power_requirements.items():
                allocated = min(required_power, remaining_power)
                power_allocation[system_id] = allocated
                remaining_power -= allocated
                
                # Update system state
                self.system_states[system_id]["power_draw"] = allocated
                
        except Exception as e:
            logger.error(f"Error optimizing power distribution: {str(e)}")
            
        return power_allocation
        
    def update_system_states(self, 
                           flight_conditions: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Update and return the state of all propulsion systems."""
        states = {}
        
        if not flight_conditions:
            logger.warning("No flight conditions provided for update")
            return states
            
        for system_id, system in self.propulsion_systems.items():
            if self.system_states[system_id]["active"]:
                try:
                    performance = system.calculate_performance(flight_conditions)
                    status = system.get_status()
                    
                    states[system_id] = {
                        "performance": performance,
                        "status": status,
                        "health": self.system_states[system_id]["health"]
                    }
                    
                    # Update performance history
                    self.performance_history[system_id].append(performance)
                    
                except Exception as e:
                    logger.error(f"Error updating system state for {system_id}: {str(e)}")
                    states[system_id] = {"error": str(e)}
                
        return states
        
    def activate_system(self, system_id: str) -> bool:
        """Activate a propulsion system."""
        if system_id not in self.propulsion_systems:
            logger.error(f"Unknown propulsion system: {system_id}")
            return False
            
        if not self.system_states[system_id]["initialized"]:
            logger.error(f"Cannot activate uninitialized system: {system_id}")
            return False
            
        self.system_states[system_id]["active"] = True
        logger.info(f"Activated propulsion system: {system_id}")
        return True
        
    def deactivate_system(self, system_id: str) -> bool:
        """Deactivate a propulsion system."""
        if system_id not in self.propulsion_systems:
            logger.error(f"Unknown propulsion system: {system_id}")
            return False
            
        self.system_states[system_id]["active"] = False
        logger.info(f"Deactivated propulsion system: {system_id}")
        return True
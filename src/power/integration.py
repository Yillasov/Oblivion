"""
Integration framework for power supply systems with neuromorphic computing.

This module provides the necessary interfaces and classes to integrate
power supply systems with the rest of the UCAV platform.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional
import numpy as np

from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.power.base import NeuromorphicPowerSupply, PowerSupplySpecs, PowerSupplyType
from src.core.utils.logging_framework import get_logger

logger = get_logger("power_integration")


class PowerIntegrator:
    """Framework for integrating power systems with UCAV platform."""
    
    def __init__(self, hardware_interface=None, config=None):
        """
        Initialize the power integrator.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
            config: Configuration for power integration
        """
        self.hardware_interface = hardware_interface
        self.config = config or {}
        self.neuromorphic_system = NeuromorphicSystem(hardware_interface)
        self.power_systems: Dict[str, NeuromorphicPowerSupply] = {}
        self.system_states: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
        
        logger.info("Initialized power integration framework")
    
    def add_power_system(self, system_id: str, system: NeuromorphicPowerSupply) -> bool:
        """
        Add a power system to the integrator.
        
        Args:
            system_id: Unique identifier for the power system
            system: Power system instance
            
        Returns:
            Success status
        """
        if system_id in self.power_systems:
            logger.warning(f"Power system '{system_id}' already exists")
            return False
        
        self.power_systems[system_id] = system
        self.system_states[system_id] = {"active": False, "initialized": False}
        
        # Add to neuromorphic system as a component
        self.neuromorphic_system.add_component(f"power_{system_id}", system)
        
        logger.info(f"Added power system '{system_id}' to integrator")
        return True
    
    def initialize(self) -> bool:
        """Initialize the power integration system."""
        if self.initialized:
            return True
            
        try:
            # Initialize neuromorphic system
            if not self.neuromorphic_system.initialize():
                logger.error("Failed to initialize neuromorphic system")
                return False
            
            # Initialize each power system
            for system_id, system in self.power_systems.items():
                if system.initialize():
                    self.system_states[system_id]["initialized"] = True
                    self.system_states[system_id]["active"] = True
                else:
                    logger.warning(f"Failed to initialize power system '{system_id}'")
            
            # Allocate resources for power systems
            self._allocate_resources()
            
            self.initialized = True
            logger.info("Power integration system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize power integration: {str(e)}")
            return False
    
    def _allocate_resources(self) -> None:
        """Allocate necessary resources for power systems."""
        try:
            # Calculate neuron count based on system complexity
            neuron_count = 0
            for system_id, system in self.power_systems.items():
                if self.system_states[system_id]["active"]:
                    # Allocate based on power system type
                    specs = system.get_specifications()
                    power_type = getattr(system, 'type', PowerSupplyType.SOLID_STATE_BATTERY)
                    
                    # Different power systems need different resources
                    if power_type == PowerSupplyType.MICRO_NUCLEAR:
                        neuron_count += 500  # Complex system needs more neurons
                    elif power_type == PowerSupplyType.WIRELESS:
                        neuron_count += 300  # Medium complexity
                    else:
                        neuron_count += 100  # Standard allocation
            
            logger.info(f"Allocated {neuron_count} neurons for power systems")
            
        except Exception as e:
            logger.error(f"Error allocating resources: {str(e)}")
    
    def connect_to_system(self, system_id: str, target_system: str, 
                         connection_type: str = "power") -> bool:
        """
        Connect a power system to another system component.
        
        Args:
            system_id: Power system identifier
            target_system: Target system component
            connection_type: Type of connection
            
        Returns:
            Success status
        """
        if system_id not in self.power_systems:
            logger.error(f"Power system '{system_id}' not found")
            return False
            
        try:
            # Connect in the neuromorphic system
            source = f"power_{system_id}"
            self.neuromorphic_system.connect(source, target_system, connection_type)
            
            logger.info(f"Connected power system '{system_id}' to '{target_system}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect systems: {str(e)}")
            return False
    
    def optimize_power_distribution(self) -> Dict[str, float]:
        """Optimize power distribution across systems."""
        power_allocation = {}
        total_power_available = self.config.get("max_power", 1000.0)  # Default 1000 kW
        
        try:
            # Get power requirements
            power_requirements = {}
            for system_id, system in self.power_systems.items():
                if self.system_states[system_id]["active"]:
                    specs = system.get_specifications()
                    power_requirements[system_id] = specs.power_output
            
            # Simple proportional allocation
            total_required = sum(power_requirements.values())
            
            if total_required <= total_power_available:
                # Can satisfy all requirements
                power_allocation = power_requirements
            else:
                # Need to scale down proportionally
                scale_factor = total_power_available / total_required
                for system_id, required in power_requirements.items():
                    power_allocation[system_id] = required * scale_factor
            
            # Apply the allocations
            for system_id, allocation in power_allocation.items():
                system = self.power_systems[system_id]
                level = (allocation / system.get_specifications().power_output) * 100
                system.set_output_level(min(level, 100.0))
            
            logger.info(f"Optimized power distribution: {power_allocation}")
            return power_allocation
            
        except Exception as e:
            logger.error(f"Error optimizing power distribution: {str(e)}")
            return {}
    
    def train_power_systems(self, training_data: Dict[str, Any]) -> bool:
        """
        Train power systems with provided data.
        
        Args:
            training_data: Training data for power systems
            
        Returns:
            Success status
        """
        if not self.initialized:
            logger.error("Power integration not initialized")
            return False
            
        success = True
        
        try:
            # Train each power system
            for system_id, system_data in training_data.items():
                if system_id in self.power_systems:
                    system = self.power_systems[system_id]
                    if not system.train(system_data):
                        logger.warning(f"Failed to train power system '{system_id}'")
                        success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error training power systems: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all power systems."""
        status = {}
        
        for system_id, system in self.power_systems.items():
            status[system_id] = system.get_status()
            
        return status
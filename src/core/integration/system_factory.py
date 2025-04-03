"""
Centralized factory for creating and initializing neuromorphic systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Optional, Dict, Any, List
from src.core.integration.neuromorphic_system import NeuromorphicSystem, NeuromorphicInterface
import logging

logger = logging.getLogger(__name__)

class NeuromorphicSystemFactory:
    """Factory for creating standardized neuromorphic system instances."""
    
    @staticmethod
    def create_system(
        hardware_interface: Optional[NeuromorphicInterface] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> NeuromorphicSystem:
        """
        Create and initialize a standardized neuromorphic system.
        
        Args:
            hardware_interface: Optional hardware interface to use
            config: Optional configuration parameters
            
        Returns:
            Initialized NeuromorphicSystem instance
        """
        # Create the system
        system = NeuromorphicSystem(hardware_interface)
        
        # Apply standard configuration if provided
        if config:
            logger.info("Applying standard configuration to neuromorphic system")
            # Apply configuration settings here
            
        # Initialize the system
        success = system.initialize()
        if not success:
            logger.warning("Failed to initialize neuromorphic system")
            
        return system
    
    @staticmethod
    def apply_standard_configuration(system: NeuromorphicSystem, subsystem_type: str) -> bool:
        """
        Apply standard configuration based on subsystem type.
        
        Args:
            system: The neuromorphic system to configure
            subsystem_type: Type of subsystem ('propulsion', 'communication', etc.)
            
        Returns:
            bool: Success status
        """
        try:
            # Apply subsystem-specific optimizations
            if subsystem_type == "propulsion":
                # Propulsion systems need fast response times and thermal management
                system.add_component("thermal_monitor", {
                    "priority": "high",
                    "update_frequency": 100  # Hz
                })
            elif subsystem_type == "communication":
                # Communication systems need signal processing capabilities
                system.add_component("signal_processor", {
                    "mode": "adaptive",
                    "noise_reduction": True
                })
            elif subsystem_type == "payload":
                # Payload systems need precision control
                system.add_component("precision_controller", {
                    "precision": "high",
                    "feedback_enabled": True
                })
            
            logger.info(f"Applied standard configuration for {subsystem_type} subsystem")
            return True
        except Exception as e:
            logger.error(f"Failed to apply standard configuration: {str(e)}")
            return False
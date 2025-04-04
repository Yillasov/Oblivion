#!/usr/bin/env python3
"""
Integration system for biomimetic hardware components.
Provides interfaces for connecting biomimetic hardware with control systems.
"""

import os
import sys
from typing import Dict, Any, List, Optional

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.biomimetic.control.cpg_models import BiomimeticCPGController

logger = get_logger("biomimetic_hardware")

class NeuromorphicInterface:
    """Base interface for neuromorphic hardware."""
    
    def __init__(self):
        """Initialize the neuromorphic interface."""
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the hardware interface."""
        self.initialized = True
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.initialized = False
    
    def get_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        return {"type": "base_neuromorphic_interface", "initialized": self.initialized}


class BiomimeticHardwareIntegration:
    """Integration system for biomimetic hardware components."""
    
    def __init__(self, hardware_interface: Optional[NeuromorphicInterface] = None):
        """
        Initialize biomimetic hardware integration.
        
        Args:
            hardware_interface: Optional neuromorphic hardware interface
        """
        self.hardware_interface = hardware_interface
        self.actuator_controller = None
        self.cpg_controller = None
        self.initialized = False
        self.integration_mappings = {}
    
    def initialize(self) -> bool:
        """Initialize the integration system."""
        try:
            # Create actuator controller
            self.actuator_controller = self._create_biomimetic_actuator_system(self.hardware_interface)
            
            # Create CPG controller
            self.cpg_controller = BiomimeticCPGController()
            
            # Set up default CPG networks for wing control
            self.cpg_controller.create_network("wing_flapping", 2)
            
            # Create integration mappings
            self._create_integration_mappings()
            
            self.initialized = True
            logger.info("Biomimetic hardware integration initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize biomimetic hardware integration: {e}")
            return False
    
    def _create_biomimetic_actuator_system(self, hardware_interface):
        """Create the actuator control system."""
        # Simplified actuator system for demonstration
        actuator_system = {
            "actuator_groups": {
                "left_wing_muscles": {"type": "muscle", "count": 4},
                "right_wing_muscles": {"type": "muscle", "count": 4},
                "tail_muscles": {"type": "muscle", "count": 2}
            }
        }
        return actuator_system
    
    def _create_integration_mappings(self) -> None:
        """Create mappings between CPG outputs and actuator inputs."""
        # Map CPG network outputs to actuator groups
        self.integration_mappings = {
            "wing_flapping": {
                "oscillator_0": {"actuator_group": "left_wing_muscles", "scaling": 1.0},
                "oscillator_1": {"actuator_group": "right_wing_muscles", "scaling": 1.0}
            }
        }
    
    def update(self, dt: float = 0.01) -> Dict[str, Any]:
        """
        Update the integrated system.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            System state
        """
        if not self.initialized:
            logger.error("Cannot update uninitialized system")
            return {}
        
        # Update CPG networks
        cpg_outputs = self.cpg_controller.update(dt)
        
        # Map CPG outputs to actuator commands
        actuator_commands = self._map_cpg_to_actuators(cpg_outputs)
        
        # Apply actuator commands
        actuator_states = self._apply_actuator_commands(actuator_commands)
        
        return {
            "cpg_outputs": cpg_outputs,
            "actuator_commands": actuator_commands,
            "actuator_states": actuator_states
        }
    
    def _map_cpg_to_actuators(self, cpg_outputs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Map CPG outputs to actuator commands."""
        actuator_commands = {}
        
        for network_name, network_outputs in cpg_outputs.items():
            if network_name in self.integration_mappings:
                mappings = self.integration_mappings[network_name]
                
                for oscillator_name, oscillator_output in network_outputs.items():
                    if oscillator_name in mappings:
                        mapping = mappings[oscillator_name]
                        actuator_group = mapping["actuator_group"]
                        scaling = mapping["scaling"]
                        
                        if actuator_group not in actuator_commands:
                            actuator_commands[actuator_group] = {}
                        
                        # Apply scaling to oscillator output
                        actuator_commands[actuator_group]["command"] = oscillator_output * scaling
        
        return actuator_commands
    
    def _apply_actuator_commands(self, actuator_commands: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Apply actuator commands and return actuator states."""
        actuator_states = {}
        
        for group_name, commands in actuator_commands.items():
            # Simulate actuator response (simplified)
            actuator_states[group_name] = {
                "position": commands.get("command", 0.0),
                "velocity": commands.get("command", 0.0) * 0.1,  # Simplified velocity
                "force": commands.get("command", 0.0) * 0.5      # Simplified force
            }
        
        return actuator_states
    
    def configure_wing_flapping(self, frequency: float, amplitude: float) -> bool:
        """
        Configure wing flapping parameters.
        
        Args:
            frequency: Flapping frequency in Hz
            amplitude: Flapping amplitude (0.0-1.0)
            
        Returns:
            Success status
        """
        if not self.initialized or not self.cpg_controller:
            logger.error("Cannot configure uninitialized system")
            return False
        
        try:
            # Configure CPG network parameters
            self.cpg_controller.set_network_parameters(
                "wing_flapping",
                {
                    "frequency": frequency,
                    "amplitude": amplitude,
                    "coupling_strength": 0.2,
                    "phase_bias": 0.0
                }
            )
            
            logger.info(f"Wing flapping configured: frequency={frequency}Hz, amplitude={amplitude}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure wing flapping: {e}")
            return False
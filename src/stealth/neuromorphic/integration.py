#!/usr/bin/env python3
"""
Integration module for neuromorphic stealth systems.

This module provides integration between neuromorphic controllers
and stealth systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, Any, List, Optional
import numpy as np

from src.stealth.neuromorphic.controller import StealthNeuromorphicController, AdaptationStrategy
from src.stealth.neuromorphic.adaptive_stealth import AdaptiveStealthSystem
from src.stealth.base.interfaces import StealthInterface, NeuromorphicStealth, StealthType
from src.stealth.base.config import StealthSystemConfig


class NeuromorphicStealthIntegration:
    """Integration for neuromorphic stealth systems."""
    
    def __init__(self, adaptation_strategy: AdaptationStrategy = AdaptationStrategy.REACTIVE):
        """Initialize neuromorphic stealth integration."""
        self.controller = StealthNeuromorphicController(adaptation_strategy)
        self.stealth_systems: Dict[str, StealthInterface] = {}
        self.neuromorphic_systems: Dict[str, NeuromorphicStealth] = {}
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize the integration."""
        self.initialized = self.controller.initialize()
        return self.initialized
        
    def register_stealth_system(self, system_id: str, system: StealthInterface) -> bool:
        """Register a stealth system with the integration."""
        if system_id in self.stealth_systems:
            return False
            
        self.stealth_systems[system_id] = system
        
        # If system is neuromorphic, register with controller
        if isinstance(system, NeuromorphicStealth):
            self.neuromorphic_systems[system_id] = system
            self.controller.register_stealth_system(system_id, system)
            
        return True
        
    def create_adaptive_system(self, 
                             system_id: str,
                             config: StealthSystemConfig,
                             stealth_type: StealthType,
                             hardware_interface=None) -> str:
        """
        Create and register a new adaptive stealth system.
        
        Args:
            system_id: Unique identifier for the system
            config: System configuration
            stealth_type: Type of stealth technology
            hardware_interface: Interface to neuromorphic hardware
            
        Returns:
            System ID if successful, empty string otherwise
        """
        if system_id in self.stealth_systems:
            return ""
            
        # Create new adaptive system
        system = AdaptiveStealthSystem(config, stealth_type, hardware_interface)
        
        # Initialize the system
        if not system.initialize():
            return ""
            
        # Register with integration
        if not self.register_stealth_system(system_id, system):
            return ""
            
        return system_id
        
    def update(self, 
              sensor_data: Dict[str, Any],
              threat_data: Dict[str, Any],
              dt: float) -> Dict[str, Dict[str, Any]]:
        """
        Update stealth systems using neuromorphic adaptation.
        
        Args:
            sensor_data: Current sensor readings
            threat_data: Detected threats and their characteristics
            dt: Time step in seconds
            
        Returns:
            Adaptations applied to each stealth system
        """
        if not self.initialized:
            return {}
            
        # Process through controller
        adaptations = self.controller.process_cycle(sensor_data, threat_data, dt)
        
        # Apply adaptations to non-neuromorphic systems
        for system_id, system in self.stealth_systems.items():
            if system_id in adaptations and system_id not in self.neuromorphic_systems:
                self._apply_adaptations(system_id, system, adaptations[system_id])
                
        return adaptations
        
    def _apply_adaptations(self, 
                         system_id: str,
                         system: StealthInterface,
                         adaptations: Dict[str, Any]) -> None:
        """Apply adaptations to a stealth system."""
        # Get current status
        status = system.get_status()
        
        # Apply power level adaptation
        if "power_level" in adaptations:
            status["power_level"] = adaptations["power_level"]
            
        # Apply mode changes
        if "mode" in adaptations:
            status["mode"] = adaptations["mode"]
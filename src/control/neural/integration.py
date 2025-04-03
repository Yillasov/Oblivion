#!/usr/bin/env python3
"""
Integration between SNN flight controllers and airframes.
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

from typing import Dict, Any, Optional
import numpy as np

from src.airframe.base import AirframeBase
from src.airframe.neuromorphic_integration import AirframeNeuromorphicAdapter
from .snn_flight_control import SNNFlightController, SNNControllerFactory

class SNNAirframeIntegration:
    
    
    def __init__(self, airframe: AirframeBase, controller_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integration between an airframe and SNN controller.
        
        Args:
            airframe: The airframe instance
            controller_config: Configuration for the SNN controller
        """
        self.airframe = airframe
        self.airframe_type = airframe.__class__.__name__.lower().replace("drone", "")
        
        # Create the appropriate adapter and controller
        self.adapter = AirframeNeuromorphicAdapter(self.airframe_type)
        
        if controller_config is None:
            controller_config = {}
        
        self.controller = SNNControllerFactory.create_controller(
            self.airframe_type, controller_config
        )
        
        self.last_sensor_data = {}
        self.last_control_outputs = {}
    
    def update(self, airframe_state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Update the SNN controller with the current airframe state.
        
        Args:
            airframe_state: Current state of the airframe
            dt: Time step in seconds
            
        Returns:
            Control commands for the airframe
        """
        # Get sensor data from the airframe state
        sensor_data = self.adapter.get_sensor_data(airframe_state)
        self.last_sensor_data = sensor_data
        
        # Process through the SNN controller
        control_outputs = self.controller.update(sensor_data, dt)
        self.last_control_outputs = control_outputs
        
        # Convert controller outputs to airframe commands
        control_commands = self.adapter.apply_control_commands(
            control_outputs, airframe_state
        )
        
        return control_commands
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get the current state of the SNN network."""
        return {
            "neuron_layers": self.controller.neuron_layers,
            "last_sensor_data": self.last_sensor_data,
            "last_control_outputs": self.last_control_outputs
        }
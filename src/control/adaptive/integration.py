#!/usr/bin/env python3
"""
Integration between adaptive controllers and neuromorphic hardware.
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

# Fix the import path for the neuromorphic hardware integration
from src.core.integration.neuromorphic_system import NeuromorphicSystem
# Change relative import to absolute import
from src.control.adaptive.flight_conditions import AdaptiveController, AdaptiveControllerFactory

class AdaptiveNeuromorphicControl:
    
    
    def __init__(self, 
                hardware_integration: NeuromorphicSystem,
                airframe_type: str,
                adaptive_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integration between adaptive control and neuromorphic hardware.
        
        Args:
            hardware_integration: The neuromorphic hardware integration
            airframe_type: Type of airframe
            adaptive_config: Configuration for the adaptive controller
        """
        self.hardware_integration = hardware_integration
        
        if adaptive_config is None:
            adaptive_config = {}
        
        self.adaptive_controller = AdaptiveControllerFactory.create_controller(
            airframe_type, adaptive_config
        )
        
        self.performance_metrics = {
            "tracking_error": 0.0,
            "energy_efficiency": 1.0,
            "stability_margin": 1.0
        }
    
    def update_performance_metrics(self, 
                                  sensor_data: Dict[str, np.ndarray],
                                  control_outputs: Dict[str, np.ndarray],
                                  reference_commands: Dict[str, np.ndarray]) -> None:
        """Update performance metrics based on current state and commands."""
        # Calculate tracking error
        error = 0.0
        count = 0
        
        for key, value in reference_commands.items():
            if key in control_outputs:
                error += np.sum(np.abs(value - control_outputs[key]))
                count += value.size
        
        if count > 0:
            self.performance_metrics["tracking_error"] = error / count
        
        # Calculate energy efficiency based on hardware power usage
        # Use a fixed value for power usage since we can't access it directly
        power_usage = 1.0  # Default power usage value
        
        # Try to get power information from the process_data method if available
        try:
            power_info = self.hardware_integration.process_data({
                "operation": "get_power_info"
            })
            if power_info and "power_consumption" in power_info:
                power_usage = power_info["power_consumption"]
        except:
            # If we can't get power info, use the default value
            pass
            
        self.performance_metrics["energy_efficiency"] = 1.0 / (1.0 + power_usage)
        
        # Simple stability margin calculation
        if "roll_rate" in sensor_data and "pitch_rate" in sensor_data:
            stability = 1.0 / (1.0 + np.sum(np.abs(sensor_data["roll_rate"])) + 
                              np.sum(np.abs(sensor_data["pitch_rate"])))
            self.performance_metrics["stability_margin"] = stability
    
    def update(self, 
              sensor_data: Dict[str, np.ndarray],
              reference_commands: Dict[str, np.ndarray],
              dt: float) -> Dict[str, np.ndarray]:
        """
        Update using adaptive control and neuromorphic hardware.
        
        Args:
            sensor_data: Sensor data for processing
            reference_commands: Reference commands for tracking
            dt: Time step in seconds
            
        Returns:
            Adapted control outputs
        """
        # Use the correct method to process data with NeuromorphicSystem
        process_result = self.hardware_integration.process_data({
            "sensor_data": sensor_data,
            "dt": dt,
            "operation": "control_update"
        })
        
        # Extract control outputs from the processing result
        control_outputs = process_result.get("control_outputs", {})
        if not control_outputs:
            # Fallback if no control outputs are returned
            control_outputs = {key: np.zeros_like(value) for key, value in reference_commands.items()}
        
        # Update performance metrics
        self.update_performance_metrics(sensor_data, control_outputs, reference_commands)
        
        # Adapt control parameters based on performance
        self.adaptive_controller.adapt_parameters(sensor_data, self.performance_metrics)
        
        # Apply adaptation to control outputs
        adapted_outputs = self.adaptive_controller.apply_adaptation(control_outputs)
        
        return adapted_outputs
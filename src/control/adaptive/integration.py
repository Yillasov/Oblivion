from typing import Dict, Any, Optional
import numpy as np

from src.hardware.neuromorphic.integration import HardwareSNNIntegration
from .flight_conditions import AdaptiveController, AdaptiveControllerFactory

class AdaptiveNeuromorphicControl:
    """Integration between adaptive controllers and neuromorphic hardware."""
    
    def __init__(self, 
                hardware_integration: HardwareSNNIntegration,
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
        power_usage = self.hardware_integration.get_power_usage()
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
        # Get control outputs from neuromorphic hardware
        control_outputs = self.hardware_integration.update(sensor_data, dt)
        
        # Update performance metrics
        self.update_performance_metrics(sensor_data, control_outputs, reference_commands)
        
        # Adapt control parameters based on performance
        self.adaptive_controller.adapt_parameters(sensor_data, self.performance_metrics)
        
        # Apply adaptation to control outputs
        adapted_outputs = self.adaptive_controller.apply_adaptation(control_outputs)
        
        return adapted_outputs
from typing import Dict, Any, Optional
import numpy as np

from src.control.neural.snn_flight_control import SNNFlightController
from .abstraction import NeuromorphicHardware, NeuromorphicHardwareFactory

class HardwareSNNIntegration:
    """Integration between SNN controllers and neuromorphic hardware."""
    
    def __init__(self, 
                controller: SNNFlightController, 
                hardware_type: str,
                hardware_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integration between an SNN controller and neuromorphic hardware.
        
        Args:
            controller: The SNN controller
            hardware_type: Type of neuromorphic hardware to use
            hardware_config: Configuration for the hardware
        """
        self.controller = controller
        
        if hardware_config is None:
            hardware_config = {}
        
        self.hardware = NeuromorphicHardwareFactory.create_interface(
            hardware_type, hardware_config
        )
        
        # Initialize the hardware with the controller's network configuration
        network_config = {
            "num_neurons": sum(layer.size for layer in controller.neuron_layers.values()),
            "num_layers": len(controller.neuron_layers),
            "layer_sizes": {name: layer.size for name, layer in controller.neuron_layers.items()}
        }
        
        self.hardware.initialize(network_config)
        
        # Load the controller's weights onto the hardware
        self.hardware.load_weights(controller.synaptic_weights)
    
    def run_inference(self, sensor_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on the neuromorphic hardware.
        
        Args:
            sensor_data: Sensor data for inference
            
        Returns:
            Control outputs from the hardware
        """
        return self.hardware.run_inference(sensor_data)
    
    def get_power_usage(self) -> float:
        """Get current power usage of the neuromorphic hardware."""
        return self.hardware.get_power_usage()
    
    def update(self, sensor_data: Dict[str, np.ndarray], dt: float) -> Dict[str, np.ndarray]:
        """
        Update using the neuromorphic hardware.
        
        Args:
            sensor_data: Sensor data for processing
            dt: Time step in seconds
            
        Returns:
            Control outputs from the hardware
        """
        # For hardware that supports direct inference
        hardware_outputs = self.run_inference(sensor_data)
        
        # If hardware doesn't provide complete outputs, fall back to software
        if not hardware_outputs:
            return self.controller.update(sensor_data, dt)
        
        return hardware_outputs
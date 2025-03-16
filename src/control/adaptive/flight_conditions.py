from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum

class FlightCondition(Enum):
    TAKEOFF = "takeoff"
    CRUISE = "cruise"
    COMBAT = "combat"
    STEALTH = "stealth"
    HYPERSONIC = "hypersonic"
    LANDING = "landing"
    EMERGENCY = "emergency"
    SPACE_TRANSITION = "space_transition"
    UNDERWATER_LAUNCH = "underwater_launch"

class AdaptiveController:
    """Base class for adaptive flight control algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_condition = FlightCondition.CRUISE
        self.adaptation_rate = config.get("adaptation_rate", 0.1)
        self.control_parameters = {}
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize control parameters for different flight conditions."""
        self.control_parameters = {
            FlightCondition.TAKEOFF: {"gain": 1.2, "damping": 0.8},
            FlightCondition.CRUISE: {"gain": 1.0, "damping": 0.7},
            FlightCondition.COMBAT: {"gain": 1.5, "damping": 0.5},
            FlightCondition.STEALTH: {"gain": 0.8, "damping": 0.9},
            FlightCondition.HYPERSONIC: {"gain": 2.0, "damping": 0.4},
            FlightCondition.LANDING: {"gain": 1.1, "damping": 1.0},
            FlightCondition.EMERGENCY: {"gain": 2.0, "damping": 1.2},
            FlightCondition.SPACE_TRANSITION: {"gain": 1.8, "damping": 0.3},
            FlightCondition.UNDERWATER_LAUNCH: {"gain": 1.3, "damping": 0.6}
        }
    
    def detect_flight_condition(self, sensor_data: Dict[str, np.ndarray]) -> FlightCondition:
        """Detect current flight condition based on sensor data."""
        # Simple detection logic - would be more sophisticated in real implementation
        if "altitude" in sensor_data:
            altitude = sensor_data["altitude"][0]
            if altitude < 10:
                if "vertical_speed" in sensor_data and sensor_data["vertical_speed"][0] > 0:
                    return FlightCondition.TAKEOFF
                else:
                    return FlightCondition.LANDING
            elif altitude > 80000:
                return FlightCondition.SPACE_TRANSITION
        
        if "airspeed" in sensor_data:
            airspeed = sensor_data["airspeed"][0]
            if airspeed > 5:  # Mach 5+
                return FlightCondition.HYPERSONIC
        
        if "depth" in sensor_data and sensor_data["depth"][0] > 0:
            return FlightCondition.UNDERWATER_LAUNCH
        
        if "radar_warning" in sensor_data and sensor_data["radar_warning"][0] > 0.5:
            return FlightCondition.STEALTH
        
        if "target_tracking" in sensor_data and sensor_data["target_tracking"][0] > 0.5:
            return FlightCondition.COMBAT
        
        return FlightCondition.CRUISE
    
    def adapt_parameters(self, sensor_data: Dict[str, np.ndarray], 
                         performance_metrics: Dict[str, float]) -> None:
        """Adapt control parameters based on performance metrics."""
        # Simple adaptation logic
        condition = self.detect_flight_condition(sensor_data)
        self.current_condition = condition
        
        # Adapt parameters based on performance
        if "tracking_error" in performance_metrics:
            error = performance_metrics["tracking_error"]
            self.control_parameters[condition]["gain"] += self.adaptation_rate * error
            self.control_parameters[condition]["damping"] -= self.adaptation_rate * error * 0.5
    
    def get_control_parameters(self) -> Dict[str, float]:
        """Get current control parameters."""
        return self.control_parameters[self.current_condition]
    
    def apply_adaptation(self, control_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply adaptation to control outputs."""
        params = self.get_control_parameters()
        adapted_outputs = {}
        
        for key, value in control_outputs.items():
            adapted_outputs[key] = value * params["gain"]
        
        return adapted_outputs

class MorphingWingAdaptiveController(AdaptiveController):
    """Adaptive controller for morphing wing drones."""
    
    def initialize_parameters(self):
        super().initialize_parameters()
        # Add morphing-specific parameters
        for condition in FlightCondition:
            self.control_parameters[condition]["morph_rate"] = 0.5
            self.control_parameters[condition]["morph_extent"] = 0.7
    
    def apply_adaptation(self, control_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        params = self.get_control_parameters()
        adapted_outputs = super().apply_adaptation(control_outputs)
        
        # Apply morphing-specific adaptations
        if "wing_morphing" in adapted_outputs:
            adapted_outputs["wing_morphing"] *= params["morph_extent"]
        
        return adapted_outputs

class HypersonicAdaptiveController(AdaptiveController):
    """Adaptive controller for hypersonic flight."""
    
    def initialize_parameters(self):
        super().initialize_parameters()
        # Add hypersonic-specific parameters
        for condition in FlightCondition:
            self.control_parameters[condition]["thermal_management"] = 0.5
            self.control_parameters[condition]["control_authority"] = 0.8
    
    def apply_adaptation(self, control_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        params = self.get_control_parameters()
        adapted_outputs = super().apply_adaptation(control_outputs)
        
        # Apply hypersonic-specific adaptations
        if "thermal_management" in adapted_outputs:
            adapted_outputs["thermal_management"] *= params["thermal_management"]
        
        if "flight_surfaces" in adapted_outputs:
            adapted_outputs["flight_surfaces"] *= params["control_authority"]
        
        return adapted_outputs

class AdaptiveControllerFactory:
    """Factory for creating adaptive controllers."""
    
    @staticmethod
    def create_controller(airframe_type: str, config: Dict[str, Any]) -> AdaptiveController:
        """Create an adaptive controller for the specified airframe type."""
        if airframe_type == "morphing_wing":
            return MorphingWingAdaptiveController(config)
        elif airframe_type == "hypersonic":
            return HypersonicAdaptiveController(config)
        else:
            return AdaptiveController(config)
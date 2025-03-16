from typing import Dict, Any, Optional
import numpy as np

from src.control.adaptive.integration import AdaptiveNeuromorphicControl
from .realtime import DecisionMakingSystem, Decision

class DecisionControlIntegration:
    """Integration between decision-making and adaptive control systems."""
    
    def __init__(self, 
                adaptive_control: AdaptiveNeuromorphicControl,
                decision_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integration between decision-making and adaptive control.
        
        Args:
            adaptive_control: The adaptive control system
            decision_config: Configuration for the decision-making system
        """
        self.adaptive_control = adaptive_control
        
        if decision_config is None:
            decision_config = {}
        
        self.decision_system = DecisionMakingSystem(decision_config)
        
        # Command mapping functions
        self.command_mappings = {
            "evasive_maneuver": self._map_evasive_maneuver,
            "activate_countermeasures": self._map_activate_countermeasures,
            "proceed_to_target": self._map_proceed_to_target,
            "avoid_obstacle": self._map_avoid_obstacle,
            "return_to_base": self._map_return_to_base
        }
    
    def _map_evasive_maneuver(self, 
                             sensor_data: Dict[str, np.ndarray],
                             mission_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Map evasive maneuver decision to control commands."""
        # Generate aggressive evasive maneuver commands
        return {
            "roll_command": np.array([1.0]),
            "pitch_command": np.array([0.8]),
            "throttle_command": np.array([1.0])
        }
    
    def _map_activate_countermeasures(self, 
                                     sensor_data: Dict[str, np.ndarray],
                                     mission_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Map countermeasure activation to control commands."""
        return {
            "countermeasure_command": np.array([1.0]),
            "roll_command": np.array([0.2]),
            "pitch_command": np.array([0.1])
        }
    
    def _map_proceed_to_target(self, 
                              sensor_data: Dict[str, np.ndarray],
                              mission_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Map proceed to target decision to control commands."""
        current_position = sensor_data.get("position", np.zeros(3))
        target_position = mission_params.get("target_position", np.zeros(3))
        
        # Calculate direction to target
        direction = target_position - current_position
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        # Simple commands to move toward target
        return {
            "heading_command": np.array([np.arctan2(direction[1], direction[0])]),
            "altitude_command": np.array([target_position[2]]),
            "throttle_command": np.array([0.7])
        }
    
    def _map_avoid_obstacle(self, 
                           sensor_data: Dict[str, np.ndarray],
                           mission_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Map obstacle avoidance to control commands."""
        # Simple obstacle avoidance commands
        return {
            "roll_command": np.array([0.5]),
            "pitch_command": np.array([0.3]),
            "altitude_command": np.array([sensor_data.get("altitude", np.array([100.0]))[0] + 20.0])
        }
    
    def _map_return_to_base(self, 
                           sensor_data: Dict[str, np.ndarray],
                           mission_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Map return to base decision to control commands."""
        current_position = sensor_data.get("position", np.zeros(3))
        base_position = mission_params.get("base_position", np.zeros(3))
        
        # Calculate direction to base
        direction = base_position - current_position
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        # Commands to return to base
        return {
            "heading_command": np.array([np.arctan2(direction[1], direction[0])]),
            "altitude_command": np.array([base_position[2]]),
            "throttle_command": np.array([0.6])
        }
    
    def update(self, 
              sensor_data: Dict[str, np.ndarray],
              mission_params: Dict[str, Any],
              dt: float) -> Dict[str, np.ndarray]:
        """
        Update using decision-making and adaptive control.
        
        Args:
            sensor_data: Sensor data for processing
            mission_params: Mission parameters
            dt: Time step in seconds
            
        Returns:
            Control outputs
        """
        # Get decisions from decision-making system
        decisions = self.decision_system.update(sensor_data, mission_params)
        
        # Get highest priority decision
        highest_decision = self.decision_system.get_highest_priority_decision(decisions)
        
        # Map decision to reference commands
        reference_commands = {}
        if highest_decision and highest_decision.value in self.command_mappings:
            mapping_func = self.command_mappings[highest_decision.value]
            reference_commands = mapping_func(sensor_data, mission_params)
        
        # Update adaptive control with reference commands
        control_outputs = self.adaptive_control.update(
            sensor_data, reference_commands, dt
        )
        
        return control_outputs
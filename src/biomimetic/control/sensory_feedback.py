#!/usr/bin/env python3
"""
Sensory feedback integration for biomimetic CPG controllers.
Enables adaptive locomotion based on environmental stimuli.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.biomimetic.control.cpg_models import BiomimeticCPGController, CPGParameters, CPGType
from src.simulation.sensors.sensor_framework import SensorType

logger = get_logger("sensory_feedback")


class FeedbackType(Enum):
    """Types of sensory feedback for CPG modulation."""
    PHASE_MODULATION = "phase"        # Modifies oscillator phase
    FREQUENCY_MODULATION = "frequency"  # Modifies oscillator frequency
    AMPLITUDE_MODULATION = "amplitude"  # Modifies oscillator amplitude
    COUPLING_MODULATION = "coupling"  # Modifies coupling between oscillators
    RESET_TRIGGER = "reset"           # Triggers oscillator reset
    GAIT_TRANSITION = "gait"          # Triggers gait transition


@dataclass
class FeedbackMapping:
    """Mapping between sensor data and CPG parameters."""
    sensor_type: SensorType           # Type of sensor providing feedback
    sensor_key: str                   # Key in sensor data dictionary
    feedback_type: FeedbackType       # Type of feedback to apply
    target_network: str               # Target CPG network name
    target_oscillators: List[int]     # Target oscillator indices (empty for all)
    scaling_factor: float = 1.0       # Scaling factor for feedback
    threshold: float = 0.0            # Threshold for feedback activation
    adaptation_rate: float = 0.1      # Rate of adaptation to feedback
    custom_function: Optional[Callable] = None  # Custom feedback function


class SensoryFeedbackIntegration:
    """Integrates sensory feedback with CPG controllers."""
    
    def __init__(self, cpg_controller: BiomimeticCPGController):
        """
        Initialize sensory feedback integration.
        
        Args:
            cpg_controller: The CPG controller to integrate with
        """
        self.cpg_controller = cpg_controller
        self.feedback_mappings: List[FeedbackMapping] = []
        self.sensor_history: Dict[str, List[np.ndarray]] = {}
        self.history_length = 10  # Number of sensor readings to keep
        logger.info("Initialized sensory feedback integration")
    
    def add_feedback_mapping(self, mapping: FeedbackMapping) -> bool:
        """
        Add a feedback mapping between sensor data and CPG parameters.
        
        Args:
            mapping: The feedback mapping to add
            
        Returns:
            Success flag
        """
        # Validate mapping
        if mapping.target_network not in self.cpg_controller.cpg_networks:
            logger.warning(f"Target network {mapping.target_network} does not exist")
            return False
        
        # Add mapping
        self.feedback_mappings.append(mapping)
        logger.info(f"Added feedback mapping: {mapping.sensor_type.value} -> {mapping.feedback_type.value}")
        return True
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> None:
        """
        Process sensor data and apply feedback to CPG controller.
        
        Args:
            sensor_data: Dictionary of sensor data
        """
        # Update sensor history
        for key, value in sensor_data.items():
            if key not in self.sensor_history:
                self.sensor_history[key] = []
            
            self.sensor_history[key].append(value)
            if len(self.sensor_history[key]) > self.history_length:
                self.sensor_history[key].pop(0)
        
        # Process each feedback mapping
        for mapping in self.feedback_mappings:
            # Check if sensor data is available
            if mapping.sensor_key not in sensor_data:
                continue
            
            # Get sensor value
            sensor_value = sensor_data[mapping.sensor_key]
            
            # Apply threshold
            if abs(sensor_value) < mapping.threshold:
                continue
            
            # Get target network
            network = self.cpg_controller.cpg_networks.get(mapping.target_network)
            if not network:
                continue
            
            # Determine target oscillators
            target_oscillators = mapping.target_oscillators
            if not target_oscillators:
                target_oscillators = list(range(network.num_oscillators))
            
            # Apply custom function if provided
            if mapping.custom_function:
                mapping.custom_function(network, sensor_value, target_oscillators)
                continue
            
            # Apply feedback based on type
            self._apply_feedback(mapping, network, sensor_value, target_oscillators)
    
    def _apply_feedback(self, mapping: FeedbackMapping, network, sensor_value, target_oscillators):
        """Apply feedback to CPG network based on mapping type."""
        scaled_value = sensor_value * mapping.scaling_factor
        
        if mapping.feedback_type == FeedbackType.FREQUENCY_MODULATION:
            # Modulate frequency
            current_freq = network.oscillators[0].params.frequency
            new_freq = current_freq + scaled_value * mapping.adaptation_rate
            new_freq = max(0.1, new_freq)  # Ensure positive frequency
            network.set_frequency(new_freq)
            
        elif mapping.feedback_type == FeedbackType.AMPLITUDE_MODULATION:
            # Modulate amplitude
            current_amp = network.oscillators[0].params.amplitude
            new_amp = current_amp + scaled_value * mapping.adaptation_rate
            new_amp = max(0.0, new_amp)  # Ensure non-negative amplitude
            network.set_amplitude(new_amp)
            
        elif mapping.feedback_type == FeedbackType.PHASE_MODULATION:
            # Modulate phase (only for Kuramoto oscillators)
            if network.cpg_type == CPGType.KURAMOTO_OSCILLATOR:
                for i in target_oscillators:
                    if i < len(network.oscillators[0].state):
                        network.oscillators[0].state[i] += scaled_value
            
        elif mapping.feedback_type == FeedbackType.COUPLING_MODULATION:
            # Modulate coupling strength
            for osc in network.oscillators:
                osc.params.coupling_strength += scaled_value * mapping.adaptation_rate
                osc.params.coupling_strength = max(0.0, min(1.0, osc.params.coupling_strength))
            
        elif mapping.feedback_type == FeedbackType.RESET_TRIGGER:
            # Reset oscillators if value exceeds threshold
            if abs(scaled_value) > mapping.threshold:
                for i in target_oscillators:
                    if i < len(network.oscillators):
                        network.oscillators[i].reset()
            
        elif mapping.feedback_type == FeedbackType.GAIT_TRANSITION:
            # Trigger gait transition (implemented in derived classes)
            pass
    
    def create_obstacle_avoidance_feedback(self, 
                                          sensor_key: str = "obstacle_distance", 
                                          target_network: str = "walking_gait") -> FeedbackMapping:
        """
        Create a feedback mapping for obstacle avoidance.
        
        Args:
            sensor_key: Key for obstacle distance in sensor data
            target_network: Target CPG network name
            
        Returns:
            FeedbackMapping for obstacle avoidance
        """
        return FeedbackMapping(
            sensor_type=SensorType.LIDAR,
            sensor_key=sensor_key,
            feedback_type=FeedbackType.FREQUENCY_MODULATION,
            target_network=target_network,
            target_oscillators=[],  # All oscillators
            scaling_factor=-0.5,    # Slow down when obstacles are near
            threshold=2.0,          # Only respond to close obstacles
            adaptation_rate=0.2
        )
    
    def create_terrain_adaptation_feedback(self,
                                          sensor_key: str = "terrain_roughness",
                                          target_network: str = "walking_gait") -> FeedbackMapping:
        """
        Create a feedback mapping for terrain adaptation.
        
        Args:
            sensor_key: Key for terrain roughness in sensor data
            target_network: Target CPG network name
            
        Returns:
            FeedbackMapping for terrain adaptation
        """
        return FeedbackMapping(
            sensor_type=SensorType.ALTIMETER,
            sensor_key=sensor_key,
            feedback_type=FeedbackType.AMPLITUDE_MODULATION,
            target_network=target_network,
            target_oscillators=[],  # All oscillators
            scaling_factor=0.3,     # Increase amplitude on rough terrain
            threshold=0.1,
            adaptation_rate=0.1
        )
    
    def create_energy_efficiency_feedback(self,
                                         sensor_key: str = "energy_level",
                                         target_network: str = "walking_gait") -> FeedbackMapping:
        """
        Create a feedback mapping for energy efficiency.
        
        Args:
            sensor_key: Key for energy level in sensor data
            target_network: Target CPG network name
            
        Returns:
            FeedbackMapping for energy efficiency
        """
        return FeedbackMapping(
            sensor_type=SensorType.POWER,
            sensor_key=sensor_key,
            feedback_type=FeedbackType.FREQUENCY_MODULATION,
            target_network=target_network,
            target_oscillators=[],  # All oscillators
            scaling_factor=-0.2,    # Slow down when energy is low
            threshold=0.3,          # Only respond when energy is below 30%
            adaptation_rate=0.05
        )


class AdaptiveLocomotionController:
    """Controller for adaptive locomotion using CPGs with sensory feedback."""
    
    def __init__(self, cpg_controller: BiomimeticCPGController):
        """
        Initialize adaptive locomotion controller.
        
        Args:
            cpg_controller: The CPG controller to use
        """
        self.cpg_controller = cpg_controller
        self.feedback_integration = SensoryFeedbackIntegration(cpg_controller)
        self.current_gait = "walking"
        self.available_gaits = ["walking", "running", "turning"]
        logger.info("Initialized adaptive locomotion controller")
    
    def initialize(self) -> bool:
        """
        Initialize the controller.
        
        Returns:
            Success flag
        """
        # Initialize CPG controller
        if not self.cpg_controller.initialized:
            self.cpg_controller.initialize()
        
        # Add default feedback mappings
        self._setup_default_feedback_mappings()
        
        return True
    
    def _setup_default_feedback_mappings(self) -> None:
        """Set up default feedback mappings."""
        # Add obstacle avoidance feedback
        obstacle_feedback = self.feedback_integration.create_obstacle_avoidance_feedback()
        self.feedback_integration.add_feedback_mapping(obstacle_feedback)
        
        # Add terrain adaptation feedback
        terrain_feedback = self.feedback_integration.create_terrain_adaptation_feedback()
        self.feedback_integration.add_feedback_mapping(terrain_feedback)
        
        # Add energy efficiency feedback
        energy_feedback = self.feedback_integration.create_energy_efficiency_feedback()
        self.feedback_integration.add_feedback_mapping(energy_feedback)
    
    def step(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advance the controller by one time step.
        
        Args:
            sensor_data: Dictionary of sensor data
            
        Returns:
            Control outputs
        """
        # Process sensor data for feedback
        self.feedback_integration.process_sensor_data(sensor_data)
        
        # Step CPG controller
        control_outputs = self.cpg_controller.step(sensor_data)
        
        # Check for gait transitions
        self._check_gait_transition(sensor_data)
        
        return control_outputs
    
    def _check_gait_transition(self, sensor_data: Dict[str, Any]) -> None:
        """Check if a gait transition is needed based on sensor data."""
        # Example: transition to running gait when velocity is high
        if "velocity" in sensor_data:
            velocity = sensor_data["velocity"]
            
            if self.current_gait == "walking" and velocity > 5.0:
                self._transition_to_gait("running")
            elif self.current_gait == "running" and velocity < 3.0:
                self._transition_to_gait("walking")
    
    def _transition_to_gait(self, gait_name: str) -> None:
        """
        Transition to a different gait.
        
        Args:
            gait_name: Name of the gait to transition to
        """
        if gait_name not in self.available_gaits:
            logger.warning(f"Gait {gait_name} is not available")
            return
        
        if gait_name == self.current_gait:
            return
        
        logger.info(f"Transitioning from {self.current_gait} to {gait_name} gait")
        
        # Update gait-specific parameters
        if gait_name == "walking":
            # Walking gait: moderate frequency, moderate amplitude
            for network in self.cpg_controller.cpg_networks.values():
                network.set_frequency(1.0)
                network.set_amplitude(1.0)
                
        elif gait_name == "running":
            # Running gait: higher frequency, higher amplitude
            for network in self.cpg_controller.cpg_networks.values():
                network.set_frequency(2.0)
                network.set_amplitude(1.5)
                
        elif gait_name == "turning":
            # Turning gait: asymmetric amplitudes
            for name, network in self.cpg_controller.cpg_networks.items():
                if name == "walking_gait" and len(network.oscillators) >= 4:
                    # Reduce amplitude on one side
                    network.oscillators[0].params.amplitude = 0.5
                    network.oscillators[2].params.amplitude = 0.5
                    network.oscillators[1].params.amplitude = 1.0
                    network.oscillators[3].params.amplitude = 1.0
        
        self.current_gait = gait_name


# Example usage function
def create_adaptive_locomotion_controller(controller_type: str) -> AdaptiveLocomotionController:
    """
    Create an adaptive locomotion controller.
    
    Args:
        controller_type: Type of controller (e.g., "quadruped", "hexapod")
        
    Returns:
        Configured AdaptiveLocomotionController
    """
    from src.biomimetic.control.cpg_models import create_cpg_controller
    
    # Create CPG controller
    cpg_controller = create_cpg_controller(controller_type)
    
    # Create adaptive controller
    adaptive_controller = AdaptiveLocomotionController(cpg_controller)
    adaptive_controller.initialize()
    
    return adaptive_controller
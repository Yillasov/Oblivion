#!/usr/bin/env python3
"""
Biomimetic actuator models for neuromorphic hardware integration.
Provides muscle-like actuators for biomimetic movement control.
"""

import os
import sys
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.landing_gear.hardware_interface import NeuromorphicHardwareInterface

logger = get_logger("biomimetic_actuators")


class ActuatorType(Enum):
    """Types of biomimetic actuators."""
    MUSCLE_FIBER = "muscle_fiber"
    TENDON = "tendon"
    HYDRAULIC = "hydraulic"
    PNEUMATIC = "pneumatic"
    SMA_WIRE = "shape_memory_alloy"
    EAP = "electroactive_polymer"
    HYBRID = "hybrid"


@dataclass
class ActuatorProperties:
    """Properties of a biomimetic actuator."""
    max_force: float  # Newtons
    contraction_ratio: float  # % of length
    response_time: float  # seconds
    energy_efficiency: float  # 0-1
    fatigue_rate: float  # 0-1 per hour
    recovery_rate: float  # 0-1 per hour
    mass: float  # kg
    biological_inspiration: str


class BiomimeticActuator:
    """Base class for biomimetic actuators."""
    
    def __init__(self, actuator_id: str, actuator_type: ActuatorType, properties: ActuatorProperties):
        """
        Initialize a biomimetic actuator.
        
        Args:
            actuator_id: Unique identifier
            actuator_type: Type of actuator
            properties: Actuator properties
        """
        self.actuator_id = actuator_id
        self.actuator_type = actuator_type
        self.properties = properties
        self.activation = 0.0  # Current activation level (0-1)
        self.current_length = 1.0  # Normalized length (1.0 = rest length)
        self.current_force = 0.0  # Current force output (N)
        self.fatigue = 0.0  # Current fatigue level (0-1)
        self.temperature = 20.0  # Current temperature (°C)
        self.last_update_time = 0.0  # Last update time
        
        logger.info(f"Initialized {actuator_type.value} actuator: {actuator_id}")
    
    def activate(self, level: float, dt: float) -> Dict[str, float]:
        """
        Activate the actuator.
        
        Args:
            level: Activation level (0-1)
            dt: Time step in seconds
            
        Returns:
            Current actuator state
        """
        # Clamp activation level
        self.activation = max(0.0, min(1.0, level))
        
        # Calculate contraction based on activation and fatigue
        effective_activation = self.activation * (1.0 - self.fatigue)
        target_contraction = effective_activation * self.properties.contraction_ratio
        
        # Apply response time dynamics
        response_factor = min(dt / self.properties.response_time, 1.0)
        current_contraction = (1.0 - self.current_length)
        new_contraction = current_contraction + response_factor * (target_contraction - current_contraction)
        
        # Update length
        self.current_length = 1.0 - new_contraction
        
        # Calculate force output (simplified model)
        self.current_force = effective_activation * self.properties.max_force
        
        # Update fatigue
        self.fatigue += self.activation * self.properties.fatigue_rate * dt
        self.fatigue = max(0.0, min(1.0, self.fatigue))
        
        # Apply recovery when not fully activated
        recovery = (1.0 - self.activation) * self.properties.recovery_rate * dt
        self.fatigue = max(0.0, self.fatigue - recovery)
        
        # Update temperature (simplified model)
        temp_increase = self.activation * 5.0 * dt  # 5°C per second at full activation
        self.temperature += temp_increase
        temp_decrease = (40.0 - self.temperature) * 0.1 * dt  # Cooling proportional to difference
        self.temperature += temp_decrease
        
        # Update time
        self.last_update_time += dt
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, float]:
        """Get current actuator state."""
        return {
            "activation": self.activation,
            "length": self.current_length,
            "force": self.current_force,
            "fatigue": self.fatigue,
            "temperature": self.temperature
        }
    
    def reset(self) -> None:
        """Reset actuator to initial state."""
        self.activation = 0.0
        self.current_length = 1.0
        self.current_force = 0.0
        self.fatigue = 0.0
        self.temperature = 20.0


class MuscleFiberActuator(BiomimeticActuator):
    """Muscle fiber actuator model."""
    
    def __init__(self, actuator_id: str, fiber_type: str = "fast_twitch"):
        """
        Initialize a muscle fiber actuator.
        
        Args:
            actuator_id: Unique identifier
            fiber_type: Type of muscle fiber (fast_twitch or slow_twitch)
        """
        # Set properties based on fiber type
        if fiber_type == "fast_twitch":
            properties = ActuatorProperties(
                max_force=2.5,
                contraction_ratio=0.3,
                response_time=0.05,
                energy_efficiency=0.6,
                fatigue_rate=0.2,
                recovery_rate=0.1,
                mass=0.01,
                biological_inspiration="vertebrate_fast_twitch_muscle"
            )
        else:  # slow_twitch
            properties = ActuatorProperties(
                max_force=1.8,
                contraction_ratio=0.25,
                response_time=0.12,
                energy_efficiency=0.8,
                fatigue_rate=0.05,
                recovery_rate=0.15,
                mass=0.01,
                biological_inspiration="vertebrate_slow_twitch_muscle"
            )
        
        super().__init__(actuator_id, ActuatorType.MUSCLE_FIBER, properties)
        self.fiber_type = fiber_type
        
        # Additional muscle-specific properties
        self.calcium_level = 0.0
        self.atp_consumption = 0.0
    
    def activate(self, level: float, dt: float) -> Dict[str, float]:
        """
        Activate the muscle fiber with calcium dynamics.
        
        Args:
            level: Activation level (0-1)
            dt: Time step in seconds
            
        Returns:
            Current actuator state
        """
        # Simulate calcium dynamics
        calcium_response_time = 0.02  # seconds
        calcium_response_factor = min(dt / calcium_response_time, 1.0)
        self.calcium_level += calcium_response_factor * (level - self.calcium_level)
        
        # Use calcium level as the actual activation
        result = super().activate(self.calcium_level, dt)
        
        # Calculate ATP consumption
        self.atp_consumption += self.activation * dt * (1.0 + self.current_force * 0.5)
        
        # Add muscle-specific state variables
        result["calcium_level"] = self.calcium_level
        result["atp_consumption"] = self.atp_consumption
        
        return result


class ElectroactivePolymerActuator(BiomimeticActuator):
    """Electroactive polymer actuator model."""
    
    def __init__(self, actuator_id: str, polymer_type: str = "dielectric"):
        """
        Initialize an electroactive polymer actuator.
        
        Args:
            actuator_id: Unique identifier
            polymer_type: Type of EAP (dielectric or ionic)
        """
        # Set properties based on polymer type
        if polymer_type == "dielectric":
            properties = ActuatorProperties(
                max_force=1.2,
                contraction_ratio=0.4,
                response_time=0.02,
                energy_efficiency=0.7,
                fatigue_rate=0.1,
                recovery_rate=0.2,
                mass=0.005,
                biological_inspiration="insect_flight_muscle"
            )
        else:  # ionic
            properties = ActuatorProperties(
                max_force=0.8,
                contraction_ratio=0.5,
                response_time=0.08,
                energy_efficiency=0.75,
                fatigue_rate=0.08,
                recovery_rate=0.25,
                mass=0.004,
                biological_inspiration="cephalopod_muscle"
            )
        
        super().__init__(actuator_id, ActuatorType.EAP, properties)
        self.polymer_type = polymer_type
        
        # Additional EAP-specific properties
        self.voltage = 0.0
        self.charge = 0.0
    
    def activate(self, level: float, dt: float) -> Dict[str, float]:
        """
        Activate the EAP actuator with electrical dynamics.
        
        Args:
            level: Activation level (0-1)
            dt: Time step in seconds
            
        Returns:
            Current actuator state
        """
        # Simulate electrical dynamics
        self.voltage = level * 1000.0  # Scale to voltage (0-1000V)
        
        # Charge dynamics
        charge_time_constant = 0.01  # seconds
        charge_factor = min(dt / charge_time_constant, 1.0)
        target_charge = level
        self.charge += charge_factor * (target_charge - self.charge)
        
        # Use charge as the actual activation
        result = super().activate(self.charge, dt)
        
        # Add EAP-specific state variables
        result["voltage"] = self.voltage
        result["charge"] = self.charge
        
        return result


class BiomimeticActuatorController:
    """Controller for biomimetic actuators with neuromorphic integration."""
    
    def __init__(self, hardware_interface: Optional[NeuromorphicHardwareInterface] = None):
        """
        Initialize biomimetic actuator controller.
        
        Args:
            hardware_interface: Optional neuromorphic hardware interface
        """
        self.hardware_interface = hardware_interface
        self.actuators: Dict[str, BiomimeticActuator] = {}
        self.actuator_groups: Dict[str, List[str]] = {}
        self.neuron_mappings: Dict[int, str] = {}
        self.initialized = False
        
        logger.info("Initialized biomimetic actuator controller")
    
    def initialize(self) -> bool:
        """Initialize the controller and hardware interface."""
        try:
            if self.hardware_interface:
                # Initialize hardware
                config = {
                    "neuron_model": "LIF",
                    "simulation_mode": True
                }
                self.hardware_interface.initialize(config)
                
                # Allocate neurons for actuator control
                neuron_params = {
                    "threshold": 0.5,
                    "decay": 0.95,
                    "refractory_period": 0.001
                }
                neuron_count = len(self.actuators)
                if neuron_count > 0:
                    neuron_ids = self.hardware_interface.allocate_neurons(neuron_count, neuron_params)
                    
                    # Create neuron-to-actuator mappings
                    for i, actuator_id in enumerate(self.actuators.keys()):
                        if i < len(neuron_ids):
                            self.neuron_mappings[neuron_ids[i]] = actuator_id
            
            self.initialized = True
            logger.info("Biomimetic actuator controller initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize actuator controller: {e}")
            return False
    
    def add_actuator(self, actuator: BiomimeticActuator) -> None:
        """
        Add an actuator to the controller.
        
        Args:
            actuator: The actuator to add
        """
        self.actuators[actuator.actuator_id] = actuator
        logger.info(f"Added actuator: {actuator.actuator_id}")
    
    def create_actuator_group(self, group_name: str, actuator_ids: List[str]) -> None:
        """
        Create a group of actuators.
        
        Args:
            group_name: Name of the group
            actuator_ids: List of actuator IDs in the group
        """
        # Verify all actuators exist
        valid_ids = [aid for aid in actuator_ids if aid in self.actuators]
        if len(valid_ids) != len(actuator_ids):
            logger.warning(f"Some actuator IDs in group {group_name} are invalid")
        
        self.actuator_groups[group_name] = valid_ids
        logger.info(f"Created actuator group '{group_name}' with {len(valid_ids)} actuators")
    
    def activate_actuator(self, actuator_id: str, level: float, dt: float = 0.01) -> Dict[str, float]:
        """
        Activate a specific actuator.
        
        Args:
            actuator_id: ID of the actuator to activate
            level: Activation level (0-1)
            dt: Time step in seconds
            
        Returns:
            Actuator state
        """
        if actuator_id not in self.actuators:
            logger.error(f"Unknown actuator ID: {actuator_id}")
            return {}
        
        return self.actuators[actuator_id].activate(level, dt)
    
    def activate_group(self, group_name: str, level: float, dt: float = 0.01) -> Dict[str, Dict[str, float]]:
        """
        Activate a group of actuators.
        
        Args:
            group_name: Name of the actuator group
            level: Activation level (0-1)
            dt: Time step in seconds
            
        Returns:
            States of all actuators in the group
        """
        if group_name not in self.actuator_groups:
            logger.error(f"Unknown actuator group: {group_name}")
            return {}
        
        results = {}
        for actuator_id in self.actuator_groups[group_name]:
            results[actuator_id] = self.activate_actuator(actuator_id, level, dt)
        
        return results
    
    def process_neural_spikes(self, spike_data: Dict[int, List[float]], dt: float = 0.01) -> Dict[str, Dict[str, float]]:
        """
        Process neural spikes to activate actuators.
        
        Args:
            spike_data: Dictionary mapping neuron IDs to spike times
            dt: Time step in seconds
            
        Returns:
            Actuator states
        """
        results = {}
        
        for neuron_id, spike_times in spike_data.items():
            if neuron_id in self.neuron_mappings:
                actuator_id = self.neuron_mappings[neuron_id]
                
                # Calculate activation from spike frequency
                if spike_times:
                    # Simple frequency-based activation
                    num_spikes = len(spike_times)
                    activation = min(num_spikes * 0.2, 1.0)  # Scale: 5 spikes = full activation
                    
                    # Activate the actuator
                    results[actuator_id] = self.activate_actuator(actuator_id, activation, dt)
        
        return results
    
    def get_all_states(self) -> Dict[str, Dict[str, float]]:
        """Get states of all actuators."""
        return {aid: actuator.get_state() for aid, actuator in self.actuators.items()}
    
    def reset_all(self) -> None:
        """Reset all actuators."""
        for actuator in self.actuators.values():
            actuator.reset()
        logger.info("Reset all actuators")


def create_biomimetic_actuator_system(hardware_interface=None) -> BiomimeticActuatorController:
    """
    Create a complete biomimetic actuator system.
    
    Args:
        hardware_interface: Optional neuromorphic hardware interface
        
    Returns:
        Configured actuator controller
    """
    # Create controller
    controller = BiomimeticActuatorController(hardware_interface)
    
    # Create muscle actuators for wings
    # Left wing
    controller.add_actuator(MuscleFiberActuator("left_primary_elevator", "slow_twitch"))
    controller.add_actuator(MuscleFiberActuator("left_primary_depressor", "fast_twitch"))
    controller.add_actuator(MuscleFiberActuator("left_secondary_elevator", "slow_twitch"))
    controller.add_actuator(MuscleFiberActuator("left_secondary_depressor", "fast_twitch"))
    
    # Right wing
    controller.add_actuator(MuscleFiberActuator("right_primary_elevator", "slow_twitch"))
    controller.add_actuator(MuscleFiberActuator("right_primary_depressor", "fast_twitch"))
    controller.add_actuator(MuscleFiberActuator("right_secondary_elevator", "slow_twitch"))
    controller.add_actuator(MuscleFiberActuator("right_secondary_depressor", "fast_twitch"))
    
    # Wing control surfaces
    controller.add_actuator(ElectroactivePolymerActuator("left_wing_tip", "dielectric"))
    controller.add_actuator(ElectroactivePolymerActuator("right_wing_tip", "dielectric"))
    controller.add_actuator(ElectroactivePolymerActuator("left_trailing_edge", "ionic"))
    controller.add_actuator(ElectroactivePolymerActuator("right_trailing_edge", "ionic"))
    
    # Create actuator groups
    controller.create_actuator_group("left_wing_muscles", 
                                   ["left_primary_elevator", "left_primary_depressor", 
                                    "left_secondary_elevator", "left_secondary_depressor"])
    
    controller.create_actuator_group("right_wing_muscles", 
                                   ["right_primary_elevator", "right_primary_depressor", 
                                    "right_secondary_elevator", "right_secondary_depressor"])
    
    controller.create_actuator_group("left_wing_control", 
                                   ["left_wing_tip", "left_trailing_edge"])
    
    controller.create_actuator_group("right_wing_control", 
                                   ["right_wing_tip", "right_trailing_edge"])
    
    return controller
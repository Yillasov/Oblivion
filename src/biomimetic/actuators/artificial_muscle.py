#!/usr/bin/env python3
"""
Artificial muscle simulation and control for biomimetic systems.
Provides realistic muscle dynamics and control interfaces.
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
from src.landing_gear.hardware_interface import NeuromorphicHardwareInterface, SimulatedNeuromorphicHardware

logger = get_logger("artificial_muscle")


class MuscleType(Enum):
    """Types of artificial muscles."""
    SMA = "shape_memory_alloy"
    EAP = "electroactive_polymer"
    PNEUMATIC = "pneumatic"
    HYDRAULIC = "hydraulic"
    DIELECTRIC = "dielectric_elastomer"
    IONIC = "ionic_polymer_metal_composite"


@dataclass
class MuscleProperties:
    """Physical properties of an artificial muscle."""
    max_force: float  # Newtons
    max_contraction: float  # % of length
    response_time: float  # seconds
    power_density: float  # W/kg
    efficiency: float  # 0-1
    fatigue_rate: float  # 0-1 per hour
    recovery_rate: float  # 0-1 per hour
    mass_density: float  # kg/m³
    operating_voltage: float  # Volts (if applicable)
    operating_pressure: float  # kPa (if applicable)


class ArtificialMuscle:
    """Base class for artificial muscle simulation."""
    
    def __init__(self, muscle_id: str, muscle_type: MuscleType, properties: MuscleProperties,
                 length: float = 0.1, cross_section: float = 0.0001):
        """
        Initialize artificial muscle.
        
        Args:
            muscle_id: Unique identifier
            muscle_type: Type of artificial muscle
            properties: Muscle properties
            length: Resting length in meters
            cross_section: Cross-sectional area in m²
        """
        self.muscle_id = muscle_id
        self.muscle_type = muscle_type
        self.properties = properties
        self.length = length
        self.cross_section = cross_section
        
        # State variables
        self.activation = 0.0  # Current activation level (0-1)
        self.contraction = 0.0  # Current contraction ratio (0-1)
        self.force = 0.0  # Current force output (N)
        self.velocity = 0.0  # Contraction velocity (m/s)
        self.temperature = 20.0  # Current temperature (°C)
        self.fatigue = 0.0  # Current fatigue level (0-1)
        self.power_consumption = 0.0  # Current power consumption (W)
        
        # Simulation variables
        self.time = 0.0
        self.last_update_time = 0.0
        
        logger.info(f"Initialized {muscle_type.value} muscle: {muscle_id}")
    
    def activate(self, level: float, dt: float) -> Dict[str, float]:
        """
        Activate the muscle.
        
        Args:
            level: Activation level (0-1)
            dt: Time step in seconds
            
        Returns:
            Current muscle state
        """
        # Clamp activation level
        self.activation = max(0.0, min(1.0, level))
        
        # Update time
        self.time += dt
        
        # Calculate target contraction based on activation and fatigue
        effective_activation = self.activation * (1.0 - self.fatigue)
        target_contraction = effective_activation * self.properties.max_contraction
        
        # Apply response time dynamics (first-order system)
        response_factor = min(dt / self.properties.response_time, 1.0)
        delta_contraction = response_factor * (target_contraction - self.contraction)
        
        # Calculate contraction velocity
        self.velocity = delta_contraction / dt if dt > 0 else 0.0
        
        # Update contraction
        self.contraction += delta_contraction
        
        # Calculate force output (Hill's muscle model simplified)
        # F = F_max * (1 - v/v_max) * activation
        normalized_velocity = abs(self.velocity) / (self.properties.max_contraction / self.properties.response_time)
        velocity_factor = max(0.0, 1.0 - normalized_velocity)
        self.force = self.properties.max_force * effective_activation * velocity_factor
        
        # Update fatigue
        self.fatigue += self.activation * self.properties.fatigue_rate * dt
        self.fatigue = max(0.0, min(1.0, self.fatigue))
        
        # Apply recovery when not fully activated
        recovery = (1.0 - self.activation) * self.properties.recovery_rate * dt
        self.fatigue = max(0.0, self.fatigue - recovery)
        
        # Calculate power consumption
        self.power_consumption = self.force * abs(self.velocity) / self.properties.efficiency
        
        # Update temperature (simplified model)
        heat_generation = self.power_consumption * (1.0 - self.properties.efficiency) * dt
        temp_increase = heat_generation / (self.mass * 4.184)  # Specific heat capacity approximation
        self.temperature += temp_increase
        
        # Cooling proportional to temperature difference with environment
        ambient_temp = 20.0
        cooling_factor = 0.01  # W/(K·m²)
        cooling = cooling_factor * self.surface_area * (self.temperature - ambient_temp) * dt
        self.temperature -= cooling / (self.mass * 4.184)
        
        self.last_update_time = self.time
        
        return self.get_state()
    
    @property
    def mass(self) -> float:
        """Calculate muscle mass based on dimensions and density."""
        return self.length * self.cross_section * self.properties.mass_density
    
    @property
    def surface_area(self) -> float:
        """Approximate surface area for heat transfer."""
        radius = np.sqrt(self.cross_section / np.pi)
        return 2 * np.pi * radius * self.length
    
    @property
    def current_length(self) -> float:
        """Get current muscle length based on contraction."""
        return self.length * (1.0 - self.contraction)
    
    def get_state(self) -> Dict[str, float]:
        """Get current muscle state."""
        return {
            "activation": self.activation,
            "contraction": self.contraction,
            "force": self.force,
            "velocity": self.velocity,
            "temperature": self.temperature,
            "fatigue": self.fatigue,
            "power": self.power_consumption,
            "length": self.current_length
        }
    
    def reset(self) -> None:
        """Reset muscle to initial state."""
        self.activation = 0.0
        self.contraction = 0.0
        self.force = 0.0
        self.velocity = 0.0
        self.temperature = 20.0
        self.fatigue = 0.0
        self.power_consumption = 0.0
        logger.info(f"Reset muscle: {self.muscle_id}")


class ShapeMemoryAlloyMuscle(ArtificialMuscle):
    """Shape Memory Alloy (SMA) artificial muscle."""
    
    def __init__(self, muscle_id: str, length: float = 0.1, cross_section: float = 0.0001):
        """Initialize SMA muscle."""
        properties = MuscleProperties(
            max_force=200.0,  # N
            max_contraction=0.05,  # 5%
            response_time=1.0,  # seconds
            power_density=30.0,  # W/kg
            efficiency=0.02,  # 2%
            fatigue_rate=0.01,  # 1% per hour
            recovery_rate=0.005,  # 0.5% per hour
            mass_density=6500.0,  # kg/m³
            operating_voltage=5.0,  # V
            operating_pressure=0.0  # kPa
        )
        super().__init__(muscle_id, MuscleType.SMA, properties, length, cross_section)
        
        # SMA-specific properties
        self.phase_transition_temp = 70.0  # °C
        self.electrical_resistance = 100.0  # Ohms
        self.current = 0.0  # Amps
    
    def activate(self, level: float, dt: float) -> Dict[str, float]:
        """Activate SMA muscle with electrical current."""
        # Calculate current based on activation level
        self.current = level * (self.properties.operating_voltage / self.electrical_resistance)
        
        # Calculate heating from electrical current
        joule_heating = self.current**2 * self.electrical_resistance * dt
        temp_increase = joule_heating / (self.mass * 4.184)
        self.temperature += temp_increase
        
        # Determine activation based on temperature
        temp_activation = max(0.0, min(1.0, (self.temperature - 20.0) / (self.phase_transition_temp - 20.0)))
        
        # Use temperature-based activation instead of direct level
        result = super().activate(temp_activation, dt)
        
        # Add SMA-specific state variables
        result["current"] = self.current
        result["resistance"] = self.electrical_resistance
        
        return result


class ElectroactiveMuscle(ArtificialMuscle):
    """Electroactive Polymer (EAP) artificial muscle."""
    
    def __init__(self, muscle_id: str, length: float = 0.1, cross_section: float = 0.0001):
        """Initialize EAP muscle."""
        properties = MuscleProperties(
            max_force=10.0,  # N
            max_contraction=0.3,  # 30%
            response_time=0.05,  # seconds
            power_density=150.0,  # W/kg
            efficiency=0.3,  # 30%
            fatigue_rate=0.02,  # 2% per hour
            recovery_rate=0.01,  # 1% per hour
            mass_density=1200.0,  # kg/m³
            operating_voltage=3000.0,  # V
            operating_pressure=0.0  # kPa
        )
        super().__init__(muscle_id, MuscleType.EAP, properties, length, cross_section)
        
        # EAP-specific properties
        self.capacitance = 1e-9  # F
        self.voltage = 0.0  # V
        self.charge = 0.0  # C
    
    def activate(self, level: float, dt: float) -> Dict[str, float]:
        """Activate EAP muscle with high voltage."""
        # Calculate voltage based on activation level
        target_voltage = level * self.properties.operating_voltage
        
        # Voltage response dynamics
        voltage_response_time = 0.01  # seconds
        voltage_response_factor = min(dt / voltage_response_time, 1.0)
        self.voltage += voltage_response_factor * (target_voltage - self.voltage)
        
        # Calculate charge
        self.charge = self.capacitance * self.voltage
        
        # Calculate electrostatic pressure (simplified)
        # P = ε₀εᵣE² = ε₀εᵣ(V/d)²
        permittivity = 8.85e-12  # F/m
        rel_permittivity = 3.0
        thickness = 0.0001  # m
        electrostatic_pressure = permittivity * rel_permittivity * (self.voltage / thickness)**2
        
        # Calculate effective activation from electrostatic pressure
        pressure_activation = electrostatic_pressure / 1e6  # Normalize to 0-1 range
        effective_activation = min(1.0, pressure_activation)
        
        # Use pressure-based activation
        result = super().activate(effective_activation, dt)
        
        # Add EAP-specific state variables
        result["voltage"] = self.voltage
        result["charge"] = self.charge
        
        return result


class PneumaticMuscle(ArtificialMuscle):
    """Pneumatic artificial muscle (PAM)."""
    
    def __init__(self, muscle_id: str, length: float = 0.1, cross_section: float = 0.0001):
        """Initialize pneumatic muscle."""
        properties = MuscleProperties(
            max_force=500.0,  # N
            max_contraction=0.25,  # 25%
            response_time=0.2,  # seconds
            power_density=50.0,  # W/kg
            efficiency=0.4,  # 40%
            fatigue_rate=0.005,  # 0.5% per hour
            recovery_rate=0.02,  # 2% per hour
            mass_density=1000.0,  # kg/m³
            operating_voltage=0.0,  # V
            operating_pressure=500.0  # kPa
        )
        super().__init__(muscle_id, MuscleType.PNEUMATIC, properties, length, cross_section)
        
        # PAM-specific properties
        self.pressure = 0.0  # kPa
        self.volume = np.pi * (cross_section / np.pi)**2 * length  # m³
        self.braided_angle = 20.0  # degrees
    
    def activate(self, level: float, dt: float) -> Dict[str, float]:
        """Activate pneumatic muscle with pressure."""
        # Calculate pressure based on activation level
        target_pressure = level * self.properties.operating_pressure
        
        # Pressure response dynamics
        pressure_response_time = 0.1  # seconds
        pressure_response_factor = min(dt / pressure_response_time, 1.0)
        self.pressure += pressure_response_factor * (target_pressure - self.pressure)
        
        # Calculate force based on pressure and geometry (McKibben muscle model)
        # F = P·πr²·(3cos²θ - 1)
        radius = np.sqrt(self.cross_section / np.pi)
        theta_rad = np.radians(self.braided_angle)
        force_factor = 3 * np.cos(theta_rad)**2 - 1
        pressure_force = self.pressure * 1000 * np.pi * radius**2 * force_factor
        
        # Calculate effective activation from pressure
        pressure_activation = self.pressure / self.properties.operating_pressure
        
        # Use pressure-based activation but override force calculation
        result = super().activate(pressure_activation, dt)
        self.force = pressure_force
        result["force"] = self.force
        
        # Update volume based on contraction
        contraction_factor = 1.0 - self.contraction
        self.volume = np.pi * (radius * contraction_factor)**2 * self.length / contraction_factor
        
        # Add PAM-specific state variables
        result["pressure"] = self.pressure
        result["volume"] = self.volume
        
        return result


class MuscleController:
    """Controller for artificial muscles with neuromorphic integration."""
    
    def __init__(self, hardware_interface: Optional[NeuromorphicHardwareInterface] = None,
                 time_step: float = 0.01):
        """
        Initialize muscle controller.
        
        Args:
            hardware_interface: Optional neuromorphic hardware interface
            time_step: Simulation time step in seconds
        """
        self.hardware_interface = hardware_interface or SimulatedNeuromorphicHardware()
        self.time_step = time_step
        self.muscles: Dict[str, ArtificialMuscle] = {}
        self.muscle_groups: Dict[str, List[str]] = {}
        self.neuron_mappings: Dict[int, str] = {}
        self.simulation_time = 0.0
        self.initialized = False
        
        logger.info("Initialized muscle controller")
    
    def initialize(self) -> bool:
        """Initialize the controller and hardware interface."""
        try:
            # Initialize hardware
            config = {
                "neuron_model": "LIF",
                "simulation_mode": True
            }
            self.hardware_interface.initialize(config)
            
            # Allocate neurons for muscle control
            neuron_params = {
                "threshold": 0.5,
                "decay": 0.95,
                "refractory_period": 0.001
            }
            neuron_count = len(self.muscles)
            if neuron_count > 0:
                neuron_ids = self.hardware_interface.allocate_neurons(neuron_count, neuron_params)
                
                # Create neuron-to-muscle mappings
                for i, muscle_id in enumerate(self.muscles.keys()):
                    if i < len(neuron_ids):
                        self.neuron_mappings[neuron_ids[i]] = muscle_id
            
            self.initialized = True
            logger.info("Muscle controller initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize muscle controller: {e}")
            return False
    
    def add_muscle(self, muscle: ArtificialMuscle) -> None:
        """
        Add a muscle to the controller.
        
        Args:
            muscle: The muscle to add
        """
        self.muscles[muscle.muscle_id] = muscle
        logger.info(f"Added muscle: {muscle.muscle_id}")
    
    def create_muscle_group(self, group_name: str, muscle_ids: List[str]) -> None:
        """
        Create a group of muscles.
        
        Args:
            group_name: Name of the group
            muscle_ids: List of muscle IDs in the group
        """
        # Verify all muscles exist
        valid_ids = [mid for mid in muscle_ids if mid in self.muscles]
        if len(valid_ids) != len(muscle_ids):
            logger.warning(f"Some muscle IDs in group {group_name} are invalid")
        
        self.muscle_groups[group_name] = valid_ids
        logger.info(f"Created muscle group '{group_name}' with {len(valid_ids)} muscles")
    
    def activate_muscle(self, muscle_id: str, level: float) -> Dict[str, float]:
        """
        Activate a specific muscle.
        
        Args:
            muscle_id: ID of the muscle to activate
            level: Activation level (0-1)
            
        Returns:
            Muscle state
        """
        if muscle_id not in self.muscles:
            logger.error(f"Unknown muscle ID: {muscle_id}")
            return {}
        
        return self.muscles[muscle_id].activate(level, self.time_step)
    
    def activate_group(self, group_name: str, level: float) -> Dict[str, Dict[str, float]]:
        """
        Activate a group of muscles.
        
        Args:
            group_name: Name of the muscle group
            level: Activation level (0-1)
            
        Returns:
            States of all muscles in the group
        """
        if group_name not in self.muscle_groups:
            logger.error(f"Unknown muscle group: {group_name}")
            return {}
        
        results = {}
        for muscle_id in self.muscle_groups[group_name]:
            results[muscle_id] = self.activate_muscle(muscle_id, level)
        
        return results
    
    def process_neural_spikes(self, duration: float) -> Dict[str, Dict[str, float]]:
        """
        Process neural spikes to activate muscles.
        
        Args:
            duration: Simulation duration in seconds
            
        Returns:
            Muscle states
        """
        if not self.initialized:
            logger.error("Controller not initialized")
            return {}
        
        # Run neuromorphic simulation
        spike_data = self.hardware_interface.run_simulation(duration)
        
        results = {}
        for neuron_id, spike_times in spike_data.items():
            if neuron_id in self.neuron_mappings:
                muscle_id = self.neuron_mappings[neuron_id]
                
                # Calculate activation from spike frequency
                if spike_times:
                    # Simple frequency-based activation
                    num_spikes = len(spike_times)
                    activation = min(num_spikes * 0.2, 1.0)  # Scale: 5 spikes = full activation
                    
                    # Activate the muscle
                    results[muscle_id] = self.activate_muscle(muscle_id, activation)
        
        # Update simulation time
        self.simulation_time += duration
        
        return results
    
    def step_simulation(self) -> Dict[str, Dict[str, float]]:
        """
        Step the simulation forward by one time step.
        
        Returns:
            States of all muscles
        """
        results = {}
        
        # Update all muscles with current activation
        for muscle_id, muscle in self.muscles.items():
            results[muscle_id] = muscle.activate(muscle.activation, self.time_step)
        
        # Update simulation time
        self.simulation_time += self.time_step
        
        return results
    
    def run_simulation(self, duration: float, control_callback=None) -> Dict[str, List[Dict[str, float]]]:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration in seconds
            control_callback: Optional callback function for control inputs
            
        Returns:
            Time series of muscle states
        """
        steps = int(duration / self.time_step)
        history = {muscle_id: [] for muscle_id in self.muscles}
        
        for step in range(steps):
            # Get control inputs from callback if provided
            if control_callback:
                control_inputs = control_callback(self.get_all_states(), self.simulation_time)
                
                # Apply control inputs
                for muscle_id, level in control_inputs.items():
                    if muscle_id in self.muscles:
                        self.activate_muscle(muscle_id, level)
            
            # Step simulation
            states = self.step_simulation()
            
            # Record history
            for muscle_id, state in states.items():
                history[muscle_id].append(state)
        
        return history
    
    def get_all_states(self) -> Dict[str, Dict[str, float]]:
        """Get states of all muscles."""
        return {mid: muscle.get_state() for mid, muscle in self.muscles.items()}
    
    def reset_all(self) -> None:
        """Reset all muscles."""
        for muscle in self.muscles.values():
            muscle.reset()
        logger.info("Reset all muscles")


def create_wing_muscle_system() -> MuscleController:
    """
    Create a complete wing muscle system.
    
    Returns:
        Configured muscle controller
    """
    # Create controller with simulated hardware
    hardware = SimulatedNeuromorphicHardware()
    controller = MuscleController(hardware)
    
    # Create muscles for left wing
    controller.add_muscle(PneumaticMuscle("left_primary_elevator", 0.05, 0.0002))
    controller.add_muscle(PneumaticMuscle("left_primary_depressor", 0.05, 0.0002))
    controller.add_muscle(ElectroactiveMuscle("left_wing_twist", 0.03, 0.0001))
    
    # Create muscles for right wing
    controller.add_muscle(PneumaticMuscle("right_primary_elevator", 0.05, 0.0002))
    controller.add_muscle(PneumaticMuscle("right_primary_depressor", 0.05, 0.0002))
    controller.add_muscle(ElectroactiveMuscle("right_wing_twist", 0.03, 0.0001))
    
    # Create muscles for control surfaces
    controller.add_muscle(ShapeMemoryAlloyMuscle("left_aileron", 0.02, 0.0001))
    controller.add_muscle(ShapeMemoryAlloyMuscle("right_aileron", 0.02, 0.0001))
    controller.add_muscle(ElectroactiveMuscle("elevator", 0.04, 0.00015))
    controller.add_muscle(ElectroactiveMuscle("rudder", 0.04, 0.00015))
    
    # Create muscle groups
    controller.create_muscle_group("left_wing", 
                                ["left_primary_elevator", "left_primary_depressor", "left_wing_twist"])
    
    controller.create_muscle_group("right_wing", 
                                ["right_primary_elevator", "right_primary_depressor", "right_wing_twist"])
    
    controller.create_muscle_group("control_surfaces", 
                                ["left_aileron", "right_aileron", "elevator", "rudder"])
    
    # Initialize controller
    controller.initialize()
    
    return controller


def run_wing_flapping_demo(duration: float = 5.0) -> Dict[str, Any]:
    """
    Run a demonstration of wing flapping with artificial muscles.
    
    Args:
        duration: Simulation duration in seconds
        
    Returns:
        Simulation results
    """
    # Create wing muscle system
    controller = create_wing_muscle_system()
    
    # Define control function for wing flapping
    def flapping_control(states, time):
        # Sinusoidal flapping pattern
        freq = 5.0  # Hz
        phase = time * freq * 2 * np.pi
        
        # Alternating activation of elevator and depressor muscles
        elevator_activation = max(0.0, np.sin(phase))
        depressor_activation = max(0.0, -np.sin(phase))
        
        # Wing twist based on flapping phase
        twist_activation = 0.5 + 0.5 * np.sin(phase + np.pi/4)
        
        return {
            "left_primary_elevator": elevator_activation,
            "left_primary_depressor": depressor_activation,
            "left_wing_twist": twist_activation,
            "right_primary_elevator": elevator_activation,
            "right_primary_depressor": depressor_activation,
            "right_wing_twist": twist_activation,
            "left_aileron": 0.3,
            "right_aileron": 0.3,
            "elevator": 0.5,
            "rudder": 0.2
        }
    
    # Run simulation
    history = controller.run_simulation(duration, flapping_control)
    
    # Calculate performance metrics
    total_power = 0.0
    max_force = 0.0
    flapping_cycles = 0
    
    for muscle_id, states in history.items():
        if "primary" in muscle_id:
            # Count flapping cycles from primary elevator
            activations = [state["activation"] for state in states]
            # Count zero crossings
            crossings = sum(1 for i in range(1, len(activations)) 
                          if activations[i-1] < 0.5 and activations[i] >= 0.5)
            flapping_cycles = max(flapping_cycles, crossings)
        
        # Calculate average power consumption
        avg_power = sum(state["power"] for state in states) / len(states)
        total_power += avg_power
        
        # Find maximum force
        muscle_max_force = max(state["force"] for state in states)
        max_force = max(max_force, muscle_max_force)
    
    return {
        "duration": duration,
        "flapping_frequency": flapping_cycles / duration,
        "total_power_consumption": total_power,
        "max_force": max_force,
        "muscle_history": history
    }


if __name__ == "__main__":
    # Run demo
    results = run_wing_flapping_demo(2.0)
    
    print(f"Simulation completed:")
    print(f"  Duration: {results['duration']} seconds")
    print(f"  Flapping frequency: {results['flapping_frequency']:.2f} Hz")
    print(f"  Total power consumption: {results['total_power_consumption']:.2f} W")
    print(f"  Maximum force: {results['max_force']:.2f} N")
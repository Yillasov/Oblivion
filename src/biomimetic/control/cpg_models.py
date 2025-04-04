#!/usr/bin/env python3
"""
Biomimetic Control Systems based on Central Pattern Generators (CPGs)
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.biomimetic.design.principles import BiomimeticPrinciple

logger = get_logger("cpg_models")


class CPGType(Enum):
    """Types of biological CPG models."""
    MATSUOKA_OSCILLATOR = "matsuoka"  # Mutual inhibition oscillator
    HOPF_OSCILLATOR = "hopf"          # Limit cycle oscillator
    KURAMOTO_OSCILLATOR = "kuramoto"  # Phase-coupled oscillator
    SALAMANDER_CPG = "salamander"     # Salamander locomotion CPG
    INSECT_HEXAPOD = "hexapod"        # Insect hexapod locomotion
    BIRD_WING = "bird_wing"           # Bird wing flapping pattern
    BAT_WING = "bat_wing"             # Bat wing articulation


@dataclass
class CPGParameters:
    """Parameters for CPG models."""
    frequency: float = 1.0            # Base oscillation frequency (Hz)
    amplitude: float = 1.0            # Oscillation amplitude
    phase_offset: float = 0.0         # Phase offset (radians)
    coupling_strength: float = 0.2    # Coupling strength between oscillators
    adaptation_rate: float = 0.05     # Rate of adaptation to external inputs
    time_scale: float = 0.1           # Time scale of the oscillator
    bias: float = 0.0                 # Bias term
    
    # Matsuoka oscillator specific parameters
    tau_rise: float = 0.1             # Time constant for neuron activation
    tau_adaptation: float = 0.5       # Time constant for adaptation
    
    # Hopf oscillator specific parameters
    convergence_rate: float = 1.0     # Rate of convergence to limit cycle
    
    # Kuramoto oscillator specific parameters
    natural_frequencies: List[float] = None  # Natural frequencies of oscillators


class CPGOscillator:
    """Base class for CPG oscillators."""
    
    def __init__(self, params: CPGParameters):
        """Initialize the oscillator with parameters."""
        self.params = params
        self.state = np.zeros(2)  # Basic state: [x, y]
        self.output = 0.0
        self.time = 0.0
        logger.info(f"Initialized CPG oscillator with frequency {params.frequency} Hz")
    
    def step(self, dt: float, inputs: Optional[np.ndarray] = None) -> float:
        """Advance the oscillator by one time step."""
        self.time += dt
        # Override in subclasses
        return self.output
    
    def reset(self) -> None:
        """Reset the oscillator state."""
        self.state = np.zeros_like(self.state)
        self.output = 0.0
        self.time = 0.0
    
    def get_state(self) -> np.ndarray:
        """Get the current state of the oscillator."""
        return self.state.copy()


class MatsuokaOscillator(CPGOscillator):
    """Matsuoka oscillator based on mutual inhibition."""
    
    def __init__(self, params: CPGParameters):
        super().__init__(params)
        # State: [x1, x2, v1, v2] where x are neuron activations and v are adaptation variables
        self.state = np.zeros(4)
        logger.info("Initialized Matsuoka oscillator")
    
    def step(self, dt: float, inputs: Optional[np.ndarray] = None) -> float:
        """Advance the Matsuoka oscillator by one time step."""
        super().step(dt, inputs)
        
        # Unpack state
        x1, x2, v1, v2 = self.state
        
        # External inputs
        u1 = self.params.bias
        u2 = self.params.bias
        if inputs is not None and len(inputs) >= 2:
            u1 += inputs[0]
            u2 += inputs[1]
        
        # Compute derivatives
        dx1 = (-x1 - self.params.coupling_strength * max(0, x2) - v1 + u1) / self.params.tau_rise
        dx2 = (-x2 - self.params.coupling_strength * max(0, x1) - v2 + u2) / self.params.tau_rise
        dv1 = (-v1 + max(0, x1)) / self.params.tau_adaptation
        dv2 = (-v2 + max(0, x2)) / self.params.tau_adaptation
        
        # Update state using Euler integration
        self.state[0] += dx1 * dt
        self.state[1] += dx2 * dt
        self.state[2] += dv1 * dt
        self.state[3] += dv2 * dt
        
        # Compute output
        self.output = self.params.amplitude * (max(0, x1) - max(0, x2))
        
        return self.output


class HopfOscillator(CPGOscillator):
    """Hopf oscillator with stable limit cycle."""
    
    def __init__(self, params: CPGParameters):
        super().__init__(params)
        # State: [x, y]
        self.state = np.array([0.1, 0.0])  # Small initial perturbation
        logger.info("Initialized Hopf oscillator")
    
    def step(self, dt: float, inputs: Optional[np.ndarray] = None) -> float:
        """Advance the Hopf oscillator by one time step."""
        super().step(dt, inputs)
        
        # Unpack state
        x, y = self.state
        
        # Calculate radius
        r = np.sqrt(x**2 + y**2)
        
        # External inputs
        input_x = 0.0
        input_y = 0.0
        if inputs is not None and len(inputs) >= 2:
            input_x = inputs[0]
            input_y = inputs[1]
        
        # Compute derivatives (Hopf normal form)
        omega = 2 * np.pi * self.params.frequency
        dx = (self.params.convergence_rate * (1 - r**2) * x - omega * y) / self.params.time_scale + input_x
        dy = (self.params.convergence_rate * (1 - r**2) * y + omega * x) / self.params.time_scale + input_y
        
        # Update state using Euler integration
        self.state[0] += dx * dt
        self.state[1] += dy * dt
        
        # Compute output
        self.output = self.params.amplitude * x
        
        return self.output


class KuramotoOscillator(CPGOscillator):
    """Kuramoto oscillator based on phase coupling."""
    
    def __init__(self, params: CPGParameters, num_oscillators: int = 4):
        super().__init__(params)
        # State: [theta_1, theta_2, ..., theta_n]
        self.num_oscillators = num_oscillators
        self.state = np.random.uniform(0, 2*np.pi, num_oscillators)
        
        # Set natural frequencies if not provided
        if params.natural_frequencies is None or len(params.natural_frequencies) != num_oscillators:
            self.natural_frequencies = np.ones(num_oscillators) * self.params.frequency
            # Add slight variations
            self.natural_frequencies += np.random.normal(0, 0.1, num_oscillators)
        else:
            self.natural_frequencies = np.array(params.natural_frequencies)
        
        logger.info(f"Initialized Kuramoto oscillator with {num_oscillators} oscillators")
    
    def step(self, dt: float, inputs: Optional[np.ndarray] = None) -> float:
        """Advance the Kuramoto oscillator by one time step."""
        super().step(dt, inputs)
        
        # External inputs
        if inputs is not None and len(inputs) == self.num_oscillators:
            input_freqs = inputs
        else:
            input_freqs = np.zeros(self.num_oscillators)
        
        # Compute phase derivatives
        dtheta = np.zeros(self.num_oscillators)
        
        # Natural frequencies + inputs
        dtheta += 2 * np.pi * (self.natural_frequencies + input_freqs)
        
        # Coupling terms
        for i in range(self.num_oscillators):
            for j in range(self.num_oscillators):
                if i != j:
                    dtheta[i] += self.params.coupling_strength * np.sin(self.state[j] - self.state[i])
        
        # Update state using Euler integration
        self.state += dtheta * dt
        
        # Wrap phases to [0, 2π]
        self.state = np.mod(self.state, 2*np.pi)
        
        # Compute output (mean field)
        x = np.mean(np.cos(self.state))
        y = np.mean(np.sin(self.state))
        self.output = self.params.amplitude * x
        
        return self.output


class CPGNetwork:
    """Network of coupled CPG oscillators."""
    
    def __init__(self, cpg_type: CPGType, num_oscillators: int = 4, coupling_matrix: Optional[np.ndarray] = None):
        """
        Initialize a CPG network.
        
        Args:
            cpg_type: Type of CPG oscillator
            num_oscillators: Number of oscillators in the network
            coupling_matrix: Matrix defining coupling between oscillators
        """
        self.cpg_type = cpg_type
        self.num_oscillators = num_oscillators
        self.oscillators = []
        
        # Initialize coupling matrix if not provided
        if coupling_matrix is None:
            # Default: nearest-neighbor coupling
            self.coupling_matrix = np.zeros((num_oscillators, num_oscillators))
            for i in range(num_oscillators):
                if i > 0:
                    self.coupling_matrix[i, i-1] = 1.0
                if i < num_oscillators - 1:
                    self.coupling_matrix[i, i+1] = 1.0
        else:
            self.coupling_matrix = coupling_matrix
        
        # Create oscillators
        for i in range(num_oscillators):
            # Create parameters with phase offsets
            params = CPGParameters(
                frequency=1.0,
                amplitude=1.0,
                phase_offset=i * 2 * np.pi / num_oscillators,
                coupling_strength=0.2
            )
            
            # Create oscillator based on type
            if cpg_type == CPGType.MATSUOKA_OSCILLATOR:
                self.oscillators.append(MatsuokaOscillator(params))
            elif cpg_type == CPGType.HOPF_OSCILLATOR:
                self.oscillators.append(HopfOscillator(params))
            elif cpg_type == CPGType.KURAMOTO_OSCILLATOR:
                # For Kuramoto, we use a single oscillator with multiple phases
                if i == 0:
                    self.oscillators.append(KuramotoOscillator(params, num_oscillators))
                    break
            else:
                # Default to Hopf oscillator
                self.oscillators.append(HopfOscillator(params))
        
        logger.info(f"Initialized CPG network with {len(self.oscillators)} oscillators of type {cpg_type.value}")
    
    def step(self, dt: float, inputs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance the CPG network by one time step.
        
        Args:
            dt: Time step
            inputs: External inputs to oscillators
            
        Returns:
            Array of oscillator outputs
        """
        # Special case for Kuramoto (single oscillator with multiple phases)
        if self.cpg_type == CPGType.KURAMOTO_OSCILLATOR:
            self.oscillators[0].step(dt, inputs)
            # Return outputs based on phases
            phases = self.oscillators[0].state
            return self.oscillators[0].params.amplitude * np.cos(phases)
        
        # For other oscillator types
        outputs = np.zeros(len(self.oscillators))
        
        # First pass: collect current outputs
        for i, osc in enumerate(self.oscillators):
            outputs[i] = osc.output
        
        # Second pass: step oscillators with coupling
        for i, osc in enumerate(self.oscillators):
            # Compute coupling inputs
            coupling_input = np.sum(self.coupling_matrix[i] * outputs)
            
            # Get external input if available
            external_input = None
            if inputs is not None and i < len(inputs):
                external_input = np.array([inputs[i], coupling_input])
            else:
                external_input = np.array([0.0, coupling_input])
            
            # Step the oscillator
            outputs[i] = osc.step(dt, external_input)
        
        return outputs
    
    def reset(self) -> None:
        """Reset all oscillators in the network."""
        for osc in self.oscillators:
            osc.reset()
    
    def set_frequency(self, frequency: float) -> None:
        """Set the frequency of all oscillators."""
        for osc in self.oscillators:
            osc.params.frequency = frequency
    
    def set_amplitude(self, amplitude: float) -> None:
        """Set the amplitude of all oscillators."""
        for osc in self.oscillators:
            osc.params.amplitude = amplitude


class BiomimeticCPGController:
    """Biomimetic controller based on CPG networks."""
    
    def __init__(self, neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """
        Initialize the biomimetic CPG controller.
        
        Args:
            neuromorphic_system: Optional neuromorphic system for integration
        """
        self.neuromorphic_system = neuromorphic_system
        self.cpg_networks: Dict[str, CPGNetwork] = {}
        self.output_mappings: Dict[str, Dict[str, Any]] = {}
        self.input_mappings: Dict[str, Dict[str, Any]] = {}
        self.time = 0.0
        self.dt = 0.01  # Default time step (10ms)
        self.initialized = False
        
        logger.info("Initialized biomimetic CPG controller")
    
    def add_cpg_network(self, 
                       name: str, 
                       cpg_type: CPGType, 
                       num_oscillators: int = 4,
                       coupling_matrix: Optional[np.ndarray] = None,
                       output_mapping: Optional[Dict[str, Any]] = None,
                       input_mapping: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a CPG network to the controller.
        
        Args:
            name: Name of the CPG network
            cpg_type: Type of CPG oscillator
            num_oscillators: Number of oscillators in the network
            coupling_matrix: Matrix defining coupling between oscillators
            output_mapping: Mapping from oscillator outputs to control signals
            input_mapping: Mapping from sensor inputs to oscillator inputs
            
        Returns:
            Success flag
        """
        if name in self.cpg_networks:
            logger.warning(f"CPG network {name} already exists")
            return False
        
        # Create CPG network
        self.cpg_networks[name] = CPGNetwork(cpg_type, num_oscillators, coupling_matrix)
        
        # Store mappings
        self.output_mappings[name] = output_mapping or {}
        self.input_mappings[name] = input_mapping or {}
        
        logger.info(f"Added CPG network {name} with {num_oscillators} oscillators")
        return True
    
    def initialize(self) -> bool:
        """Initialize the controller and its components."""
        if self.initialized:
            return True
        
        try:
            # Initialize neuromorphic system if available
            if self.neuromorphic_system:
                self.neuromorphic_system.initialize()
            
            # Reset all CPG networks
            for network in self.cpg_networks.values():
                network.reset()
            
            self.initialized = True
            logger.info("Biomimetic CPG controller initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize biomimetic CPG controller: {e}")
            return False
    
    def integrate_sensory_feedback(self, sensor_interface) -> bool:
        """
        Integrate sensory feedback with the CPG controller.
        
        Args:
            sensor_interface: Interface to sensor system
            
        Returns:
            Success flag
        """
        try:
            from src.biomimetic.control.sensory_feedback import SensoryFeedbackIntegration
            
            # Create sensory feedback integration
            self.sensory_feedback = SensoryFeedbackIntegration(self)
            
            # Set up default feedback mappings based on available networks
            for name, network in self.cpg_networks.items():
                if "walking" in name or "gait" in name:
                    # Add obstacle avoidance feedback
                    self.sensory_feedback.add_feedback_mapping(
                        self.sensory_feedback.create_obstacle_avoidance_feedback(
                            target_network=name
                        )
                    )
                    
                    # Add terrain adaptation feedback
                    self.sensory_feedback.add_feedback_mapping(
                        self.sensory_feedback.create_terrain_adaptation_feedback(
                            target_network=name
                        )
                    )
                
                elif "flying" in name or "wing" in name:
                    # Add air current feedback for flying
                    from src.simulation.sensors.sensor_framework import SensorType
                    from src.biomimetic.control.sensory_feedback import FeedbackMapping, FeedbackType
                    
                    self.sensory_feedback.add_feedback_mapping(
                        FeedbackMapping(
                            sensor_type=SensorType.AIR_FLOW,
                            sensor_key="air_current",
                            feedback_type=FeedbackType.PHASE_MODULATION,
                            target_network=name,
                            target_oscillators=[],  # All oscillators
                            scaling_factor=0.2,
                            threshold=0.1,
                            adaptation_rate=0.3
                        )
                    )
            
            logger.info("Integrated sensory feedback with CPG controller")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate sensory feedback: {e}")
            return False
    
    # Override step method to include sensory feedback processing
    def step(self, sensor_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Advance the controller by one time step.
        
        Args:
            sensor_inputs: Dictionary of sensor inputs
            
        Returns:
            Dictionary of control outputs
        """
        if not self.initialized:
            self.initialize()
        
        self.time += self.dt
        
        # Process sensory feedback if available
        if hasattr(self, 'sensory_feedback') and sensor_inputs:
            self.sensory_feedback.process_sensor_data(sensor_inputs)
        
        # Continue with original step implementation
        control_outputs = {}
        
        # Process each CPG network
        for name, network in self.cpg_networks.items():
            # Map sensor inputs to oscillator inputs
            oscillator_inputs = None
            if sensor_inputs and name in self.input_mappings:
                mapping = self.input_mappings[name]
                if callable(mapping.get("function")):
                    oscillator_inputs = mapping["function"](sensor_inputs)
                elif "mapping" in mapping:
                    input_map = mapping["mapping"]
                    oscillator_inputs = np.zeros(network.num_oscillators)
                    for i, sensor_key in enumerate(input_map):
                        if sensor_key in sensor_inputs and i < network.num_oscillators:
                            oscillator_inputs[i] = sensor_inputs[sensor_key]
            
            # Step the CPG network
            outputs = network.step(self.dt, oscillator_inputs)
            
            # Map oscillator outputs to control outputs
            if name in self.output_mappings:
                mapping = self.output_mappings[name]
                if callable(mapping.get("function")):
                    control_outputs[name] = mapping["function"](outputs)
                elif "mapping" in mapping:
                    output_map = mapping["mapping"]
                    network_outputs = {}
                    for i, control_key in enumerate(output_map):
                        if i < len(outputs):
                            network_outputs[control_key] = outputs[i]
                    control_outputs[name] = network_outputs
            else:
                # Default: just pass through the raw outputs
                control_outputs[name] = outputs
        
        return control_outputs
    
    def configure_for_locomotion(self, locomotion_type: str) -> bool:
        """
        Configure the controller for a specific type of locomotion.
        
        Args:
            locomotion_type: Type of locomotion (e.g., "walking", "flying", "swimming")
            
        Returns:
            Success flag
        """
        if locomotion_type == "walking":
            # Configure for quadruped walking gait
            coupling = np.array([
                [0, 1, 0, 1],  # Front left
                [1, 0, 1, 0],  # Front right
                [0, 1, 0, 1],  # Rear left
                [1, 0, 1, 0]   # Rear right
            ])
            
            return self.add_cpg_network(
                "walking_gait",
                CPGType.HOPF_OSCILLATOR,
                num_oscillators=4,
                coupling_matrix=coupling,
                output_mapping={"mapping": ["fl_leg", "fr_leg", "rl_leg", "rr_leg"]}
            )
            
        elif locomotion_type == "flying":
            # Configure for bird-like wing flapping
            return self.add_cpg_network(
                "wing_flapping",
                CPGType.MATSUOKA_OSCILLATOR,
                num_oscillators=2,
                output_mapping={"mapping": ["left_wing", "right_wing"]}
            )
            
        elif locomotion_type == "swimming":
            # Configure for fish-like swimming
            num_segments = 8
            coupling = np.zeros((num_segments, num_segments))
            for i in range(num_segments-1):
                coupling[i, i+1] = 0.5
                coupling[i+1, i] = 0.5
            
            return self.add_cpg_network(
                "swimming",
                CPGType.KURAMOTO_OSCILLATOR,
                num_oscillators=num_segments,
                coupling_matrix=coupling,
                output_mapping={"mapping": [f"segment_{i}" for i in range(num_segments)]}
            )
            
        else:
            logger.warning(f"Unknown locomotion type: {locomotion_type}")
            return False
    
    def configure_for_wing_morphing(self, wing_type: str) -> bool:
        """
        Configure the controller for wing morphing.
        
        Args:
            wing_type: Type of wing ("bird", "bat", "insect")
            
        Returns:
            Success flag
        """
        try:
            from src.biomimetic.control.wing_morphing import create_wing_morphing_controller
            
            # Create wing morphing controller
            self.wing_morphing = create_wing_morphing_controller(wing_type, self)
            
            # Add CPG network for morphing if not already present
            if "wing_morphing" not in self.cpg_networks:
                # Create appropriate coupling matrix based on wing type
                if wing_type.lower() == "bird":
                    num_modes = 4  # camber, twist, sweep, span
                    coupling = np.array([
                        [0, 0.3, 0.2, 0.1],  # camber
                        [0.3, 0, 0.1, 0.1],  # twist
                        [0.2, 0.1, 0, 0.4],  # sweep
                        [0.1, 0.1, 0.4, 0]   # span
                    ])
                elif wing_type.lower() == "bat":
                    num_modes = 3  # camber, membrane, articulated
                    coupling = np.array([
                        [0, 0.5, 0.3],      # camber
                        [0.5, 0, 0.6],      # membrane
                        [0.3, 0.6, 0]       # articulated
                    ])
                else:
                    num_modes = 2  # camber, twist
                    coupling = np.array([
                        [0, 0.3],
                        [0.3, 0]
                    ])
                
                # Add the network
                self.add_cpg_network(
                    "wing_morphing",
                    CPGType.HOPF_OSCILLATOR,
                    num_oscillators=num_modes,
                    coupling_matrix=coupling
                )
                
                # Set low frequency for smooth morphing
                self.cpg_networks["wing_morphing"].set_frequency(0.2)
                self.cpg_networks["wing_morphing"].set_amplitude(0.5)
            
            logger.info(f"Configured for {wing_type} wing morphing")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure for wing morphing: {e}")
            return False
    
    def visualize_outputs(self, duration: float = 5.0, dt: float = 0.01) -> plt.Figure:
        """
        Visualize CPG outputs over time.
        
        Args:
            duration: Duration of simulation (seconds)
            dt: Time step (seconds)
            
        Returns:
            Matplotlib figure
        """
        # Reset networks
        for network in self.cpg_networks.values():
            network.reset()
        
        # Prepare data structures
        time_points = np.arange(0, duration, dt)
        outputs = {name: np.zeros((len(time_points), network.num_oscillators)) 
                  for name, network in self.cpg_networks.items()}
        
        # Run simulation
        for i, t in enumerate(time_points):
            for name, network in self.cpg_networks.items():
                outputs[name][i] = network.step(dt)
        
        # Create figure
        fig = plt.figure(figsize=(12, 4 * len(self.cpg_networks)))
        
        # Plot each network
        for i, (name, data) in enumerate(outputs.items()):
            ax = fig.add_subplot(len(self.cpg_networks), 1, i+1)
            for j in range(data.shape[1]):
                ax.plot(time_points, data[:, j], label=f"Oscillator {j+1}")
            ax.set_title(f"CPG Network: {name}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Output")
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        return fig


# Factory function to create biomimetic CPG controllers
def create_cpg_controller(controller_type: str, neuromorphic_system: Optional[NeuromorphicSystem] = None) -> BiomimeticCPGController:
    """
    Create a biomimetic CPG controller of a specific type.
    
    Args:
        controller_type: Type of controller (e.g., "quadruped", "hexapod", "flying")
        neuromorphic_system: Optional neuromorphic system for integration
        
    Returns:
        Configured BiomimeticCPGController
    """
    controller = BiomimeticCPGController(neuromorphic_system)
    
    if controller_type == "quadruped":
        # Configure for quadruped locomotion
        controller.configure_for_locomotion("walking")
        
    elif controller_type == "hexapod":
        # Configure for hexapod locomotion (6 legs)
        coupling = np.zeros((6, 6))
        # Alternating tripod gait coupling
        tripod1 = [0, 2, 4]  # Front-left, middle-right, rear-left
        tripod2 = [1, 3, 5]  # Front-right, middle-left, rear-right
        
        # Couple within tripods
        for i in tripod1:
            for j in tripod1:
                if i != j:
                    coupling[i, j] = 0.5
        
        for i in tripod2:
            for j in tripod2:
                if i != j:
                    coupling[i, j] = 0.5
        
        # Anti-phase coupling between tripods
        for i in tripod1:
            for j in tripod2:
                coupling[i, j] = -0.5
                coupling[j, i] = -0.5
        
        controller.add_cpg_network(
            "hexapod_gait",
            CPGType.HOPF_OSCILLATOR,
            num_oscillators=6,
            coupling_matrix=coupling,
            output_mapping={"mapping": ["fl_leg", "fr_leg", "ml_leg", "mr_leg", "rl_leg", "rr_leg"]}
        )
        
    elif controller_type == "flying":
        # Configure for flying with wing articulation
        controller.configure_for_locomotion("flying")
        
        # Add tail control
        controller.add_cpg_network(
            "tail_control",
            CPGType.MATSUOKA_OSCILLATOR,
            num_oscillators=1,
            output_mapping={"mapping": ["tail"]}
        )
        
    elif controller_type == "swimming":
        # Configure for swimming
        controller.configure_for_locomotion("swimming")
        
    elif controller_type == "snake":
        # Configure for snake-like locomotion
        num_segments = 10
        coupling = np.zeros((num_segments, num_segments))
        
        # Nearest-neighbor coupling with phase lag
        for i in range(num_segments-1):
            coupling[i, i+1] = 0.5
            coupling[i+1, i] = -0.5  # Negative coupling for wave propagation
        
        controller.add_cpg_network(
            "snake_locomotion",
            CPGType.KURAMOTO_OSCILLATOR,
            num_oscillators=num_segments,
            coupling_matrix=coupling,
            output_mapping={"mapping": [f"segment_{i}" for i in range(num_segments)]}
        )
    
    return controller


# Integration with neuromorphic system
def integrate_with_neuromorphic_system(cpg_controller: BiomimeticCPGController, 
                                      neuromorphic_system: NeuromorphicSystem) -> bool:
    """
    Integrate a CPG controller with a neuromorphic system.
    
    Args:
        cpg_controller: Biomimetic CPG controller
        neuromorphic_system: Neuromorphic system
        
    Returns:
        Success flag
    """
    if not neuromorphic_system:
        logger.error("No neuromorphic system provided")
        return False
    
    try:
        # Set the neuromorphic system
        cpg_controller.neuromorphic_system = neuromorphic_system
        
        # Register CPG networks as components in the neuromorphic system
        for name, network in cpg_controller.cpg_networks.items():
            # Create a component configuration
            config = {
                "type": "cpg_network",
                "cpg_type": network.cpg_type.value,
                "num_oscillators": network.num_oscillators,
                "coupling_matrix": network.coupling_matrix.tolist(),
                "output_mapping": cpg_controller.output_mappings.get(name, {}),
                "input_mapping": cpg_controller.input_mappings.get(name, {})
            }
            
            # Add component to neuromorphic system
            neuromorphic_system.add_component(f"cpg_{name}", config)
            
            # Create connections between CPG and other components
            if "input_sources" in cpg_controller.input_mappings.get(name, {}):
                for source in cpg_controller.input_mappings[name]["input_sources"]:
                    neuromorphic_system.connect_components(source, f"cpg_{name}")
            
            if "output_targets" in cpg_controller.output_mappings.get(name, {}):
                for target in cpg_controller.output_mappings[name]["output_targets"]:
                    neuromorphic_system.connect_components(f"cpg_{name}", target)
        
        logger.info(f"Integrated CPG controller with neuromorphic system")
        return True
        
    except Exception as e:
        logger.error(f"Failed to integrate with neuromorphic system: {e}")
        return False
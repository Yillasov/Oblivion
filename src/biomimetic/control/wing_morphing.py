#!/usr/bin/env python3
"""
Biomimetic wing morphing control algorithms for adaptive flight.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.biomimetic.control.cpg_models import BiomimeticCPGController, CPGParameters, CPGType
from src.biomimetic.design.wing_structures import WingStructure, JointStructure

logger = get_logger("wing_morphing")


class MorphingMode(Enum):
    """Wing morphing modes."""
    CAMBER = "camber"              # Camber morphing
    TWIST = "twist"                # Wing twist
    SWEEP = "sweep"                # Wing sweep
    SPAN = "span"                  # Wingspan change
    DIHEDRAL = "dihedral"          # Dihedral angle
    ARTICULATED = "articulated"    # Multi-joint articulation
    MEMBRANE = "membrane"          # Membrane tension control


@dataclass
class MorphingParameters:
    """Parameters for wing morphing control."""
    mode: MorphingMode = MorphingMode.CAMBER
    max_deflection: float = 15.0   # Maximum deflection in degrees
    morph_rate: float = 1.0        # Morphing rate (normalized)
    response_time: float = 0.2     # Response time in seconds
    energy_efficiency: float = 0.8 # Energy efficiency factor (0-1)
    coupling_factor: float = 0.5   # Coupling between morphing modes
    
    # Articulated wing parameters
    joint_coordination: Dict[str, float] = None  # Joint coordination factors


class WingMorphingController:
    """Controller for biomimetic wing morphing."""
    
    def __init__(self, cpg_controller: Optional[BiomimeticCPGController] = None,
                wing_structure: Optional[WingStructure] = None):
        """
        Initialize wing morphing controller.
        
        Args:
            cpg_controller: Optional CPG controller for oscillatory patterns
            wing_structure: Optional wing structure model
        """
        self.cpg_controller = cpg_controller
        self.wing_structure = wing_structure
        self.morphing_params = MorphingParameters()
        self.current_morphing = {}
        self.target_morphing = {}
        self.morphing_modes = []
        self.initialized = False
        
        # Initialize default morphing modes
        self._init_default_morphing_modes()
        
        logger.info("Initialized wing morphing controller")
    
    def _init_default_morphing_modes(self):
        """Initialize default morphing modes based on wing structure."""
        # Default modes for all wing types
        self.morphing_modes = [
            MorphingMode.CAMBER,
            MorphingMode.TWIST
        ]
        
        # Add structure-specific modes if wing structure is provided
        if self.wing_structure:
            if hasattr(self.wing_structure, "wing_type"):
                if "BAT" in str(self.wing_structure.wing_type):
                    self.morphing_modes.extend([
                        MorphingMode.MEMBRANE,
                        MorphingMode.ARTICULATED
                    ])
                elif "BIRD" in str(self.wing_structure.wing_type):
                    self.morphing_modes.extend([
                        MorphingMode.SWEEP,
                        MorphingMode.SPAN
                    ])
            
            # Initialize current and target morphing states
            for mode in self.morphing_modes:
                self.current_morphing[mode.value] = 0.0
                self.target_morphing[mode.value] = 0.0
    
    def initialize(self) -> bool:
        """Initialize the controller with CPG networks if needed."""
        if self.initialized:
            return True
        
        try:
            # Initialize CPG controller if available
            if self.cpg_controller and not self.cpg_controller.initialized:
                self.cpg_controller.initialize()
            
            # Set up CPG networks for morphing control if not already present
            if self.cpg_controller:
                self._setup_morphing_cpg_networks()
            
            self.initialized = True
            logger.info("Wing morphing controller initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize wing morphing controller: {e}")
            return False
    
    def _setup_morphing_cpg_networks(self):
        """Set up CPG networks for morphing control."""
        # Check if morphing networks already exist
        if "wing_morphing" in self.cpg_controller.cpg_networks:
            return
        
        # Create CPG network for coordinated morphing
        self.cpg_controller.add_cpg_network(
            "wing_morphing",
            CPGType.HOPF_OSCILLATOR,
            num_oscillators=len(self.morphing_modes),
            output_mapping={"mapping": [mode.value for mode in self.morphing_modes]}
        )
        
        # Set low frequency for smooth morphing
        self.cpg_controller.cpg_networks["wing_morphing"].set_frequency(0.2)
        self.cpg_controller.cpg_networks["wing_morphing"].set_amplitude(0.5)
    
    def set_morphing_target(self, mode: MorphingMode, value: float) -> None:
        """
        Set target value for a morphing mode.
        
        Args:
            mode: Morphing mode to set
            value: Target value (normalized 0-1)
        """
        if mode.value not in self.target_morphing:
            logger.warning(f"Morphing mode {mode.value} not available")
            return
        
        # Clamp value to [0, 1]
        value = max(0.0, min(1.0, value))
        self.target_morphing[mode.value] = value
        logger.debug(f"Set {mode.value} morphing target to {value}")
    
    def update(self, dt: float, flight_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Update morphing control based on flight state.
        
        Args:
            dt: Time step
            flight_state: Current flight state
            
        Returns:
            Dict of current morphing values
        """
        if not self.initialized:
            self.initialize()
        
        # Update morphing values toward targets
        for mode in self.morphing_modes:
            mode_key = mode.value
            current = self.current_morphing[mode_key]
            target = self.target_morphing[mode_key]
            
            # Smooth transition toward target
            rate = self.morphing_params.morph_rate / self.morphing_params.response_time
            self.current_morphing[mode_key] += (target - current) * rate * dt
        
        # Apply CPG modulation if available
        if self.cpg_controller and "wing_morphing" in self.cpg_controller.cpg_networks:
            self._apply_cpg_modulation()
        
        return self.current_morphing
    
    def _apply_cpg_modulation(self):
        """Apply CPG modulation to morphing values."""
        # Get CPG outputs
        cpg_outputs = self.cpg_controller.cpg_networks["wing_morphing"].step(0.01)
        
        # Apply modulation to current morphing values
        for i, mode in enumerate(self.morphing_modes):
            if i < len(cpg_outputs):
                mode_key = mode.value
                # Add small oscillation around current value
                modulation = cpg_outputs[i] * 0.1
                self.current_morphing[mode_key] = max(0.0, min(1.0, 
                                                             self.current_morphing[mode_key] + modulation))
    
    def optimize_for_flight_condition(self, flight_condition: str) -> Dict[str, float]:
        """
        Optimize morphing for a specific flight condition.
        
        Args:
            flight_condition: Flight condition identifier
            
        Returns:
            Dict of optimized morphing targets
        """
        # Predefined morphing configurations for different flight conditions
        configurations = {
            "cruise": {
                MorphingMode.CAMBER.value: 0.3,
                MorphingMode.TWIST.value: 0.2,
                MorphingMode.SWEEP.value: 0.4,
                MorphingMode.SPAN.value: 0.8,
                MorphingMode.DIHEDRAL.value: 0.3,
                MorphingMode.ARTICULATED.value: 0.5,
                MorphingMode.MEMBRANE.value: 0.4
            },
            "maneuver": {
                MorphingMode.CAMBER.value: 0.7,
                MorphingMode.TWIST.value: 0.6,
                MorphingMode.SWEEP.value: 0.3,
                MorphingMode.SPAN.value: 0.6,
                MorphingMode.DIHEDRAL.value: 0.5,
                MorphingMode.ARTICULATED.value: 0.8,
                MorphingMode.MEMBRANE.value: 0.7
            },
            "loiter": {
                MorphingMode.CAMBER.value: 0.4,
                MorphingMode.TWIST.value: 0.3,
                MorphingMode.SWEEP.value: 0.2,
                MorphingMode.SPAN.value: 1.0,
                MorphingMode.DIHEDRAL.value: 0.2,
                MorphingMode.ARTICULATED.value: 0.3,
                MorphingMode.MEMBRANE.value: 0.3
            },
            "highspeed": {
                MorphingMode.CAMBER.value: 0.2,
                MorphingMode.TWIST.value: 0.4,
                MorphingMode.SWEEP.value: 0.8,
                MorphingMode.SPAN.value: 0.5,
                MorphingMode.DIHEDRAL.value: 0.1,
                MorphingMode.ARTICULATED.value: 0.2,
                MorphingMode.MEMBRANE.value: 0.6
            }
        }
        
        # Get configuration for requested flight condition
        config = configurations.get(flight_condition.lower(), configurations["cruise"])
        
        # Set targets for available morphing modes
        for mode in self.morphing_modes:
            if mode.value in config:
                self.set_morphing_target(mode, config[mode.value])
        
        return {k: v for k, v in config.items() if k in self.current_morphing}
    
    def get_joint_commands(self) -> Dict[str, float]:
        """
        Get joint commands for articulated wing control.
        
        Returns:
            Dict of joint angles
        """
        if not self.wing_structure or not hasattr(self.wing_structure, "joints"):
            return {}
        
        joint_commands = {}
        
        # Map morphing values to joint angles
        for joint in self.wing_structure.joints:
            joint_name = joint.name
            
            # Calculate joint angle based on morphing modes
            angle = 0.0
            
            # Apply camber morphing
            if MorphingMode.CAMBER.value in self.current_morphing:
                if "elbow" in joint_name or "wrist" in joint_name:
                    angle += self.current_morphing[MorphingMode.CAMBER.value] * 30.0
            
            # Apply twist morphing
            if MorphingMode.TWIST.value in self.current_morphing:
                if "wrist" in joint_name or "digit" in joint_name:
                    angle += self.current_morphing[MorphingMode.TWIST.value] * 20.0
            
            # Apply sweep morphing
            if MorphingMode.SWEEP.value in self.current_morphing:
                if "shoulder" in joint_name:
                    angle += self.current_morphing[MorphingMode.SWEEP.value] * 40.0
            
            # Apply articulated morphing
            if MorphingMode.ARTICULATED.value in self.current_morphing:
                if "digit" in joint_name:
                    angle += self.current_morphing[MorphingMode.ARTICULATED.value] * 50.0
            
            joint_commands[joint_name] = angle
        
        return joint_commands


class BirdWingMorphingController(WingMorphingController):
    """Specialized controller for bird-inspired wing morphing."""
    
    def __init__(self, cpg_controller=None, wing_structure=None):
        super().__init__(cpg_controller, wing_structure)
        
        # Override with bird-specific morphing modes
        self.morphing_modes = [
            MorphingMode.CAMBER,
            MorphingMode.TWIST,
            MorphingMode.SWEEP,
            MorphingMode.SPAN
        ]
        
        # Initialize current and target morphing states
        for mode in self.morphing_modes:
            self.current_morphing[mode.value] = 0.0
            self.target_morphing[mode.value] = 0.0
    
    def update_feather_configuration(self, flight_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update feather configuration based on flight state.
        
        Args:
            flight_state: Current flight state
            
        Returns:
            Feather configuration parameters
        """
        # Extract relevant flight parameters
        airspeed = flight_state.get("airspeed", 0.0)
        aoa = flight_state.get("angle_of_attack", 0.0)
        
        # Calculate feather parameters
        feather_config = {
            "alula_deployment": 0.0,
            "primary_feather_separation": 0.0,
            "covert_feather_angle": 0.0
        }
        
        # Alula deployment (leading-edge feathers) - activates at high AoA
        if aoa > 10.0:
            feather_config["alula_deployment"] = min(1.0, (aoa - 10.0) / 5.0)
        
        # Primary feather separation - increases with airspeed
        feather_config["primary_feather_separation"] = min(1.0, airspeed / 30.0)
        
        # Covert feather angle - adjusts with camber
        if MorphingMode.CAMBER.value in self.current_morphing:
            feather_config["covert_feather_angle"] = self.current_morphing[MorphingMode.CAMBER.value]
        
        return feather_config


class BatWingMorphingController(WingMorphingController):
    """Specialized controller for bat-inspired wing morphing."""
    
    def __init__(self, cpg_controller=None, wing_structure=None):
        super().__init__(cpg_controller, wing_structure)
        
        # Override with bat-specific morphing modes
        self.morphing_modes = [
            MorphingMode.CAMBER,
            MorphingMode.MEMBRANE,
            MorphingMode.ARTICULATED
        ]
        
        # Initialize current and target morphing states
        for mode in self.morphing_modes:
            self.current_morphing[mode.value] = 0.0
            self.target_morphing[mode.value] = 0.0
        
        # Digit positions for articulated control
        self.digit_positions = {}
    
    def update_membrane_tension(self, flight_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Update membrane tension based on flight state.
        
        Args:
            flight_state: Current flight state
            
        Returns:
            Membrane tension parameters
        """
        # Extract relevant flight parameters
        airspeed = flight_state.get("airspeed", 0.0)
        
        # Calculate membrane tension parameters
        membrane_params = {
            "leading_edge_tension": 0.0,
            "trailing_edge_tension": 0.0,
            "membrane_camber": 0.0
        }
        
        # Leading edge tension - increases with airspeed
        membrane_params["leading_edge_tension"] = min(1.0, airspeed / 20.0)
        
        # Trailing edge tension - controlled by articulated morphing
        if MorphingMode.ARTICULATED.value in self.current_morphing:
            membrane_params["trailing_edge_tension"] = self.current_morphing[MorphingMode.ARTICULATED.value]
        
        # Membrane camber - controlled by camber morphing
        if MorphingMode.CAMBER.value in self.current_morphing:
            membrane_params["membrane_camber"] = self.current_morphing[MorphingMode.CAMBER.value]
        
        return membrane_params
    
    def calculate_digit_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Calculate digit positions for articulated control.
        
        Returns:
            Dict of digit positions (x, y, z)
        """
        digit_positions = {}
        
        # Base positions
        base_positions = {
            "digit1": (0.3, 0.0, 0.0),  # Thumb
            "digit2": (0.5, 0.2, 0.0),  # Index
            "digit3": (0.7, 0.4, 0.0),  # Middle
            "digit4": (0.8, 0.6, 0.0),  # Ring
            "digit5": (0.9, 0.8, 0.0)   # Pinky
        }
        
        # Apply articulation based on morphing values
        articulation = self.current_morphing.get(MorphingMode.ARTICULATED.value, 0.0)
        camber = self.current_morphing.get(MorphingMode.CAMBER.value, 0.0)
        
        for digit, base_pos in base_positions.items():
            x, y, z = base_pos
            
            # Apply articulation - extend digits outward
            if digit in ["digit3", "digit4", "digit5"]:
                y += articulation * 0.2
            
            # Apply camber - curve digits downward
            if digit in ["digit2", "digit3", "digit4", "digit5"]:
                z -= camber * 0.1 * y  # More effect on outer digits
            
            digit_positions[digit] = (x, y, z)
        
        return digit_positions


# Factory function to create wing morphing controllers
def create_wing_morphing_controller(wing_type: str, 
                                   cpg_controller: Optional[BiomimeticCPGController] = None,
                                   wing_structure: Optional[WingStructure] = None) -> WingMorphingController:
    """
    Create a wing morphing controller for a specific wing type.
    
    Args:
        wing_type: Type of wing ("bird", "bat", "insect", "general")
        cpg_controller: Optional CPG controller
        wing_structure: Optional wing structure
        
    Returns:
        Configured WingMorphingController
    """
    if wing_type.lower() == "bird":
        controller = BirdWingMorphingController(cpg_controller, wing_structure)
    elif wing_type.lower() == "bat":
        controller = BatWingMorphingController(cpg_controller, wing_structure)
    else:
        controller = WingMorphingController(cpg_controller, wing_structure)
    
    controller.initialize()
    return controller
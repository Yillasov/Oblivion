"""
Bio-inspired propulsion control system mimicking natural locomotion patterns.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.propulsion.base import PropulsionInterface
from src.control.flight.dragonfly_controller import CompoundEye
from src.propulsion.neuromorphic_control import PropulsionControlInterface


class BioPropulsionMode(Enum):
    """Bio-inspired propulsion modes."""
    IDLE = 0
    CRUISE = 1
    BURST = 2
    GLIDE = 3
    HOVER = 4


@dataclass
class MotionPattern:
    """Natural motion pattern parameters."""
    frequency: float  # Oscillation frequency (Hz)
    amplitude: float  # Motion amplitude
    phase: float  # Phase offset
    duration: float  # Pattern duration (s)
    energy_cost: float  # Normalized energy cost (0-1)


class BioPropulsionController(PropulsionControlInterface):
    """Bio-inspired propulsion controller mimicking natural systems."""
    
    def __init__(self, visual_system: Optional[CompoundEye] = None):
        """Initialize bio-inspired controller."""
        self.mode = BioPropulsionMode.IDLE
        self.visual_system = visual_system or CompoundEye(num_ommatidia=120)
        self.current_pattern: Optional[MotionPattern] = None
        self.energy_level = 1.0
        self.adaptation_rate = 0.2
        self.pattern_memory: List[MotionPattern] = []
        
        # Initialize basic motion patterns
        self.motion_patterns = {
            BioPropulsionMode.CRUISE: MotionPattern(
                frequency=2.0,
                amplitude=0.6,
                phase=0.0,
                duration=5.0,
                energy_cost=0.5
            ),
            BioPropulsionMode.BURST: MotionPattern(
                frequency=5.0,
                amplitude=1.0,
                phase=0.0,
                duration=1.0,
                energy_cost=0.9
            ),
            BioPropulsionMode.GLIDE: MotionPattern(
                frequency=0.5,
                amplitude=0.3,
                phase=np.pi/4,
                duration=8.0,
                energy_cost=0.2
            ),
            BioPropulsionMode.HOVER: MotionPattern(
                frequency=8.0,
                amplitude=0.4,
                phase=np.pi/2,
                duration=3.0,
                energy_cost=0.7
            )
        }
    
    def process_inputs(self, sensor_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process sensor inputs using bio-inspired algorithms."""
        # Process visual information
        if "visual" in sensor_data:
            motion_response = self.visual_system.process_visual_input(
                sensor_data["visual"],
                dt=0.01
            )
        else:
            motion_response = np.zeros(self.visual_system.num_sensors)
            
        # Generate motion pattern based on current mode
        if self.current_pattern:
            t = float(sensor_data.get("time", 0.0))  # Convert to float
            pattern_output = self._generate_pattern(self.current_pattern, t)
        else:
            pattern_output = np.zeros_like(motion_response)
            
        # Combine visual and pattern responses
        control_output = {
            "thrust": pattern_output * (1 + 0.2 * np.mean(motion_response)),
            "frequency": self.current_pattern.frequency if self.current_pattern else 0.0,
            "amplitude": self.current_pattern.amplitude if self.current_pattern else 0.0
        }
        
        return control_output
    
    def adapt(self, performance_metrics: Dict[str, float]) -> None:
        """Adapt control parameters based on performance feedback."""
        if not self.current_pattern:
            return
            
        # Extract relevant metrics
        efficiency = performance_metrics.get("efficiency", 0.0)
        energy_usage = performance_metrics.get("energy_usage", 0.0)
        
        # Adapt pattern parameters
        self.current_pattern.frequency *= (1 + self.adaptation_rate * (efficiency - 0.5))
        self.current_pattern.amplitude *= (1 - self.adaptation_rate * (energy_usage - 0.5))
        
        # Store adapted pattern
        self.pattern_memory.append(self.current_pattern)
        if len(self.pattern_memory) > 10:
            self.pattern_memory.pop(0)
    
    def set_mode(self, mode: BioPropulsionMode, conditions: Dict[str, float]) -> bool:
        """Set propulsion mode based on conditions."""
        if mode == self.mode:
            return True
            
        # Check energy requirements
        if mode in self.motion_patterns:
            required_energy = self.motion_patterns[mode].energy_cost
            if required_energy > self.energy_level:
                return False
                
        # Set new mode and pattern
        self.mode = mode
        if mode in self.motion_patterns:
            self.current_pattern = self.motion_patterns[mode]
            self.energy_level -= 0.1 * self.current_pattern.energy_cost
        else:
            self.current_pattern = None
            
        return True
    
    def _generate_pattern(self, pattern: MotionPattern, t: float) -> np.ndarray:
        """Generate motion pattern output."""
        # Basic sinusoidal pattern with frequency and amplitude modulation
        base_output = pattern.amplitude * np.sin(
            2 * np.pi * pattern.frequency * t + pattern.phase
        )
        
        # Add natural variation
        variation = 0.1 * np.random.randn()
        
        return base_output + variation
"""
Magnetic levitation system for advanced propulsion applications.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.propulsion.base import PropulsionInterface


class MaglevMode(Enum):
    """Magnetic levitation system operating modes."""
    OFF = 0
    STANDBY = 1
    HOVER = 2
    CRUISE = 3
    PRECISION = 4
    EMERGENCY = 5


@dataclass
class MaglevSpecs:
    """Specifications for magnetic levitation system."""
    max_lift_force: float  # Maximum lift force in Newtons
    max_current: float  # Maximum current in Amperes
    coil_resistance: float  # Coil resistance in Ohms
    min_gap: float  # Minimum air gap in mm
    max_gap: float  # Maximum air gap in mm
    response_time: float  # Response time in milliseconds
    stability_factor: float  # Stability factor (0-1)
    power_efficiency: float  # Power efficiency (0-1)


class MagneticLevitationController:
    """Controller for magnetic levitation propulsion system."""
    
    def __init__(self, specs: MaglevSpecs):
        """Initialize magnetic levitation controller."""
        self.specs = specs
        self.mode = MaglevMode.OFF
        self.current_gap = specs.max_gap  # Current air gap in mm
        self.target_gap = specs.max_gap  # Target air gap in mm
        self.current_current = 0.0  # Current in Amperes
        self.current_force = 0.0  # Current lift force in Newtons
        self.stability_control = 1.0  # Stability control factor
        self.power_consumption = 0.0  # Power consumption in kW
        self.temperature = 293.0  # Temperature in K
        self.performance_history: List[Dict[str, float]] = []
    
    def set_mode(self, mode: MaglevMode) -> bool:
        """Set operating mode of magnetic levitation system."""
        if mode == self.mode:
            return True
            
        # Set new mode
        self.mode = mode
        
        # Adjust parameters based on mode
        if mode == MaglevMode.OFF:
            self.target_gap = self.specs.max_gap
            self.stability_control = 0.0
        elif mode == MaglevMode.STANDBY:
            self.target_gap = self.specs.max_gap * 0.9
            self.stability_control = 0.5
        elif mode == MaglevMode.HOVER:
            self.target_gap = self.specs.max_gap * 0.5
            self.stability_control = 0.8
        elif mode == MaglevMode.CRUISE:
            self.target_gap = self.specs.max_gap * 0.3
            self.stability_control = 0.7
        elif mode == MaglevMode.PRECISION:
            self.target_gap = self.specs.min_gap * 1.5
            self.stability_control = 1.0
        elif mode == MaglevMode.EMERGENCY:
            self.target_gap = self.specs.min_gap * 2.0
            self.stability_control = 0.9
            
        return True
    
    def update(self, conditions: Dict[str, float], dt: float) -> Dict[str, Any]:
        """
        Update magnetic levitation system state.
        
        Args:
            conditions: Current operating conditions
            dt: Time step in seconds
            
        Returns:
            Updated system status
        """
        if self.mode == MaglevMode.OFF:
            self.current_current = 0.0
            self.current_force = 0.0
            self.power_consumption = 0.0
            return self._get_status()
            
        # Extract relevant conditions
        payload_mass = conditions.get("payload_mass", 1000.0)  # kg
        external_force = conditions.get("external_force", 0.0)  # N
        vibration = conditions.get("vibration", 0.0)  # m/s²
        
        # Calculate required force (F = m*g + external forces)
        gravity = 9.81  # m/s²
        required_force = payload_mass * gravity + external_force
        
        # Adjust gap based on target and current values
        gap_error = self.target_gap - self.current_gap
        gap_adjustment = gap_error * min(1.0, dt * 1000 / self.specs.response_time)
        self.current_gap += gap_adjustment
        
        # Calculate current based on force and gap
        # Using simplified magnetic force equation: F = k * (I²/gap²)
        # Solving for I: I = sqrt(F * gap² / k)
        k = self.specs.max_lift_force * (self.specs.min_gap/1000)**2 / self.specs.max_current**2
        ideal_current = np.sqrt(required_force * (self.current_gap/1000)**2 / k)
        
        # Apply stability control and limits
        stability_factor = 1.0 + (np.random.randn() * 0.05 * (1.0 - self.stability_control))
        target_current = ideal_current * stability_factor
        
        # Limit current to maximum
        self.current_current = min(target_current, self.specs.max_current)
        
        # Calculate actual force
        self.current_force = k * (self.current_current**2) / (self.current_gap/1000)**2
        
        # Calculate power consumption (P = I²R)
        self.power_consumption = (self.current_current**2 * self.specs.coil_resistance) / 1000  # kW
        
        # Update temperature (simplified model)
        heat_generation = self.power_consumption * (1.0 - self.specs.power_efficiency) * 1000  # W
        self.temperature += heat_generation * dt / 100  # Simplified thermal model
        self.temperature -= (self.temperature - 293.0) * 0.01 * dt  # Cooling
        
        # Record performance
        self.performance_history.append({
            "time": conditions.get("time", 0.0),
            "gap": self.current_gap,
            "current": self.current_current,
            "force": self.current_force,
            "power": self.power_consumption,
            "temperature": self.temperature
        })
        
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)
            
        return self._get_status()
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current status of magnetic levitation system."""
        return {
            "mode": self.mode.name,
            "gap": self.current_gap,
            "target_gap": self.target_gap,
            "current": self.current_current,
            "force": self.current_force,
            "power_consumption": self.power_consumption,
            "temperature": self.temperature,
            "stability": self.stability_control
        }
    
    def adjust_gap(self, target_gap: float) -> bool:
        """Manually adjust target air gap."""
        if target_gap < self.specs.min_gap or target_gap > self.specs.max_gap:
            return False
            
        self.target_gap = target_gap
        return True
    
    def emergency_shutdown(self) -> Dict[str, Any]:
        """Perform emergency shutdown of magnetic levitation system."""
        self.mode = MaglevMode.OFF
        self.current_current = 0.0
        self.target_gap = self.specs.max_gap
        
        return {
            "status": "emergency_shutdown_complete",
            "mode": self.mode.name,
            "current": self.current_current
        }


class MaglevPropulsionSystem(PropulsionInterface):
    """Magnetic levitation propulsion system implementation."""
    
    def __init__(self, maglev_controller: MagneticLevitationController):
        """Initialize magnetic levitation propulsion system."""
        self.controller = maglev_controller
        self.thrust_vector = np.array([0.0, 0.0, 0.0])
        self.guidance_active = False
        
    def set_thrust(self, thrust_vector: np.ndarray) -> bool:
        """Set thrust vector for propulsion."""
        magnitude = np.linalg.norm(thrust_vector)
        if magnitude > self.controller.specs.max_lift_force:
            # Scale down if exceeding maximum
            thrust_vector = thrust_vector * (self.controller.specs.max_lift_force / magnitude)
            
        self.thrust_vector = thrust_vector
        
        # Convert thrust to gap adjustment
        # Lower gap = more force
        force_magnitude = np.linalg.norm(thrust_vector)
        force_ratio = force_magnitude / self.controller.specs.max_lift_force
        
        # Inverse relationship between force and gap
        gap_range = self.controller.specs.max_gap - self.controller.specs.min_gap
        target_gap = float(self.controller.specs.max_gap - (gap_range * force_ratio))
        
        self.controller.adjust_gap(target_gap)
        return True
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the propulsion system."""
        controller_status = self.controller._get_status()
        return {
            **controller_status,
            "thrust_vector": self.thrust_vector.tolist(),
            "guidance_active": self.guidance_active
        }
        
    def update(self, flight_conditions: Dict[str, float], dt: float) -> Dict[str, Any]:
        """Update propulsion system state."""
        return self.controller.update(flight_conditions, dt)
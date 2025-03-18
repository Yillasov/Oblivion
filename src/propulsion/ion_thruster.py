"""
Ion Thruster Control System for UCAV propulsion.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.propulsion.base import PropulsionInterface, PropulsionSpecs, PropulsionType


class IonThrusterMode(Enum):
    """Operating modes for ion thrusters."""
    OFF = 0
    STANDBY = 1
    LOW_POWER = 2
    NORMAL = 3
    HIGH_POWER = 4
    EMERGENCY = 5


@dataclass
class IonThrusterSpecs:
    """Specifications for ion thruster systems."""
    max_power: float  # Maximum power in kW
    specific_impulse: float  # Specific impulse in seconds
    thrust_efficiency: float  # Thrust efficiency (0-1)
    propellant_type: str  # Type of propellant (xenon, krypton, etc.)
    grid_voltage_range: Tuple[float, float]  # Min/max grid voltage in V
    beam_current_range: Tuple[float, float]  # Min/max beam current in A
    mass_flow_range: Tuple[float, float]  # Min/max mass flow rate in mg/s
    thermal_limits: Dict[str, float]  # Thermal limits for components


class IonThrusterController(PropulsionInterface):
    """Ion thruster control system for space-grade propulsion."""
    
    def __init__(self, specs: IonThrusterSpecs):
        """Initialize ion thruster controller."""
        self.specs = specs
        self.mode = IonThrusterMode.OFF
        self.power_level = 0.0
        self.grid_voltage = 0.0
        self.beam_current = 0.0
        self.mass_flow_rate = 0.0
        self.temperature = 293.0  # K
        self.performance_history: List[Dict[str, float]] = []
        self.propulsion_specs = self._create_propulsion_specs()
        
    def initialize(self) -> bool:
        """Initialize the propulsion system."""
        if self.mode != IonThrusterMode.OFF:
            return False
            
        # Set to standby mode
        self.mode = IonThrusterMode.STANDBY
        self.grid_voltage = self.specs.grid_voltage_range[0] * 0.5
        self.beam_current = self.specs.beam_current_range[0]
        self.mass_flow_rate = self.specs.mass_flow_range[0]
        
        return True
    
    def get_specifications(self) -> PropulsionSpecs:
        """Get the physical specifications of the propulsion system."""
        return self.propulsion_specs
    
    def calculate_performance(self, flight_conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance metrics under given flight conditions."""
        # Extract relevant conditions
        altitude = flight_conditions.get("altitude", 0.0)
        ambient_pressure = flight_conditions.get("pressure", 0.0)
        
        # Calculate thrust based on power level and efficiency
        power = self.power_level * self.specs.max_power
        thrust_efficiency = self.specs.thrust_efficiency
        
        # Simple thrust calculation for ion thruster
        # F = 2 * P * η / v_e, where v_e is exhaust velocity
        exhaust_velocity = self.specs.specific_impulse * 9.81  # m/s
        thrust = 2 * power * thrust_efficiency / exhaust_velocity
        
        # Calculate specific impulse (may vary with operating conditions)
        specific_impulse = self.specs.specific_impulse * (0.9 + 0.1 * self.power_level)
        
        # Calculate propellant consumption
        propellant_flow = self.mass_flow_rate / 1000.0  # Convert mg/s to g/s
        
        # Calculate efficiency
        efficiency = thrust_efficiency * (0.8 + 0.2 * self.power_level)
        
        # Record performance data
        performance = {
            "thrust": thrust,
            "specific_impulse": specific_impulse,
            "power": power,
            "efficiency": efficiency,
            "propellant_flow": propellant_flow,
            "grid_voltage": self.grid_voltage,
            "beam_current": self.beam_current,
            "temperature": self.temperature
        }
        
        self.performance_history.append(performance)
        return performance
    
    def set_power_state(self, state: Dict[str, Any]) -> bool:
        """Set the power state of the propulsion system."""
        # Extract power level if provided
        if "power_level" in state:
            power_level = max(0.0, min(1.0, state["power_level"]))
            self.power_level = power_level
            
            # Update operating parameters based on power level
            self._update_operating_parameters()
            
        # Handle specific parameters if provided
        if "grid_voltage" in state:
            self.grid_voltage = max(self.specs.grid_voltage_range[0],
                                  min(state["grid_voltage"], self.specs.grid_voltage_range[1]))
                                  
        if "beam_current" in state:
            self.beam_current = max(self.specs.beam_current_range[0],
                                  min(state["beam_current"], self.specs.beam_current_range[1]))
                                  
        if "mass_flow_rate" in state:
            self.mass_flow_rate = max(self.specs.mass_flow_range[0],
                                    min(state["mass_flow_rate"], self.specs.mass_flow_range[1]))
        
        # Handle fuel flow rate if provided (for compatibility)
        if "fuel_flow_rate" in state:
            self.set_fuel_flow(state["fuel_flow_rate"])
            
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the propulsion system."""
        return {
            "mode": self.mode.name,
            "power_level": self.power_level,
            "grid_voltage": self.grid_voltage,
            "beam_current": self.beam_current,
            "mass_flow_rate": self.mass_flow_rate,
            "temperature": self.temperature,
            "thrust_estimate": self._estimate_thrust(),
            "propellant_type": self.specs.propellant_type
        }
    
    def set_fuel_flow(self, flow_rate: float) -> bool:
        """
        Set the fuel flow rate for the propulsion system.
        
        Args:
            flow_rate: Fuel flow rate in kg/s
            
        Returns:
            Success status
        """
        # Convert kg/s to mg/s for ion thruster propellant
        mass_flow_mg = flow_rate * 1_000_000  # kg/s to mg/s
        
        # Clamp to valid range
        clamped_flow = max(self.specs.mass_flow_range[0],
                          min(mass_flow_mg, self.specs.mass_flow_range[1]))
        
        self.mass_flow_rate = clamped_flow
        
        # Update power level based on flow rate
        flow_range = self.specs.mass_flow_range[1] - self.specs.mass_flow_range[0]
        if flow_range > 0:
            normalized_flow = (clamped_flow - self.specs.mass_flow_range[0]) / flow_range
            # Only update power if significantly different to avoid oscillation
            if abs(normalized_flow - self.power_level) > 0.1:
                self.power_level = normalized_flow
                self._update_operating_parameters()
        
        return True
    
    def set_mode(self, mode: IonThrusterMode) -> bool:
        """Set operating mode of ion thruster."""
        if mode == self.mode:
            return True
            
        # Check for valid mode transitions
        if mode == IonThrusterMode.OFF and self.mode != IonThrusterMode.STANDBY:
            # Can only turn off from standby
            return False
            
        if mode == IonThrusterMode.NORMAL and self.mode == IonThrusterMode.OFF:
            # Can't go directly from OFF to NORMAL
            return False
            
        # Set new mode
        self.mode = mode
        
        # Adjust parameters based on mode
        if mode == IonThrusterMode.OFF:
            self.power_level = 0.0
            self.grid_voltage = 0.0
            self.beam_current = 0.0
            self.mass_flow_rate = 0.0
        elif mode == IonThrusterMode.STANDBY:
            self.power_level = 0.1
            self.grid_voltage = self.specs.grid_voltage_range[0] * 0.5
            self.beam_current = self.specs.beam_current_range[0]
            self.mass_flow_rate = self.specs.mass_flow_range[0]
        elif mode == IonThrusterMode.LOW_POWER:
            self.power_level = 0.3
            self._update_operating_parameters()
        elif mode == IonThrusterMode.NORMAL:
            self.power_level = 0.6
            self._update_operating_parameters()
        elif mode == IonThrusterMode.HIGH_POWER:
            self.power_level = 0.9
            self._update_operating_parameters()
        elif mode == IonThrusterMode.EMERGENCY:
            self.power_level = 1.0
            self._update_operating_parameters()
            
        return True
    
    def _update_operating_parameters(self) -> None:
        """Update operating parameters based on power level."""
        # Update grid voltage
        voltage_range = self.specs.grid_voltage_range
        self.grid_voltage = voltage_range[0] + self.power_level * (voltage_range[1] - voltage_range[0])
        
        # Update beam current
        current_range = self.specs.beam_current_range
        self.beam_current = current_range[0] + self.power_level * (current_range[1] - current_range[0])
        
        # Update mass flow rate
        flow_range = self.specs.mass_flow_range
        self.mass_flow_rate = flow_range[0] + self.power_level * (flow_range[1] - flow_range[0])
    
    def _estimate_thrust(self) -> float:
        """Estimate current thrust output in Newtons."""
        if self.mode == IonThrusterMode.OFF:
            return 0.0
            
        # Simple thrust calculation
        power = self.power_level * self.specs.max_power
        thrust_efficiency = self.specs.thrust_efficiency
        exhaust_velocity = self.specs.specific_impulse * 9.81  # m/s
        
        return 2 * power * thrust_efficiency / exhaust_velocity
    
    def _create_propulsion_specs(self) -> PropulsionSpecs:
        """Create PropulsionSpecs object from ion thruster specs."""
        return PropulsionSpecs(
            propulsion_type=PropulsionType.ION_THRUSTER,
            thrust_rating=self._estimate_thrust() * 1.1,  # Max thrust with margin
            power_rating=self.specs.max_power,
            specific_impulse=self.specs.specific_impulse,
            weight=50.0,  # Example weight in kg
            volume={"length": 0.5, "width": 0.3, "height": 0.3},  # Example dimensions
            thermal_limits=self.specs.thermal_limits,
            cooling_capacity=2.0,  # Example cooling capacity
            thermal_response_time=30.0,  # Example response time
            efficiency_curve={"power_levels": [0.2, 0.4, 0.6, 0.8, 1.0],
                             "efficiency": [0.5, 0.6, 0.7, 0.8, 0.85]},
            thrust_curve={"power_levels": [0.2, 0.4, 0.6, 0.8, 1.0],
                         "thrust": [0.01, 0.03, 0.06, 0.08, 0.1]},
            fuel_consumption_curve={"power_levels": [0.2, 0.4, 0.6, 0.8, 1.0],
                                  "consumption": [0.1, 0.2, 0.3, 0.4, 0.5]},
            operational_envelope={"altitude": {"min": 200000, "max": 35786000},
                                "temperature": {"min": 100, "max": 400}},
            startup_time=120.0,  # Longer startup for ion thrusters
            shutdown_time=60.0,
            throttle_range={"min": 0.2, "max": 1.0},
            altitude_limits={"min": 200000, "max": 35786000},
            temperature_range={"min": 100, "max": 400},
            pressure_range={"min": 0, "max": 0.01},
            mounting_requirements={"orientation": "any", "vibration_isolation": True},
            power_interface={"voltage": 28.0, "current": 10.0},
            control_interface={"protocol": "CAN", "data_rate": 1000000},
            emergency_shutdown_time=10.0,
            safety_margins={"temperature": 0.2, "voltage": 0.1},
            failure_modes=["grid_erosion", "neutralizer_failure", "power_loss"],
            service_interval=10000.0,  # Hours
            critical_components=["grid_assembly", "neutralizer", "power_processing_unit"],
            lifetime_hours=50000.0
        )
"""
Base interfaces and abstract classes for UCAV propulsion systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class PropulsionType(Enum):
    HYBRID_ELECTRIC = "hybrid_electric"
    SOLAR_ELECTRIC = "solar_electric"
    FUEL_CELL = "fuel_cell"
    SCRAMJET = "scramjet"
    PULSE_DETONATION = "pulse_detonation"
    ION_THRUSTER = "ion_thruster"
    PLASMA_THRUSTER = "plasma_thruster"
    NUCLEAR_THERMAL = "nuclear_thermal"
    BIO_INSPIRED = "bio_inspired"
    MAGNETIC_LEVITATION = "magnetic_levitation"


@dataclass
class PropulsionSpecs:
    """Specifications for propulsion systems."""
    # Basic specifications
    propulsion_type: PropulsionType
    thrust_rating: float  # Maximum thrust in Newtons
    power_rating: float  # Power rating in kW
    specific_impulse: float  # Specific impulse in seconds
    weight: float  # Weight in kg
    volume: Dict[str, float]  # Dimensions in meters
    
    # Thermal characteristics
    thermal_limits: Dict[str, float]  # Temperature limits
    cooling_capacity: float  # Cooling capacity in kW
    thermal_response_time: float  # Thermal response time in seconds
    
    # Performance curves
    efficiency_curve: Dict[str, List[float]]  # Efficiency at different operating points
    thrust_curve: Dict[str, List[float]]  # Thrust profile at different conditions
    fuel_consumption_curve: Dict[str, List[float]]  # Fuel consumption rates
    
    # Operational parameters
    operational_envelope: Dict[str, Dict[str, float]]  # Operating limits
    startup_time: float  # Time to full power in seconds
    shutdown_time: float  # Time to safe state in seconds
    throttle_range: Dict[str, float]  # Min/max throttle settings
    
    # Environmental parameters
    altitude_limits: Dict[str, float]  # Operating altitude range
    temperature_range: Dict[str, float]  # Ambient temperature range
    pressure_range: Dict[str, float]  # Operating pressure range
    
    # Integration requirements
    mounting_requirements: Dict[str, Any]  # Mounting specifications
    power_interface: Dict[str, Any]  # Power interface specifications
    control_interface: Dict[str, Any]  # Control system requirements
    
    # Safety parameters
    emergency_shutdown_time: float  # Emergency shutdown time in seconds
    safety_margins: Dict[str, float]  # Safety thresholds
    failure_modes: List[str]  # Known failure modes
    
    # Maintenance parameters
    service_interval: float  # Service interval in hours
    critical_components: List[str]  # List of critical components
    lifetime_hours: float  # Expected operational lifetime


class PropulsionInterface(ABC):
    """Base interface for all propulsion systems."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the propulsion system."""
        pass
    
    @abstractmethod
    def get_specifications(self) -> PropulsionSpecs:
        """Get the physical specifications of the propulsion system."""
        pass
    
    @abstractmethod
    def calculate_performance(self, flight_conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance metrics under given flight conditions."""
        pass
    
    @abstractmethod
    def set_power_state(self, state: Dict[str, Any]) -> bool:
        """Set the power state of the propulsion system."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the propulsion system."""
        pass
    
    @abstractmethod
    def set_fuel_flow(self, flow_rate: float) -> bool:
        """
        Set the fuel flow rate for the propulsion system.
        
        Args:
            flow_rate: Fuel flow rate in kg/s
            
        Returns:
            Success status
        """
        pass


class NeuromorphicPropulsion(PropulsionInterface):
    """Base class for propulsion systems with neuromorphic control capabilities."""
    
    def __init__(self, hardware_interface=None):
        """
        Initialize a neuromorphic propulsion system.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
        """
        self.hardware_interface = hardware_interface
        self.initialized = False
        self.specs = None
        self.status = {
            "active": False,
            "power_level": 0.0,
            "temperature": 20.0,
            "thrust": 0.0,
            "efficiency": 0.0,
            "health": 1.0
        }
    
    def initialize(self) -> bool:
        """Initialize the neuromorphic propulsion system."""
        if self.hardware_interface:
            self.initialized = True
            return True
        return False
    
    @abstractmethod
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using neuromorphic computing.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Dict containing processed results
        """
        pass
    
    @abstractmethod
    def optimize_performance(self, flight_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize propulsion performance for given flight conditions.
        
        Args:
            flight_conditions: Current flight conditions
            
        Returns:
            Dict containing optimization results
        """
        pass
    
    @abstractmethod
    def train(self, training_data: Dict[str, Any]) -> bool:
        """
        Train the neuromorphic control system.
        
        Args:
            training_data: Data for training
            
        Returns:
            Success status
        """
        pass
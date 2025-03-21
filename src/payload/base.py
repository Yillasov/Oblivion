"""
Base classes and interfaces for UCAV payload systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class PayloadSpecs:
    """Specifications for a payload system."""
    weight: float  # Weight in kg
    volume: Dict[str, float]  # Dimensions in meters (length, width, height)
    power_requirements: float  # Power requirements in watts
    mounting_points: List[str]  # Required mounting points on the UCAV


"""Base interfaces and data structures for payload systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class PayloadSpecs:
    """Specifications for a payload system."""
    weight: float  # Weight in kg
    volume: Dict[str, float]  # Dimensions in meters (length, width, height)
    power_requirements: float  # Power requirements in watts
    mounting_points: List[str]


class PayloadInterface(ABC):
    """Base interface for all payload systems."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the payload system."""
        pass
    
    @abstractmethod
    def get_specifications(self) -> PayloadSpecs:
        """Get the physical specifications of the payload."""
        pass
    
    @abstractmethod
    def calculate_impact(self) -> Dict[str, float]:
        """Calculate the impact of this payload on UCAV performance."""
        pass
    
    @abstractmethod
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """Deploy or activate the payload."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the payload."""
        pass
    
    @abstractmethod
    def set_power_level(self, power_level: float) -> bool:
        """
        Set the power level for this payload system.
        
        Args:
            power_level: Power level as a percentage (0-100)
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def set_power(self, power_ratio: float) -> bool:
        """
        Set the power ratio for this payload system.
        
        Args:
            power_ratio: Power ratio (0.0-1.0)
            
        Returns:
            Success status
        """
        pass


class NeuromorphicPayload(PayloadInterface):
    """Base class for payloads with neuromorphic capabilities."""
    
    def __init__(self, hardware_interface=None):
        """
        Initialize a neuromorphic payload.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
        """
        self.hardware_interface = hardware_interface
        self.initialized = False
        self.specs = None
    
    def initialize(self) -> bool:
        """Initialize the neuromorphic payload system."""
        if self.hardware_interface:
            # Initialize connection to neuromorphic hardware
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
    def train(self, training_data: Dict[str, Any]) -> bool:
        """
        Train the neuromorphic components of the payload.
        
        Args:
            training_data: Data for training
            
        Returns:
            Success status
        """
        pass
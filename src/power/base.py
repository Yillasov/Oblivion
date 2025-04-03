"""
Base power supply systems for UCAV platforms.

This module provides the foundation for all power supply systems
with neuromorphic integration capabilities.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from src.core.integration.neuromorphic_system import NeuromorphicSystem


class PowerSupplyType(Enum):
    """Types of power supply systems."""
    WIRELESS = "wireless_power_transmission"
    MICRO_NUCLEAR = "micro_nuclear_reactor"
    SOLID_STATE_BATTERY = "solid_state_battery"
    SUPERCAPACITOR = "supercapacitor_array"
    SOLAR = "solar_thin_film"
    HYDROGEN_FUEL_CELL = "hydrogen_fuel_cell"
    KINETIC = "kinetic_energy_harvesting"
    THERMAL = "thermal_energy_converter"
    PIEZOELECTRIC = "piezoelectric_generator"
    BIOFUEL = "biofuel_generator"


@dataclass
class PowerSupplySpecs:
    """Specifications for power supply systems."""
    weight: float  # Weight in kg
    volume: Dict[str, float]  # Volume specifications in meters
    power_output: float  # Maximum power output in kW
    energy_density: float  # Energy density in Wh/kg
    efficiency: float  # Efficiency as a decimal (0-1)
    lifespan: float  # Expected lifespan in hours
    response_time: float  # Response time in milliseconds
    mounting_points: List[str]  # Valid mounting locations
    additional_specs: Dict[str, Any] = field(default_factory=dict)


class PowerSupplyInterface(ABC):
    """Base interface for all power supply systems."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the power supply system."""
        pass
    
    @abstractmethod
    def get_specifications(self) -> PowerSupplySpecs:
        """Get the physical specifications of the power supply system."""
        pass
    
    @abstractmethod
    def calculate_output(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate output metrics under given conditions."""
        pass
    
    @abstractmethod
    def set_output_level(self, level: float) -> bool:
        """
        Set the output level of the power supply system.
        
        Args:
            level: Output level as a percentage (0-100)
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the power supply system."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        Safely shutdown the power supply system.
        
        Returns:
            Success status
        """
        pass


class NeuromorphicPowerSupply(PowerSupplyInterface):
    """Base class for neuromorphic power supply systems."""
    
    def __init__(self, hardware_interface=None):
        """
        Initialize a neuromorphic power supply system.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
        """
        self.hardware_interface = hardware_interface
        self.neuromorphic_system = NeuromorphicSystem(hardware_interface)
        self.initialized = False
        self.specs = None
        self.status = {
            "active": False,
            "output_level": 0.0,
            "temperature": 20.0,
            "current_output": 0.0,
            "efficiency": 0.0,
            "health": 1.0
        }
    
    def initialize(self) -> bool:
        """Initialize the power supply system."""
        if self.initialized:
            return True
            
        try:
            self.neuromorphic_system.initialize()
            self.initialized = True
            self.status["active"] = True
            return True
        except Exception as e:
            self.status["error"] = str(e)
            return False
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using neuromorphic computing."""
        if not self.initialized:
            return {"error": "System not initialized"}
        
        try:
            return self.neuromorphic_system.process_data(input_data)
        except Exception as e:
            return {"error": str(e)}
    
    def train(self, training_data: Dict[str, Any]) -> bool:
        """Train the neuromorphic system with provided data."""
        if not self.initialized:
            return False
            
        try:
            # Instead of calling a non-existent train method, we'll use the existing
            # components and learning algorithms in the neuromorphic system
            
            # Add a learning algorithm if needed
            if "algorithm" in training_data and "algorithm_name" in training_data:
                self.neuromorphic_system.add_learning_algorithm(
                    training_data["algorithm_name"], 
                    training_data["algorithm"]
                )
            
            # Process training data through the system
            if "inputs" in training_data:
                self.neuromorphic_system.process_data(training_data["inputs"])
                
            # If there are specific components to train
            if "component_data" in training_data:
                for component_name, data in training_data["component_data"].items():
                    if component_name in self.neuromorphic_system.components:
                        component = self.neuromorphic_system.components[component_name]
                        if hasattr(component, 'train'):
                            component.train(data)
            
            return True
        except Exception as e:
            self.status["error"] = f"Training error: {str(e)}"
            return False
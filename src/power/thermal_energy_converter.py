from typing import Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger("thermal_energy_converter")

@dataclass
class ThermalConverterSpecs:
    """Specifications for thermal energy converter."""
    max_output: float  # Maximum power output in kW
    efficiency: float  # Conversion efficiency (0-1)
    weight: float  # Weight in kg
    operating_temperature: Tuple[float, float]  # Min/max operating temperature in K

class ThermalEnergyConverter:
    """Thermal energy conversion system."""
    
    def __init__(self, 
                 converter_id: str,
                 specs: ThermalConverterSpecs):
        """
        Initialize thermal energy converter.
        
        Args:
            converter_id: Unique identifier
            specs: Thermal converter specifications
        """
        self.converter_id = converter_id
        self.specs = specs
        self.current_output = 0.0  # Current power output in kW
        self.status = {"active": False, "error": None}
        
        logger.info(f"Thermal energy converter '{converter_id}' initialized with {self.specs.max_output} kW max output")

    def calculate_output(self, temperature_difference: float) -> float:
        """
        Calculate power output based on temperature difference.
        
        Args:
            temperature_difference: Difference between hot and cold side temperatures
            
        Returns:
            Current power output in kW
        """
        if not self.status["active"]:
            return 0.0
        
        # Check if temperature difference is within operating range
        if temperature_difference < self.specs.operating_temperature[0] or temperature_difference > self.specs.operating_temperature[1]:
            self.current_output = 0.0
        else:
            # Calculate output based on temperature difference
            output = self.specs.max_output * (temperature_difference / self.specs.operating_temperature[1]) * self.specs.efficiency
            self.current_output = min(output, self.specs.max_output)
        
        logger.info(f"Thermal energy converter '{self.converter_id}' output calculated: {self.current_output:.2f} kW")
        
        return self.current_output

    def activate(self) -> bool:
        """Activate the thermal energy converter."""
        self.status["active"] = True
        logger.info(f"Thermal energy converter '{self.converter_id}' activated")
        return True

    def deactivate(self) -> bool:
        """Deactivate the thermal energy converter."""
        self.status["active"] = False
        logger.info(f"Thermal energy converter '{self.converter_id}' deactivated")
        return True
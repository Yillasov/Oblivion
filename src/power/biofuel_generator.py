from typing import Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("biofuel_generator")

@dataclass
class BiofuelGeneratorSpecs:
    """Specifications for biofuel generator."""
    max_output: float  # Maximum power output in kW
    efficiency: float  # Conversion efficiency (0-1)
    fuel_type: str  # Type of biofuel used
    fuel_consumption_rate: float  # Fuel consumption rate in kg/h

class BiofuelGenerator:
    """Biofuel energy conversion system."""
    
    def __init__(self, 
                 generator_id: str,
                 specs: BiofuelGeneratorSpecs):
        """
        Initialize biofuel generator.
        
        Args:
            generator_id: Unique identifier
            specs: Biofuel generator specifications
        """
        self.generator_id = generator_id
        self.specs = specs
        self.current_output = 0.0  # Current power output in kW
        self.status = {"active": False, "error": None}
        
        logger.info(f"Biofuel generator '{generator_id}' initialized with {self.specs.max_output} kW max output")

    def calculate_output(self, fuel_available: float) -> float:
        """
        Calculate power output based on available fuel.
        
        Args:
            fuel_available: Amount of fuel available in kg
            
        Returns:
            Current power output in kW
        """
        if not self.status["active"]:
            return 0.0
        
        # Check if there is enough fuel to generate power
        if fuel_available <= 0:
            self.current_output = 0.0
        else:
            # Calculate output based on fuel availability
            output = self.specs.max_output * (fuel_available / self.specs.fuel_consumption_rate) * self.specs.efficiency
            self.current_output = min(output, self.specs.max_output)
        
        logger.info(f"Biofuel generator '{self.generator_id}' output calculated: {self.current_output:.2f} kW")
        
        return self.current_output

    def activate(self) -> bool:
        """Activate the biofuel generator."""
        self.status["active"] = True
        logger.info(f"Biofuel generator '{self.generator_id}' activated")
        return True

    def deactivate(self) -> bool:
        """Deactivate the biofuel generator."""
        self.status["active"] = False
        logger.info(f"Biofuel generator '{self.generator_id}' deactivated")
        return True
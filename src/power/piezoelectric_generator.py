from typing import Dict, Any
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger("piezoelectric_generator")

@dataclass
class PiezoelectricGeneratorSpecs:
    """Specifications for piezoelectric generator."""
    max_output: float  # Maximum power output in kW
    efficiency: float  # Conversion efficiency (0-1)
    weight: float  # Weight in kg
    response_time: float  # Response time in seconds
    min_stress_threshold: float  # Minimum mechanical stress threshold for generation

class PiezoelectricGenerator:
    """Piezoelectric energy conversion system."""
    
    def __init__(self, 
                 generator_id: str,
                 specs: PiezoelectricGeneratorSpecs):
        """
        Initialize piezoelectric generator.
        
        Args:
            generator_id: Unique identifier
            specs: Piezoelectric generator specifications
        """
        self.generator_id = generator_id
        self.specs = specs
        self.current_output = 0.0  # Current power output in kW
        self.status = {"active": False, "error": None}
        
        logger.info(f"Piezoelectric generator '{generator_id}' initialized with {self.specs.max_output} kW max output")

    def calculate_output(self, stress_level: float) -> float:
        """
        Calculate power output based on mechanical stress level.
        
        Args:
            stress_level: Mechanical stress applied to the piezoelectric material
            
        Returns:
            Current power output in kW
        """
        if not self.status["active"]:
            return 0.0
        
        # Check if stress level is above threshold
        if stress_level < self.specs.min_stress_threshold:
            self.current_output = 0.0
        else:
            # Calculate output based on stress level
            output = self.specs.max_output * (stress_level / 10.0) * self.specs.efficiency
            self.current_output = min(output, self.specs.max_output)
        
        logger.info(f"Piezoelectric generator '{self.generator_id}' output calculated: {self.current_output:.2f} kW")
        
        return self.current_output

    def activate(self) -> bool:
        """Activate the piezoelectric generator."""
        self.status["active"] = True
        logger.info(f"Piezoelectric generator '{self.generator_id}' activated")
        return True

    def deactivate(self) -> bool:
        """Deactivate the piezoelectric generator."""
        self.status["active"] = False
        logger.info(f"Piezoelectric generator '{self.generator_id}' deactivated")
        return True
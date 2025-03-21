from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("hybrid_power_system")

@dataclass
class HybridPowerSystemSpecs:
    """Specifications for hybrid power system."""
    max_output: float  # Maximum power output in kW
    efficiency: float  # Overall system efficiency (0-1)
    component_ratios: Dict[str, float]  # Ratios of components like electric, combustion, solar, etc.

class HybridPowerSystem:
    """Hybrid power system integrating multiple power sources."""
    
    def __init__(self, 
                 system_id: str,
                 specs: HybridPowerSystemSpecs,
                 power_sources: Dict[str, Any]):
        """
        Initialize hybrid power system.
        
        Args:
            system_id: Unique identifier
            specs: Hybrid power system specifications
            power_sources: Dictionary of power sources
        """
        self.system_id = system_id
        self.specs = specs
        self.power_sources = power_sources
        self.current_output = 0.0  # Current power output in kW
        self.status = {"active": False, "error": None}
        
        logger.info(f"Hybrid power system '{system_id}' initialized with {self.specs.max_output} kW max output")

    def calculate_output(self, conditions: Dict[str, float]) -> float:
        """
        Calculate power output based on environmental conditions and component ratios.
        
        Args:
            conditions: Environmental conditions
            
        Returns:
            Current power output in kW
        """
        if not self.status["active"]:
            return 0.0
        
        # Calculate output for each power source
        total_output = 0.0
        for source_id, source in self.power_sources.items():
            ratio = self.specs.component_ratios.get(source_id, 0.0)
            source_output = source.calculate_output(conditions) * ratio
            total_output += source_output
        
        # Apply overall efficiency
        self.current_output = min(total_output * self.specs.efficiency, self.specs.max_output)
        
        logger.info(f"Hybrid power system '{self.system_id}' output calculated: {self.current_output:.2f} kW")
        
        return self.current_output

    def activate(self) -> bool:
        """Activate the hybrid power system."""
        self.status["active"] = True
        logger.info(f"Hybrid power system '{self.system_id}' activated")
        return True

    def deactivate(self) -> bool:
        """Deactivate the hybrid power system."""
        self.status["active"] = False
        logger.info(f"Hybrid power system '{self.system_id}' deactivated")
        return True
from typing import Dict, Any, List
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger("thin_film_solar")

@dataclass
class ThinFilmSolarSpecs:
    """Specifications for thin-film solar cell system."""
    area: float  # Solar cell area in m²
    efficiency: float  # Cell efficiency (0-1)
    weight: float  # Weight in kg
    max_output: float  # Maximum power output in kW
    degradation_rate: float  # Annual degradation rate

class ThinFilmSolarCell:
    """Thin-film solar cell system."""
    
    def __init__(self, 
                 cell_id: str,
                 specs: ThinFilmSolarSpecs):
        """
        Initialize thin-film solar cell.
        
        Args:
            cell_id: Unique identifier
            specs: Thin-film solar specifications
        """
        self.cell_id = cell_id
        self.specs = specs
        self.current_output = 0.0  # Current power output in kW
        self.status = {"active": False, "error": None}
        
        logger.info(f"Thin-film solar cell '{cell_id}' initialized with {self.specs.area} m² area")

    def calculate_output(self, conditions: Dict[str, float]) -> float:
        """
        Calculate solar power output based on conditions.
        
        Args:
            conditions: Environmental conditions
            
        Returns:
            Current power output in kW
        """
        if not self.status["active"]:
            return 0.0
        
        # Extract relevant conditions
        altitude = conditions.get("altitude", 0)
        cloud_cover = conditions.get("cloud_cover", 0)
        sun_angle = conditions.get("sun_angle", 90)
        
        # Base output calculation
        altitude_factor = min(1.2, 1.0 + (altitude / 10000) * 0.2)
        cloud_factor = 1.0 - (cloud_cover * 0.8)
        angle_factor = np.sin(np.radians(sun_angle))
        angle_factor = max(0.1, angle_factor)
        
        # Calculate output
        base_output = self.specs.area * self.specs.efficiency
        current_output = base_output * altitude_factor * cloud_factor * angle_factor
        
        # Limit to max output
        self.current_output = min(current_output, self.specs.max_output)
        
        logger.info(f"Thin-film solar cell '{self.cell_id}' output calculated: {self.current_output:.2f} kW")
        
        return self.current_output

class ThinFilmSolarArray:
    """Array of thin-film solar cells for increased capacity."""
    
    def __init__(self, array_id: str, num_cells: int = 10):
        """
        Initialize thin-film solar array.
        
        Args:
            array_id: Unique identifier
            num_cells: Number of cells in array
        """
        self.array_id = array_id
        self.cells: Dict[str, ThinFilmSolarCell] = {}
        
        # Create cells
        for i in range(num_cells):
            cell_id = f"{array_id}_cell_{i}"
            self.cells[cell_id] = ThinFilmSolarCell(
                cell_id=cell_id,
                specs=ThinFilmSolarSpecs(
                    area=1.0,  # m²
                    efficiency=0.15,  # 15%
                    weight=0.5,  # kg
                    max_output=0.2,  # kW
                    degradation_rate=0.01  # 1% per year
                )
            )
        
        self.active = False
        self.total_output = 0.0
    
    def initialize(self) -> bool:
        """Initialize all cells in the array."""
        success = True
        for cell in self.cells.values():
            cell.status["active"] = True
        
        if success:
            self.active = True
            self._update_array_status()
        
        return success
    
    def _update_array_status(self) -> None:
        """Update array status based on individual cells."""
        total_output = sum(c.current_output for c in self.cells.values())
        self.total_output = total_output
    
    def calculate_array_output(self, conditions: Dict[str, float]) -> float:
        """
        Calculate total output of the solar array.
        
        Args:
            conditions: Environmental conditions
            
        Returns:
            Total power output in kW
        """
        if not self.active:
            return 0.0
        
        total_output = 0.0
        for cell in self.cells.values():
            total_output += cell.calculate_output(conditions)
        
        self._update_array_status()
        
        logger.info(f"Thin-film solar array '{self.array_id}' total output: {self.total_output:.2f} kW")
        
        return self.total_output
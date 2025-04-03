import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enum import Enum
from typing import Dict, Any, List, Optional
from src.manufacturing.exceptions import ManufacturingValidationError

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class DesignValidationResult:
    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.is_valid: bool = True
    
    def add_issue(self, message: str, severity: ValidationSeverity, location: Optional[Dict[str, float]] = None):
        self.issues.append({
            "message": message,
            "severity": severity.value,
            "location": location
        })
        if severity == ValidationSeverity.ERROR:
            self.is_valid = False

class DesignValidator:
    """Validates CAD designs against manufacturing constraints."""
    
    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints
        self.min_wall_thickness = constraints.get("min_wall_thickness", 1.0)  # mm
        self.max_dimensions = constraints.get("max_dimensions", {
            "x": 1000.0,  # mm
            "y": 1000.0,  # mm
            "z": 1000.0   # mm
        })
    
    def validate_design(self, design_data: Dict[str, Any]) -> DesignValidationResult:
        """
        Validate a design against manufacturing constraints.
        
        Args:
            design_data: CAD design data
            
        Returns:
            DesignValidationResult containing validation results
        """
        result = DesignValidationResult()
        
        # Validate dimensions
        self._validate_dimensions(design_data, result)
        
        # Validate wall thickness
        self._validate_wall_thickness(design_data, result)
        
        # Validate structural integrity
        self._validate_structural_integrity(design_data, result)
        
        return result
    
    def _validate_dimensions(self, design_data: Dict[str, Any], result: DesignValidationResult):
        """Validate overall dimensions."""
        dimensions = design_data.get("dimensions", {})
        
        for axis in ["x", "y", "z"]:
            dim = dimensions.get(axis, 0)
            max_dim = self.max_dimensions[axis]
            
            if dim > max_dim:
                result.add_issue(
                    f"Dimension {axis} ({dim}mm) exceeds maximum allowed ({max_dim}mm)",
                    ValidationSeverity.ERROR
                )
    
    def _validate_wall_thickness(self, design_data: Dict[str, Any], result: DesignValidationResult):
        """Validate minimum wall thickness."""
        thickness = design_data.get("min_wall_thickness", 0)
        
        if thickness < self.min_wall_thickness:
            result.add_issue(
                f"Wall thickness ({thickness}mm) below minimum allowed ({self.min_wall_thickness}mm)",
                ValidationSeverity.ERROR
            )
    
    def _validate_structural_integrity(self, design_data: Dict[str, Any], result: DesignValidationResult):
        """Basic structural integrity checks."""
        # Add warning for unsupported overhangs
        if design_data.get("has_overhangs", False):
            result.add_issue(
                "Design contains unsupported overhangs",
                ValidationSeverity.WARNING
            )
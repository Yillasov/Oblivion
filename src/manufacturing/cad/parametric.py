from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np

class ParameterType(Enum):
    LENGTH = "length"
    ANGLE = "angle"
    RATIO = "ratio"
    COUNT = "count"

@dataclass
class DesignParameter:
    name: str
    type: ParameterType
    value: float
    min_value: float
    max_value: float
    dependencies: List[str] = []
    
    def validate(self) -> bool:
        """Validate parameter value against constraints."""
        return self.min_value <= self.value <= self.max_value

class ParametricDesign:
    """Handles parametric design capabilities for UCAV components."""
    
    def __init__(self):
        self.parameters: Dict[str, DesignParameter] = {}
        self.relationships: Dict[str, str] = {}  # Parameter relationships
        
    def add_parameter(self, param: DesignParameter) -> None:
        """Add a design parameter."""
        self.parameters[param.name] = param
    
    def set_parameter(self, name: str, value: float) -> bool:
        """Set parameter value and update dependencies."""
        if name not in self.parameters:
            return False
            
        param = self.parameters[name]
        param.value = value
        
        if not param.validate():
            return False
            
        self._update_dependencies(name)
        return True
    
    def _update_dependencies(self, param_name: str) -> None:
        """Update dependent parameters."""
        for name, param in self.parameters.items():
            if param.dependencies and param_name in param.dependencies:
                self._calculate_dependent_value(name)
    
    def _calculate_dependent_value(self, param_name: str) -> None:
        """Calculate value for dependent parameter."""
        param = self.parameters[param_name]
        if not param.dependencies:
            return
            
        # Example relationship calculation
        if param.type == ParameterType.RATIO:
            dependent_param = self.parameters[param.dependencies[0]]
            param.value = dependent_param.value * param.value

class UCAVParametricDesign(ParametricDesign):
    """UCAV-specific parametric design handler."""
    
    def __init__(self):
        super().__init__()
        self._initialize_default_parameters()
    
    def _initialize_default_parameters(self):
        """Initialize default UCAV design parameters."""
        self.add_parameter(DesignParameter(
            name="wingspan",
            type=ParameterType.LENGTH,
            value=12000.0,  # mm
            min_value=8000.0,
            max_value=16000.0
        ))
        
        self.add_parameter(DesignParameter(
            name="fuselage_length",
            type=ParameterType.LENGTH,
            value=15000.0,  # mm
            min_value=10000.0,
            max_value=20000.0
        ))
        
        self.add_parameter(DesignParameter(
            name="wing_sweep",
            type=ParameterType.ANGLE,
            value=35.0,  # degrees
            min_value=20.0,
            max_value=50.0
        ))
        
        self.add_parameter(DesignParameter(
            name="aspect_ratio",
            type=ParameterType.RATIO,
            value=6.0,
            min_value=4.0,
            max_value=8.0,
            dependencies=["wingspan"]
        ))
    
    def generate_design(self) -> Dict[str, Any]:
        """Generate design data from parameters."""
        return {
            "dimensions": self._calculate_dimensions(),
            "aerodynamics": self._calculate_aerodynamics(),
            "parameters": {name: param.value for name, param in self.parameters.items()}
        }
    
    def _calculate_dimensions(self) -> Dict[str, float]:
        """Calculate overall dimensions based on parameters."""
        return {
            "wingspan": self.parameters["wingspan"].value,
            "length": self.parameters["fuselage_length"].value,
            "height": self.parameters["fuselage_length"].value * 0.15,  # Approximate
            "wing_area": (self.parameters["wingspan"].value ** 2) / 
                        self.parameters["aspect_ratio"].value
        }
    
    def _calculate_aerodynamics(self) -> Dict[str, float]:
        """Calculate basic aerodynamic properties."""
        wing_sweep = np.radians(self.parameters["wing_sweep"].value)
        return {
            "sweep_angle": self.parameters["wing_sweep"].value,
            "effective_wingspan": self.parameters["wingspan"].value * np.cos(wing_sweep),
            "aspect_ratio": self.parameters["aspect_ratio"].value
        }
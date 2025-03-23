from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from enum import Enum
import numpy as np
import math

class ParameterType(Enum):
    LENGTH = "length"
    ANGLE = "angle"
    RATIO = "ratio"
    COUNT = "count"

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class RelationshipType(Enum):
    """Types of parameter relationships."""
    PROPORTIONAL = "proportional"  # y = k*x
    INVERSE = "inverse"            # y = k/x
    QUADRATIC = "quadratic"        # y = k*x²
    CUSTOM = "custom"              # Custom function

@dataclass
class ParameterValidationResult:
    """Result of parameter validation."""
    valid: bool
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_issue(self, message: str, severity: ValidationSeverity, param_name: str):
        self.issues.append({
            "message": message,
            "severity": severity.value,
            "parameter": param_name
        })
        if severity == ValidationSeverity.ERROR:
            self.valid = False

@dataclass
class ParameterRelationship:
    """Defines a relationship between parameters."""
    source_param: str
    target_param: str
    relationship_type: RelationshipType
    coefficient: float = 1.0
    custom_function: Optional[Callable[[float], float]] = None
    
    def apply(self, source_value: float) -> float:
        """Apply the relationship to calculate target value."""
        if self.relationship_type == RelationshipType.PROPORTIONAL:
            return source_value * self.coefficient
        elif self.relationship_type == RelationshipType.INVERSE:
            return self.coefficient / source_value if source_value != 0 else 0
        elif self.relationship_type == RelationshipType.QUADRATIC:
            return self.coefficient * (source_value ** 2)
        elif self.relationship_type == RelationshipType.CUSTOM and self.custom_function:
            return self.custom_function(source_value)
        return source_value

@dataclass
class DesignParameter:
    name: str
    type: ParameterType
    value: float
    min_value: float
    max_value: float
    dependencies: List[str] = field(default_factory=list)
    validation_rules: List[Callable[[float], Tuple[bool, str]]] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate parameter value against constraints."""
        # Check range constraints
        if not (self.min_value <= self.value <= self.max_value):
            return False, f"Value {self.value} outside allowed range [{self.min_value}, {self.max_value}]"
        
        # Check custom validation rules
        for rule in self.validation_rules:
            valid, message = rule(self.value)
            if not valid:
                return False, message
                
        return True, None

class ParametricDesign:
    """Handles parametric design capabilities for UCAV components."""
    
    def __init__(self):
        self.parameters: Dict[str, DesignParameter] = {}
        self.relationships: Dict[str, List[ParameterRelationship]] = {}
        self.parameter_groups: Dict[str, Set[str]] = {}
        
    def add_parameter(self, param: DesignParameter) -> None:
        """Add a design parameter."""
        self.parameters[param.name] = param
    
    def add_relationship(self, relationship: ParameterRelationship) -> bool:
        """Add a relationship between parameters."""
        if relationship.source_param not in self.parameters or relationship.target_param not in self.parameters:
            return False
            
        if relationship.source_param not in self.relationships:
            self.relationships[relationship.source_param] = []
            
        self.relationships[relationship.source_param].append(relationship)
        
        # Update dependencies in the target parameter
        target_param = self.parameters[relationship.target_param]
        if relationship.source_param not in target_param.dependencies:
            target_param.dependencies.append(relationship.source_param)
            
        # Apply relationship immediately
        self._apply_relationship(relationship)
        return True
    
    def add_parameter_group(self, group_name: str, parameter_names: List[str]) -> bool:
        """Group related parameters for validation."""
        valid_params = [p for p in parameter_names if p in self.parameters]
        if not valid_params:
            return False
            
        self.parameter_groups[group_name] = set(valid_params)
        return True
    
    def set_parameter(self, name: str, value: float) -> bool:
        """Set parameter value and update dependencies."""
        if name not in self.parameters:
            return False
            
        param = self.parameters[name]
        old_value = param.value
        param.value = value
        
        valid, _ = param.validate()
        if not valid:
            param.value = old_value  # Revert if invalid
            return False
            
        self._update_dependencies(name)
        return True
    
    def validate_design(self) -> ParameterValidationResult:
        """Validate all parameters and their relationships."""
        result = ParameterValidationResult(valid=True)
        
        # Validate individual parameters
        for name, param in self.parameters.items():
            valid, message = param.validate()
            if not valid:
                result.add_issue(
                    message=message or f"Parameter {name} validation failed",
                    severity=ValidationSeverity.ERROR,
                    param_name=name
                )
        
        # Validate parameter groups
        for group_name, param_names in self.parameter_groups.items():
            self._validate_parameter_group(group_name, param_names, result)
        
        # Validate relationships
        self._validate_relationships(result)
        
        return result
    
    def _validate_parameter_group(self, group_name: str, param_names: Set[str], 
                                result: ParameterValidationResult) -> None:
        """Validate a group of related parameters."""
        # Example: validate aspect ratio constraints for wings
        if group_name == "wing_geometry":
            if "wingspan" in param_names and "wing_chord" in param_names:
                wingspan = self.parameters["wingspan"].value
                wing_chord = self.parameters["wing_chord"].value
                
                if wing_chord > 0:
                    aspect_ratio = wingspan / wing_chord
                    if aspect_ratio < 3.0:
                        result.add_issue(
                            message=f"Wing aspect ratio ({aspect_ratio:.2f}) is too low for efficient flight",
                            severity=ValidationSeverity.WARNING,
                            param_name="wingspan"
                        )
    
    def _validate_relationships(self, result: ParameterValidationResult) -> None:
        """Validate parameter relationships."""
        for source_name, relationships in self.relationships.items():
            source_param = self.parameters[source_name]
            
            for relationship in relationships:
                target_name = relationship.target_param
                target_param = self.parameters[target_name]
                
                # Check if the relationship produces valid target values
                expected_value = relationship.apply(source_param.value)
                if expected_value < target_param.min_value or expected_value > target_param.max_value:
                    result.add_issue(
                        message=f"Relationship from {source_name} produces invalid value for {target_name}",
                        severity=ValidationSeverity.WARNING,
                        param_name=target_name
                    )
    
    def _update_dependencies(self, param_name: str) -> None:
        """Update dependent parameters."""
        # Apply direct relationships
        if param_name in self.relationships:
            for relationship in self.relationships[param_name]:
                self._apply_relationship(relationship)
        
        # Legacy support for old dependency system
        for name, param in self.parameters.items():
            if param.dependencies and param_name in param.dependencies:
                self._calculate_dependent_value(name)
    
    def _apply_relationship(self, relationship: ParameterRelationship) -> None:
        """Apply a specific relationship to update target parameter."""
        source_param = self.parameters[relationship.source_param]
        target_param = self.parameters[relationship.target_param]
        
        # Calculate new value
        new_value = relationship.apply(source_param.value)
        
        # Ensure value is within bounds
        new_value = max(target_param.min_value, min(new_value, target_param.max_value))
        
        # Update target parameter
        target_param.value = new_value
    
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
        self._initialize_relationships()
        self._initialize_parameter_groups()
        self._initialize_validation_rules()
    
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
        
        # Add new parameters
        self.add_parameter(DesignParameter(
            name="wing_area",
            type=ParameterType.LENGTH,
            value=30.0,  # m²
            min_value=15.0,
            max_value=50.0
        ))
        
        self.add_parameter(DesignParameter(
            name="wing_loading",
            type=ParameterType.RATIO,
            value=250.0,  # kg/m²
            min_value=150.0,
            max_value=400.0
        ))
        
        self.add_parameter(DesignParameter(
            name="max_takeoff_weight",
            type=ParameterType.RATIO,
            value=7500.0,  # kg
            min_value=5000.0,
            max_value=12000.0
        ))
    
    def _initialize_relationships(self):
        """Initialize parameter relationships."""
        # Wing area based on wingspan and aspect ratio
        self.add_relationship(ParameterRelationship(
            source_param="wingspan",
            target_param="wing_area",
            relationship_type=RelationshipType.CUSTOM,
            custom_function=lambda wingspan: (wingspan/1000)**2 / self.parameters["aspect_ratio"].value
        ))
        
        # Wing loading relationship
        self.add_relationship(ParameterRelationship(
            source_param="max_takeoff_weight",
            target_param="wing_loading",
            relationship_type=RelationshipType.CUSTOM,
            custom_function=lambda mtow: mtow / self.parameters["wing_area"].value
        ))
        
        # Aspect ratio affects wingspan
        self.add_relationship(ParameterRelationship(
            source_param="aspect_ratio",
            target_param="wingspan",
            relationship_type=RelationshipType.CUSTOM,
            custom_function=lambda ar: 1000 * math.sqrt(ar * self.parameters["wing_area"].value)
        ))
    
    def _initialize_parameter_groups(self):
        """Initialize parameter groups for validation."""
        self.add_parameter_group("wing_geometry", ["wingspan", "aspect_ratio", "wing_area"])
        self.add_parameter_group("weight_balance", ["max_takeoff_weight", "wing_loading", "wing_area"])
        self.add_parameter_group("stealth_features", ["wing_sweep", "fuselage_length"])
    
    def _initialize_validation_rules(self):
        """Add custom validation rules to parameters."""
        # Add rule to check if wing sweep is appropriate for the speed regime
        if "wing_sweep" in self.parameters:
            self.parameters["wing_sweep"].validation_rules.append(
                lambda sweep: (sweep >= 30.0 or "max_speed_mach" not in self.parameters or 
                              self.parameters["max_speed_mach"].value < 0.8,
                              "Wing sweep too low for supersonic flight")
            )
        
        # Add rule to check thrust-to-weight ratio
        if "max_takeoff_weight" in self.parameters and "max_thrust" in self.parameters:
            self.parameters["max_thrust"].validation_rules.append(
                lambda thrust: (thrust / self.parameters["max_takeoff_weight"].value >= 0.3,
                               "Thrust-to-weight ratio too low for combat aircraft")
            )
    
    def generate_design(self) -> Dict[str, Any]:
        """Generate design data from parameters."""
        # Validate design before generating
        validation = self.validate_design()
        if not validation.valid:
            return {
                "error": "Design validation failed",
                "issues": validation.issues
            }
            
        return {
            "dimensions": self._calculate_dimensions(),
            "aerodynamics": self._calculate_aerodynamics(),
            "parameters": {name: param.value for name, param in self.parameters.items()},
            "validation": {"valid": True, "issues": validation.issues}
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
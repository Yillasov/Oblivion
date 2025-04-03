"""
Realistic design constraint handling for UCAV systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

# Fix the import path
from src.core.constraints.constraint_manager import (
    ConstraintManager, ConstraintDefinition, ConstraintViolation, ConstraintSeverity
)
from src.core.utils.logging_framework import get_logger

logger = get_logger("realistic_constraints")

class ConstraintCategory(Enum):
    """Categories of realistic design constraints."""
    MANUFACTURING = "manufacturing"
    AERODYNAMIC = "aerodynamic"
    STRUCTURAL = "structural"
    THERMAL = "thermal"
    STEALTH = "stealth"
    PROPULSION = "propulsion"
    WEIGHT = "weight"
    COST = "cost"

@dataclass
class RealisticConstraint:
    """Definition of a realistic design constraint with manufacturing implications."""
    name: str
    category: ConstraintCategory
    description: str
    parameters: List[str]
    subsystems: List[str]
    severity: ConstraintSeverity = ConstraintSeverity.ERROR
    dependencies: Set[str] = field(default_factory=set)
    
    # Constraint bounds
    min_values: Dict[str, float] = field(default_factory=dict)
    max_values: Dict[str, float] = field(default_factory=dict)
    ratio_constraints: List[Tuple[str, str, float, float]] = field(default_factory=list)
    
    def create_check_function(self):
        """Create a constraint check function for this realistic constraint."""
        def check_function(design_params: Dict[str, Any]) -> List[ConstraintViolation]:
            violations = []
            
            # Check min values
            for param, min_val in self.min_values.items():
                if param in design_params and design_params[param] < min_val:
                    violations.append(ConstraintViolation(
                        constraint_id=self.name,
                        message=f"{param} value {design_params[param]} is below minimum {min_val}",
                        severity=self.severity,
                        affected_parameters=[param]
                    ))
            
            # Check max values
            for param, max_val in self.max_values.items():
                if param in design_params and design_params[param] > max_val:
                    violations.append(ConstraintViolation(
                        constraint_id=self.name,
                        message=f"{param} value {design_params[param]} exceeds maximum {max_val}",
                        severity=self.severity,
                        affected_parameters=[param]
                    ))
            
            # Check ratio constraints
            for param1, param2, min_ratio, max_ratio in self.ratio_constraints:
                if param1 in design_params and param2 in design_params and design_params[param2] != 0:
                    ratio = design_params[param1] / design_params[param2]
                    if ratio < min_ratio or ratio > max_ratio:
                        violations.append(ConstraintViolation(
                            constraint_id=self.name,
                            message=f"Ratio of {param1}/{param2} ({ratio:.2f}) outside allowed range [{min_ratio}, {max_ratio}]",
                            severity=self.severity,
                            affected_parameters=[param1, param2]
                        ))
            
            return violations
        
        return check_function
    
    def to_constraint_definition(self) -> ConstraintDefinition:
        """Convert to a ConstraintDefinition for the constraint manager."""
        return ConstraintDefinition(
            constraint_id=self.name,
            description=self.description,
            severity=self.severity,
            subsystem=self.subsystems[0],  # Primary subsystem
            check_function=self.create_check_function(),
            parameters=self.parameters,
            dependencies=self.dependencies
        )

class RealisticConstraintManager:
    """Manager for realistic design constraints with manufacturing implications."""
    
    def __init__(self, constraint_manager: ConstraintManager):
        self.constraint_manager = constraint_manager
        self.realistic_constraints: Dict[str, RealisticConstraint] = {}
        self._initialize_default_constraints()
    
    def _initialize_default_constraints(self):
        """Initialize default realistic constraints."""
        # Manufacturing constraints
        self.add_constraint(RealisticConstraint(
            name="min_wall_thickness",
            category=ConstraintCategory.MANUFACTURING,
            description="Minimum wall thickness for manufacturability",
            parameters=["wall_thickness_mm"],
            subsystems=["airframe", "stealth"],
            min_values={"wall_thickness_mm": 1.0}
        ))
        
        # Aspect ratio constraint
        self.add_constraint(RealisticConstraint(
            name="wing_aspect_ratio",
            category=ConstraintCategory.AERODYNAMIC,
            description="Wing aspect ratio for aerodynamic performance",
            parameters=["wing_span_m", "wing_chord_m"],
            subsystems=["airframe", "aerodynamics"],
            ratio_constraints=[("wing_span_m", "wing_chord_m", 4.0, 12.0)]
        ))
        
        # Thermal constraints
        self.add_constraint(RealisticConstraint(
            name="thermal_material_limits",
            category=ConstraintCategory.THERMAL,
            description="Material temperature limits for thermal protection",
            parameters=["max_surface_temp_k", "material_temp_limit_k"],
            subsystems=["thermal", "materials"],
            max_values={"max_surface_temp_k": 2000.0}
        ))
        
        # Weight constraints
        self.add_constraint(RealisticConstraint(
            name="thrust_to_weight_ratio",
            category=ConstraintCategory.PROPULSION,
            description="Thrust to weight ratio for performance",
            parameters=["max_thrust_n", "total_weight_kg"],
            subsystems=["propulsion", "airframe"],
            ratio_constraints=[("max_thrust_n", "total_weight_kg", 0.8, 2.0)]
        ))
        
        # Stealth constraints
        self.add_constraint(RealisticConstraint(
            name="radar_absorbing_material_thickness",
            category=ConstraintCategory.STEALTH,
            description="Radar absorbing material thickness for stealth",
            parameters=["ram_thickness_mm"],
            subsystems=["stealth", "materials"],
            min_values={"ram_thickness_mm": 2.0},
            max_values={"ram_thickness_mm": 15.0}
        ))
    
    def add_constraint(self, constraint: RealisticConstraint):
        """Add a realistic constraint and register with constraint manager."""
        self.realistic_constraints[constraint.name] = constraint
        
        # Register with constraint manager
        constraint_def = constraint.to_constraint_definition()
        self.constraint_manager.register_constraint(constraint_def)
        
        logger.info(f"Added realistic constraint: {constraint.name}")
    
    def validate_design(self, design_params: Dict[str, Any], 
                       categories: Optional[List[ConstraintCategory]] = None) -> Dict[str, Any]:
        """
        Validate a design against realistic constraints.
        
        Args:
            design_params: Design parameters
            categories: Optional list of constraint categories to check
            
        Returns:
            Dict with validation results
        """
        # Filter subsystems based on categories
        if categories:
            subsystems = []
            for constraint in self.realistic_constraints.values():
                if constraint.category in categories:
                    subsystems.extend(constraint.subsystems)
            subsystems = list(set(subsystems))
        else:
            subsystems = None
        
        # Use constraint manager to validate
        return self.constraint_manager.validate_design(design_params, subsystems)
    
    def suggest_realistic_fixes(self, design_params: Dict[str, Any], 
                              validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest realistic fixes for constraint violations.
        
        Args:
            design_params: Current design parameters
            validation_result: Validation result from validate_design
            
        Returns:
            Dict with suggested parameter adjustments
        """
        # Extract violations
        violations = []
        for severity_list in validation_result["violations_by_severity"].values():
            violations.extend(severity_list)
        
        # Use constraint manager to suggest fixes
        fixes = self.constraint_manager.suggest_fixes(design_params, violations)
        
        # Add manufacturing feasibility assessment
        fixes["manufacturing_feasibility"] = self._assess_manufacturing_feasibility(
            fixes["suggested_parameters"]
        )
        
        return fixes
    
    def _assess_manufacturing_feasibility(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assess manufacturing feasibility of suggested parameters."""
        # Simple assessment based on known manufacturing capabilities
        feasibility = {
            "overall_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Check wall thickness if present
        if "wall_thickness_mm" in params:
            thickness = params["wall_thickness_mm"]
            if thickness < 0.5:
                feasibility["issues"].append("Wall thickness below reliable manufacturing limits")
                feasibility["recommendations"].append("Increase wall thickness to at least 0.5mm")
            elif thickness < 1.0:
                feasibility["issues"].append("Wall thickness may require specialized manufacturing")
                feasibility["recommendations"].append("Consider using advanced manufacturing techniques")
        
        # Check material compatibility if materials are specified
        if "primary_material" in params and "coating_material" in params:
            # This would be expanded with actual material compatibility logic
            feasibility["recommendations"].append("Verify material compatibility with manufacturer")
        
        # Calculate overall feasibility score (simplified)
        issue_count = len(feasibility["issues"])
        if issue_count == 0:
            feasibility["overall_score"] = 1.0
        else:
            feasibility["overall_score"] = max(0.0, 1.0 - (issue_count * 0.2))
        
        return feasibility
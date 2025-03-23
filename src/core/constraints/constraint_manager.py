"""
Constraint management system for UCAV design validation.
"""

from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from src.core.utils.logging_framework import get_logger

logger = get_logger("constraint_manager")

class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ConstraintViolation:
    """Represents a violation of a design constraint."""
    constraint_id: str
    message: str
    severity: ConstraintSeverity
    affected_parameters: List[str]
    suggested_fix: Optional[Dict[str, Any]] = None

@dataclass
class ConstraintDefinition:
    """Definition of a design constraint."""
    constraint_id: str
    description: str
    severity: ConstraintSeverity
    subsystem: str
    check_function: Callable[[Dict[str, Any]], List[ConstraintViolation]]
    parameters: List[str]
    dependencies: Set[str] = field(default_factory=set)
    
    def check(self, design_params: Dict[str, Any]) -> List[ConstraintViolation]:
        """Check if the constraint is satisfied."""
        return self.check_function(design_params)

class ConstraintManager:
    """Manages design constraints across subsystems."""
    
    def __init__(self):
        self.constraints: Dict[str, ConstraintDefinition] = {}
        self.subsystem_constraints: Dict[str, List[str]] = {}
        
    def register_constraint(self, constraint: ConstraintDefinition) -> None:
        """Register a new constraint."""
        self.constraints[constraint.constraint_id] = constraint
        
        # Add to subsystem mapping
        if constraint.subsystem not in self.subsystem_constraints:
            self.subsystem_constraints[constraint.subsystem] = []
        self.subsystem_constraints[constraint.subsystem].append(constraint.constraint_id)
        
        logger.info(f"Registered constraint: {constraint.constraint_id} for subsystem {constraint.subsystem}")
        
    def validate_design(self, design_params: Dict[str, Any], 
                       subsystems: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate a design against registered constraints.
        
        Args:
            design_params: Design parameters
            subsystems: Optional list of subsystems to check
            
        Returns:
            Dict with validation results
        """
        violations = []
        checked_constraints = []
        
        # Determine which constraints to check
        if subsystems:
            constraint_ids = []
            for subsystem in subsystems:
                constraint_ids.extend(self.subsystem_constraints.get(subsystem, []))
        else:
            constraint_ids = list(self.constraints.keys())
        
        # Check each constraint
        for constraint_id in constraint_ids:
            constraint = self.constraints.get(constraint_id)
            if not constraint:
                continue
                
            checked_constraints.append(constraint_id)
            constraint_violations = constraint.check(design_params)
            violations.extend(constraint_violations)
        
        # Group violations by severity
        violations_by_severity = {}
        for severity in ConstraintSeverity:
            violations_by_severity[severity] = [
                v for v in violations if v.severity == severity
            ]
        
        # Determine overall validity
        has_errors = any(
            len(violations_by_severity[severity]) > 0 
            for severity in [ConstraintSeverity.ERROR, ConstraintSeverity.CRITICAL]
        )
        
        return {
            "valid": not has_errors,
            "total_violations": len(violations),
            "violations_by_severity": violations_by_severity,
            "checked_constraints": checked_constraints
        }
    
    def suggest_fixes(self, design_params: Dict[str, Any], 
                     violations: List[ConstraintViolation]) -> Dict[str, Any]:
        """
        Suggest parameter adjustments to fix constraint violations.
        
        Args:
            design_params: Current design parameters
            violations: List of constraint violations
            
        Returns:
            Dict with suggested parameter adjustments
        """
        suggested_params = design_params.copy()
        adjustments = {}
        
        # Process each violation
        for violation in violations:
            constraint_id = violation.constraint_id
            constraint = self.constraints.get(constraint_id)
            
            if not constraint:
                continue
                
            # Simple adjustment logic based on constraint type
            for param in violation.affected_parameters:
                if param not in design_params:
                    continue
                    
                current_value = design_params[param]
                
                # Check if this is a min/max constraint by looking at the message
                if "below minimum" in violation.message:
                    # Extract minimum value from message
                    parts = violation.message.split("minimum ")
                    if len(parts) > 1:
                        try:
                            min_value = float(parts[1])
                            suggested_params[param] = min_value
                            adjustments[param] = {
                                "original": current_value,
                                "suggested": min_value,
                                "reason": f"Below minimum value of {min_value}"
                            }
                        except ValueError:
                            pass
                            
                elif "exceeds maximum" in violation.message:
                    # Extract maximum value from message
                    parts = violation.message.split("maximum ")
                    if len(parts) > 1:
                        try:
                            max_value = float(parts[1])
                            suggested_params[param] = max_value
                            adjustments[param] = {
                                "original": current_value,
                                "suggested": max_value,
                                "reason": f"Exceeds maximum value of {max_value}"
                            }
                        except ValueError:
                            pass
                            
                elif "outside allowed range" in violation.message:
                    # This is a ratio constraint, more complex to fix
                    # For now, we'll just log it
                    logger.info(f"Ratio constraint violation: {violation.message}")
        
        return {
            "suggested_parameters": suggested_params,
            "adjustments": adjustments
        }
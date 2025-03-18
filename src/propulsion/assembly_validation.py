"""
Assembly validation system for propulsion manufacturing.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
import time

from src.propulsion.manufacturing_workflow_simple import PropulsionManufacturingWorkflow, ManufacturingStage


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = 0
    WARNING = 1
    INFO = 2


@dataclass
class ValidationRule:
    """Rule for validating assembly steps."""
    name: str
    description: str
    severity: ValidationSeverity
    check_function: Callable[[str, Dict[str, Any]], Dict[str, Any]]


class AssemblyValidator:
    """Validates propulsion system assembly."""
    
    def __init__(self, manufacturing_workflow: Optional[PropulsionManufacturingWorkflow] = None):
        """Initialize assembly validator."""
        self.manufacturing_workflow = manufacturing_workflow
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.validation_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize with basic validation rules
        self._initialize_basic_rules()
        
    def _initialize_basic_rules(self) -> None:
        """Initialize basic validation rules."""
        # Common rules for all propulsion systems
        common_rules = [
            ValidationRule(
                name="component_count",
                description="Verify all required components are present",
                severity=ValidationSeverity.ERROR,
                check_function=self._check_component_count
            ),
            ValidationRule(
                name="fastener_torque",
                description="Verify fasteners are properly torqued",
                severity=ValidationSeverity.ERROR,
                check_function=self._check_fastener_torque
            ),
            ValidationRule(
                name="alignment",
                description="Verify component alignment",
                severity=ValidationSeverity.ERROR,
                check_function=self._check_alignment
            ),
            ValidationRule(
                name="clearance",
                description="Verify proper clearance between components",
                severity=ValidationSeverity.WARNING,
                check_function=self._check_clearance
            )
        ]
        
        self.validation_rules["common"] = common_rules
        
    def add_rule(self, system_type: str, rule: ValidationRule) -> None:
        """Add a validation rule for a specific system type."""
        if system_type not in self.validation_rules:
            self.validation_rules[system_type] = []
            
        self.validation_rules[system_type].append(rule)
        
    def validate_assembly(self, system_id: str, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate assembly for a propulsion system.
        
        Args:
            system_id: ID of the propulsion system
            assembly_data: Assembly data to validate
            
        Returns:
            Dict with validation results
        """
        if not self.manufacturing_workflow:
            return {"success": False, "error": "No manufacturing workflow connected"}
            
        if system_id not in self.manufacturing_workflow.systems:
            return {"success": False, "error": "System not found in manufacturing workflow"}
            
        # Check if system is in assembly stage
        if system_id not in self.manufacturing_workflow.stages:
            return {"success": False, "error": "System has no manufacturing stage"}
            
        current_stage = self.manufacturing_workflow.stages[system_id]
        if current_stage != ManufacturingStage.ASSEMBLY:
            return {
                "success": False, 
                "error": f"System is not in assembly stage (current: {current_stage.name})"
            }
            
        # Get system type
        system = self.manufacturing_workflow.systems[system_id]
        system_type = system.get_specifications().propulsion_type.name.lower()
        
        # Collect applicable rules
        rules = self.validation_rules.get("common", []).copy()
        if system_type in self.validation_rules:
            rules.extend(self.validation_rules[system_type])
            
        # Run validation
        issues = []
        for rule in rules:
            rule_result = rule.check_function(system_id, assembly_data)
            if not rule_result["passed"]:
                issues.append({
                    "rule": rule.name,
                    "description": rule.description,
                    "severity": rule.severity.name,
                    "details": rule_result.get("details", "")
                })
                
        # Store validation results
        result = {
            "success": True,
            "system_id": system_id,
            "timestamp": time.time(),
            "passed": len(issues) == 0,
            "issues": issues
        }
        
        if system_id not in self.validation_results:
            self.validation_results[system_id] = []
            
        self.validation_results[system_id].append(result)
        
        # Update manufacturing quality if there are issues
        if issues and self.manufacturing_workflow:
            error_count = sum(1 for issue in issues if issue["severity"] == "ERROR")
            warning_count = sum(1 for issue in issues if issue["severity"] == "WARNING")
            
            # Calculate quality impact
            quality_impact = error_count * 0.2 + warning_count * 0.05
            
            # Update quality in manufacturing workflow
            if system_id in self.manufacturing_workflow.metrics:
                current_quality = self.manufacturing_workflow.metrics[system_id].get("quality", 0.0)
                new_quality = max(0.0, current_quality - quality_impact)
                self.manufacturing_workflow.metrics[system_id]["quality"] = new_quality
                
                result["quality_impact"] = {
                    "before": current_quality,
                    "after": new_quality
                }
        
        return result
        
    # Validation check functions
    def _check_component_count(self, system_id: str, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all required components are present."""
        if "components" not in assembly_data:
            return {"passed": False, "details": "No components data found"}
            
        components = assembly_data["components"]
        required_count = assembly_data.get("required_component_count", 0)
        
        if len(components) < required_count:
            return {
                "passed": False, 
                "details": f"Missing components: found {len(components)}, required {required_count}"
            }
            
        return {"passed": True}
        
    def _check_fastener_torque(self, system_id: str, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if fasteners are properly torqued."""
        if "fasteners" not in assembly_data:
            return {"passed": True}  # Skip if no fastener data
            
        fasteners = assembly_data["fasteners"]
        issues = []
        
        for fastener in fasteners:
            if "torque" not in fastener or "required_torque" not in fastener:
                continue
                
            actual = fastener["torque"]
            required = fastener["required_torque"]
            tolerance = fastener.get("tolerance", 0.1)  # 10% tolerance by default
            
            if abs(actual - required) / required > tolerance:
                issues.append(
                    f"Fastener {fastener.get('id', 'unknown')}: "
                    f"torque {actual} outside tolerance of {required}±{tolerance*100}%"
                )
                
        if issues:
            return {"passed": False, "details": "; ".join(issues)}
            
        return {"passed": True}
        
    def _check_alignment(self, system_id: str, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check component alignment."""
        if "alignment" not in assembly_data:
            return {"passed": True}  # Skip if no alignment data
            
        alignment = assembly_data["alignment"]
        issues = []
        
        for component, data in alignment.items():
            if "actual" not in data or "required" not in data:
                continue
                
            actual = data["actual"]
            required = data["required"]
            tolerance = data.get("tolerance", 0.5)  # 0.5 degree tolerance by default
            
            if abs(actual - required) > tolerance:
                issues.append(
                    f"Component {component}: "
                    f"alignment {actual}° outside tolerance of {required}±{tolerance}°"
                )
                
        if issues:
            return {"passed": False, "details": "; ".join(issues)}
            
        return {"passed": True}
        
    def _check_clearance(self, system_id: str, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check clearance between components."""
        if "clearances" not in assembly_data:
            return {"passed": True}  # Skip if no clearance data
            
        clearances = assembly_data["clearances"]
        issues = []
        
        for item in clearances:
            if "actual" not in item or "minimum" not in item:
                continue
                
            actual = item["actual"]
            minimum = item["minimum"]
            
            if actual < minimum:
                issues.append(
                    f"Clearance between {item.get('component1', 'unknown')} and "
                    f"{item.get('component2', 'unknown')}: "
                    f"{actual}mm is less than minimum {minimum}mm"
                )
                
        if issues:
            return {"passed": False, "details": "; ".join(issues)}
            
        return {"passed": True}
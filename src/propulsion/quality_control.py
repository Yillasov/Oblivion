"""
Quality control framework for propulsion manufacturing.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
import time
import statistics
import uuid

from src.propulsion.manufacturing_workflow_simple import PropulsionManufacturingWorkflow, ManufacturingStage
from src.propulsion.assembly_validation import AssemblyValidator


class QualityCheckType(Enum):
    """Types of quality checks."""
    VISUAL = 0
    DIMENSIONAL = 1
    FUNCTIONAL = 2
    PERFORMANCE = 3
    MATERIAL = 4
    DOCUMENTATION = 5


class QualityStatus(Enum):
    """Status of quality checks."""
    PENDING = 0
    PASSED = 1
    FAILED = 2
    WAIVED = 3
    IN_PROGRESS = 4


@dataclass
class QualityCheck:
    """Quality check definition."""
    id: str
    name: str
    description: str
    check_type: QualityCheckType
    stage: ManufacturingStage
    required: bool = True
    dependencies: Set[str] = field(default_factory=set)
    check_function: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None


@dataclass
class QualityCheckResult:
    """Result of a quality check."""
    check_id: str
    system_id: str
    status: QualityStatus
    timestamp: float
    inspector_id: Optional[str] = None
    measurements: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    images: List[str] = field(default_factory=list)


class QualityControlFramework:
    """Quality control framework for propulsion manufacturing."""
    
    def __init__(
        self, 
        manufacturing_workflow: Optional[PropulsionManufacturingWorkflow] = None,
        assembly_validator: Optional[AssemblyValidator] = None
    ):
        """Initialize quality control framework."""
        self.manufacturing_workflow = manufacturing_workflow
        self.assembly_validator = assembly_validator
        self.quality_checks: Dict[str, QualityCheck] = {}
        self.check_results: Dict[str, List[QualityCheckResult]] = {}
        self.quality_metrics: Dict[str, Dict[str, float]] = {}
        
        # Initialize with standard quality checks
        self._initialize_standard_checks()
        
    def _initialize_standard_checks(self) -> None:
        """Initialize standard quality checks."""
        # Visual inspection checks
        self._add_check(QualityCheck(
            id="visual_surface",
            name="Surface Inspection",
            description="Check for surface defects, scratches, or discoloration",
            check_type=QualityCheckType.VISUAL,
            stage=ManufacturingStage.QUALITY_CHECK
        ))
        
        self._add_check(QualityCheck(
            id="visual_welds",
            name="Weld Inspection",
            description="Inspect all welds for quality and completeness",
            check_type=QualityCheckType.VISUAL,
            stage=ManufacturingStage.QUALITY_CHECK
        ))
        
        # Dimensional checks
        self._add_check(QualityCheck(
            id="dim_tolerances",
            name="Dimensional Tolerances",
            description="Verify all critical dimensions are within tolerance",
            check_type=QualityCheckType.DIMENSIONAL,
            stage=ManufacturingStage.QUALITY_CHECK
        ))
        
        # Functional checks
        self._add_check(QualityCheck(
            id="func_valves",
            name="Valve Operation",
            description="Verify all valves operate correctly",
            check_type=QualityCheckType.FUNCTIONAL,
            stage=ManufacturingStage.TESTING
        ))
        
        self._add_check(QualityCheck(
            id="func_seals",
            name="Seal Integrity",
            description="Verify all seals maintain pressure",
            check_type=QualityCheckType.FUNCTIONAL,
            stage=ManufacturingStage.TESTING,
            dependencies={"func_valves"}
        ))
        
        # Performance checks
        self._add_check(QualityCheck(
            id="perf_thrust",
            name="Thrust Performance",
            description="Verify thrust output meets specifications",
            check_type=QualityCheckType.PERFORMANCE,
            stage=ManufacturingStage.TESTING,
            dependencies={"func_valves", "func_seals"}
        ))
        
        self._add_check(QualityCheck(
            id="perf_efficiency",
            name="Efficiency Metrics",
            description="Verify efficiency metrics meet specifications",
            check_type=QualityCheckType.PERFORMANCE,
            stage=ManufacturingStage.TESTING,
            dependencies={"perf_thrust"}
        ))
        
        # Documentation checks
        self._add_check(QualityCheck(
            id="doc_complete",
            name="Documentation Completeness",
            description="Verify all required documentation is complete",
            check_type=QualityCheckType.DOCUMENTATION,
            stage=ManufacturingStage.QUALITY_CHECK
        ))
    
    def _add_check(self, check: QualityCheck) -> None:
        """Add a quality check to the framework."""
        self.quality_checks[check.id] = check
    
    def add_custom_check(self, check: QualityCheck) -> None:
        """Add a custom quality check to the framework."""
        if check.id in self.quality_checks:
            raise ValueError(f"Check with ID {check.id} already exists")
        self._add_check(check)
    
    def perform_check(
        self, 
        check_id: str, 
        system_id: str, 
        inspector_id: str,
        measurements: Dict[str, Any],
        status: QualityStatus,
        notes: str = "",
        images: List[str] = []
    ) -> QualityCheckResult:
        """
        Perform a quality check on a propulsion system.
        
        Args:
            check_id: ID of the quality check
            system_id: ID of the propulsion system
            inspector_id: ID of the inspector performing the check
            measurements: Measurements or data collected during the check
            status: Status of the check (PASSED, FAILED, etc.)
            notes: Additional notes from the inspector
            images: List of image paths documenting the check
            
        Returns:
            Result of the quality check
        """
        if check_id not in self.quality_checks:
            raise ValueError(f"Unknown check ID: {check_id}")
            
        if not self.manufacturing_workflow:
            raise ValueError("No manufacturing workflow connected")
            
        if system_id not in self.manufacturing_workflow.systems:
            raise ValueError(f"Unknown system ID: {system_id}")
            
        # Get the check definition
        check = self.quality_checks[check_id]
        
        # Verify system is in the correct stage for this check
        if system_id in self.manufacturing_workflow.stages:
            current_stage = self.manufacturing_workflow.stages[system_id]
            if current_stage != check.stage:
                raise ValueError(
                    f"System is in stage {current_stage.name}, but check requires {check.stage.name}"
                )
        
        # Verify dependencies have been completed
        if check.dependencies:
            for dep_id in check.dependencies:
                if not self._is_check_passed(dep_id, system_id):
                    raise ValueError(
                        f"Dependency check {dep_id} has not been passed for system {system_id}"
                    )
        
        # Create the check result
        result = QualityCheckResult(
            check_id=check_id,
            system_id=system_id,
            status=status,
            timestamp=time.time(),
            inspector_id=inspector_id,
            measurements=measurements,
            notes=notes,
            images=images
        )
        
        # Store the result
        if system_id not in self.check_results:
            self.check_results[system_id] = []
        self.check_results[system_id].append(result)
        
        # Update quality metrics
        self._update_quality_metrics(system_id)
        
        # Update manufacturing workflow quality if available
        if self.manufacturing_workflow and system_id in self.manufacturing_workflow.metrics:
            quality_score = self.get_quality_score(system_id)
            self.manufacturing_workflow.metrics[system_id]["quality"] = quality_score
        
        return result
    
    def _is_check_passed(self, check_id: str, system_id: str) -> bool:
        """Check if a specific quality check has been passed for a system."""
        if system_id not in self.check_results:
            return False
            
        # Find the most recent result for this check
        for result in reversed(self.check_results[system_id]):
            if result.check_id == check_id:
                return result.status == QualityStatus.PASSED
                
        return False
    
    def _update_quality_metrics(self, system_id: str) -> None:
        """Update quality metrics for a system based on check results."""
        if system_id not in self.check_results:
            return
            
        results = self.check_results[system_id]
        
        # Initialize metrics if needed
        if system_id not in self.quality_metrics:
            self.quality_metrics[system_id] = {
                "pass_rate": 0.0,
                "defect_rate": 0.0,
                "completion_rate": 0.0,
                "quality_score": 0.0
            }
        
        # Calculate metrics
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.status == QualityStatus.PASSED)
        failed_checks = sum(1 for r in results if r.status == QualityStatus.FAILED)
        
        # Calculate pass and defect rates
        pass_rate = passed_checks / total_checks if total_checks > 0 else 0.0
        defect_rate = failed_checks / total_checks if total_checks > 0 else 0.0
        
        # Calculate completion rate
        required_checks = [c for c in self.quality_checks.values() if c.required]
        completed_required = set()
        
        for result in results:
            if result.status in [QualityStatus.PASSED, QualityStatus.WAIVED]:
                completed_required.add(result.check_id)
                
        completion_rate = len(completed_required) / len(required_checks) if required_checks else 1.0
        
        # Calculate overall quality score (weighted average)
        quality_score = (pass_rate * 0.4) + (completion_rate * 0.4) + ((1 - defect_rate) * 0.2)
        
        # Update metrics
        self.quality_metrics[system_id] = {
            "pass_rate": pass_rate,
            "defect_rate": defect_rate,
            "completion_rate": completion_rate,
            "quality_score": quality_score
        }
    
    def get_quality_score(self, system_id: str) -> float:
        """Get the overall quality score for a system."""
        if system_id not in self.quality_metrics:
            return 0.0
            
        return self.quality_metrics[system_id]["quality_score"]
    
    def get_check_results(self, system_id: str, check_id: Optional[str] = None) -> List[QualityCheckResult]:
        """Get quality check results for a system."""
        if system_id not in self.check_results:
            return []
            
        results = self.check_results[system_id]
        
        if check_id:
            return [r for r in results if r.check_id == check_id]
        
        return results
    
    def get_pending_checks(self, system_id: str) -> List[QualityCheck]:
        """Get list of pending quality checks for a system."""
        if not self.manufacturing_workflow or system_id not in self.manufacturing_workflow.stages:
            return []
            
        current_stage = self.manufacturing_workflow.stages[system_id]
        completed_checks = set()
        
        if system_id in self.check_results:
            for result in self.check_results[system_id]:
                if result.status in [QualityStatus.PASSED, QualityStatus.WAIVED]:
                    completed_checks.add(result.check_id)
        
        # Get checks for the current stage that haven't been completed
        pending = []
        for check_id, check in self.quality_checks.items():
            if check.stage == current_stage and check_id not in completed_checks:
                # Check if dependencies are satisfied
                deps_satisfied = True
                for dep_id in check.dependencies:
                    if not self._is_check_passed(dep_id, system_id):
                        deps_satisfied = False
                        break
                
                if deps_satisfied:
                    pending.append(check)
        
        return pending
    
    def generate_quality_report(self, system_id: str) -> Dict[str, Any]:
        """Generate a comprehensive quality report for a system."""
        if system_id not in self.check_results:
            return {"system_id": system_id, "error": "No quality data available"}
            
        results = self.check_results[system_id]
        metrics = self.quality_metrics.get(system_id, {})
        
        # Group results by check type
        results_by_type = {}
        for check_type in QualityCheckType:
            type_results = []
            for result in results:
                check = self.quality_checks.get(result.check_id)
                if check and check.check_type == check_type:
                    type_results.append({
                        "check_id": result.check_id,
                        "name": check.name,
                        "status": result.status.name,
                        "timestamp": result.timestamp,
                        "inspector": result.inspector_id,
                        "notes": result.notes
                    })
            
            if type_results:
                results_by_type[check_type.name] = type_results
        
        # Generate summary statistics
        status_counts = {}
        for status in QualityStatus:
            count = sum(1 for r in results if r.status == status)
            if count > 0:
                status_counts[status.name] = count
        
        # Get pending checks
        pending = self.get_pending_checks(system_id)
        pending_list = [
            {"id": check.id, "name": check.name, "type": check.check_type.name}
            for check in pending
        ]
        
        # Compile the report
        report = {
            "system_id": system_id,
            "timestamp": time.time(),
            "metrics": metrics,
            "status_summary": status_counts,
            "results_by_type": results_by_type,
            "pending_checks": pending_list,
            "overall_status": "PASSED" if metrics.get("quality_score", 0) >= 0.8 else "FAILED"
        }
        
        return report
    
    def integrate_assembly_validation(self, system_id: str, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate assembly validation results into quality framework.
        
        Args:
            system_id: ID of the propulsion system
            assembly_data: Assembly data to validate
            
        Returns:
            Dict with validation and quality results
        """
        if not self.assembly_validator:
            return {"success": False, "error": "No assembly validator connected"}
            
        # Perform assembly validation
        validation_result = self.assembly_validator.validate_assembly(system_id, assembly_data)
        
        if not validation_result["success"]:
            return validation_result
            
        # Create quality check results based on validation issues
        if not validation_result["passed"]:
            # Add a failed assembly check
            self.perform_check(
                check_id="visual_surface",  # Use an existing check as a placeholder
                system_id=system_id,
                inspector_id="system",  # Automated check
                measurements={"validation_result": validation_result},
                status=QualityStatus.FAILED,
                notes=f"Assembly validation failed with {len(validation_result['issues'])} issues"
            )
        else:
            # Add a passed assembly check
            self.perform_check(
                check_id="visual_surface",  # Use an existing check as a placeholder
                system_id=system_id,
                inspector_id="system",  # Automated check
                measurements={"validation_result": validation_result},
                status=QualityStatus.PASSED,
                notes="Assembly validation passed"
            )
            
        # Return combined results
        return {
            "success": True,
            "validation": validation_result,
            "quality_score": self.get_quality_score(system_id)
        }
        
    def run_automated_checks(self, system_id: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run automated quality checks for a system.
        
        Args:
            system_id: ID of the propulsion system
            test_data: Test data for automated checks
            
        Returns:
            Dict with automated check results
        """
        if not self.manufacturing_workflow:
            return {"success": False, "error": "No manufacturing workflow connected"}
            
        if system_id not in self.manufacturing_workflow.systems:
            return {"success": False, "error": "System not found in manufacturing workflow"}
            
        # Get current stage
        if system_id not in self.manufacturing_workflow.stages:
            return {"success": False, "error": "System has no manufacturing stage"}
            
        current_stage = self.manufacturing_workflow.stages[system_id]
        
        # Find automated checks for current stage
        automated_results = []
        
        # Example automated checks based on test data
        if current_stage == ManufacturingStage.TESTING:
            # Check thrust performance if test data includes it
            if "thrust_test" in test_data:
                thrust_data = test_data["thrust_test"]
                expected_thrust = thrust_data.get("expected", 0)
                actual_thrust = thrust_data.get("actual", 0)
                tolerance = thrust_data.get("tolerance", 0.05)  # 5% tolerance
                
                if abs(actual_thrust - expected_thrust) / expected_thrust <= tolerance:
                    # Thrust is within tolerance
                    result = self.perform_check(
                        check_id="perf_thrust",
                        system_id=system_id,
                        inspector_id="automated",
                        measurements=thrust_data,
                        status=QualityStatus.PASSED,
                        notes=f"Thrust test passed: {actual_thrust} within {tolerance*100}% of {expected_thrust}"
                    )
                else:
                    # Thrust is out of tolerance
                    result = self.perform_check(
                        check_id="perf_thrust",
                        system_id=system_id,
                        inspector_id="automated",
                        measurements=thrust_data,
                        status=QualityStatus.FAILED,
                        notes=f"Thrust test failed: {actual_thrust} outside {tolerance*100}% of {expected_thrust}"
                    )
                    
                automated_results.append(result)
                
            # Check efficiency if test data includes it
            if "efficiency_test" in test_data:
                efficiency_data = test_data["efficiency_test"]
                min_efficiency = efficiency_data.get("minimum", 0)
                actual_efficiency = efficiency_data.get("actual", 0)
                
                if actual_efficiency >= min_efficiency:
                    # Efficiency meets minimum
                    result = self.perform_check(
                        check_id="perf_efficiency",
                        system_id=system_id,
                        inspector_id="automated",
                        measurements=efficiency_data,
                        status=QualityStatus.PASSED,
                        notes=f"Efficiency test passed: {actual_efficiency} >= {min_efficiency}"
                    )
                else:
                    # Efficiency below minimum
                    result = self.perform_check(
                        check_id="perf_efficiency",
                        system_id=system_id,
                        inspector_id="automated",
                        measurements=efficiency_data,
                        status=QualityStatus.FAILED,
                        notes=f"Efficiency test failed: {actual_efficiency} < {min_efficiency}"
                    )
                    
                automated_results.append(result)
        
        return {
            "success": True,
            "system_id": system_id,
            "automated_checks": len(automated_results),
            "quality_score": self.get_quality_score(system_id)
        }
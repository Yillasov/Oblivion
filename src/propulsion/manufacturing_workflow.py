"""
Manufacturing workflow system for propulsion components.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time

from src.propulsion.base import PropulsionInterface, PropulsionSpecs, PropulsionType


class ManufacturingStage(Enum):
    """Manufacturing stages for propulsion systems."""
    DESIGN = 0
    COMPONENT_FABRICATION = 1
    ASSEMBLY = 2
    INTEGRATION = 3
    TESTING = 4
    QUALITY_ASSURANCE = 5
    PACKAGING = 6


@dataclass
class ManufacturingSpec:
    """Manufacturing specifications for propulsion components."""
    propulsion_type: PropulsionType
    complexity: int  # 1-10 scale
    material_requirements: Dict[str, float]  # Material name -> quantity (kg)
    fabrication_time: float  # Hours
    assembly_time: float  # Hours
    testing_duration: float  # Hours
    quality_thresholds: Dict[str, float]  # Parameter -> minimum value


class ManufacturingWorkflow:
    """Manages manufacturing workflows for propulsion systems."""
    
    def __init__(self):
        """Initialize manufacturing workflow system."""
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.manufacturing_specs: Dict[str, ManufacturingSpec] = {}
        self.current_stage: Dict[str, ManufacturingStage] = {}
        self.stage_completion: Dict[str, Dict[ManufacturingStage, float]] = {}
        self.quality_metrics: Dict[str, Dict[str, float]] = {}
        
    def create_workflow(self, 
                      system_id: str, 
                      propulsion_system: PropulsionInterface,
                      manufacturing_spec: ManufacturingSpec) -> bool:
        """Create a manufacturing workflow for a propulsion system."""
        if system_id in self.workflows:
            return False
            
        specs = propulsion_system.get_specifications()
        
        # Initialize workflow
        self.workflows[system_id] = {
            "created_at": time.time(),
            "propulsion_type": specs.propulsion_type.name,
            "thrust_rating": specs.thrust_rating,
            "stage_history": [],
            "issues": [],
            "completed": False
        }
        
        self.manufacturing_specs[system_id] = manufacturing_spec
        self.current_stage[system_id] = ManufacturingStage.DESIGN
        
        # Initialize stage completion
        self.stage_completion[system_id] = {
            stage: 0.0 for stage in ManufacturingStage
        }
        
        # Initialize quality metrics
        self.quality_metrics[system_id] = {
            "structural_integrity": 0.0,
            "thermal_resistance": 0.0,
            "power_efficiency": 0.0,
            "weight_conformance": 0.0,
            "vibration_damping": 0.0
        }
        
        return True
        
    def advance_stage(self, system_id: str) -> bool:
        """Advance manufacturing to the next stage."""
        if system_id not in self.workflows:
            return False
            
        if self.current_stage[system_id] == ManufacturingStage.PACKAGING:
            self.workflows[system_id]["completed"] = True
            return True
            
        # Record stage completion
        current = self.current_stage[system_id]
        self.workflows[system_id]["stage_history"].append({
            "stage": current.name,
            "completed_at": time.time(),
            "quality_metrics": self.quality_metrics[system_id].copy()
        })
        
        # Advance to next stage
        next_stage_value = current.value + 1
        self.current_stage[system_id] = ManufacturingStage(next_stage_value)
        
        return True
        
    def update_progress(self, 
                      system_id: str, 
                      progress: float,
                      quality_updates: Optional[Dict[str, float]] = None) -> bool:
        """
        Update manufacturing progress for the current stage.
        
        Args:
            system_id: System identifier
            progress: Progress increment (0-1)
            quality_updates: Updates to quality metrics
            
        Returns:
            Success status
        """
        if system_id not in self.workflows:
            return False
            
        current = self.current_stage[system_id]
        
        # Update progress
        self.stage_completion[system_id][current] += progress
        self.stage_completion[system_id][current] = min(1.0, self.stage_completion[system_id][current])
        
        # Update quality metrics if provided
        if quality_updates:
            for metric, value in quality_updates.items():
                if metric in self.quality_metrics[system_id]:
                    self.quality_metrics[system_id][metric] = value
        
        return True
        
    def report_issue(self, system_id: str, issue: str, severity: int) -> bool:
        """Report a manufacturing issue."""
        if system_id not in self.workflows:
            return False
            
        self.workflows[system_id]["issues"].append({
            "stage": self.current_stage[system_id].name,
            "description": issue,
            "severity": severity,
            "reported_at": time.time(),
            "resolved": False
        })
        
        return True
        
    def resolve_issue(self, system_id: str, issue_index: int) -> bool:
        """Resolve a reported manufacturing issue."""
        if system_id not in self.workflows:
            return False
            
        if issue_index >= len(self.workflows[system_id]["issues"]):
            return False
            
        self.workflows[system_id]["issues"][issue_index]["resolved"] = True
        self.workflows[system_id]["issues"][issue_index]["resolved_at"] = time.time()
        
        return True
        
    def get_workflow_status(self, system_id: str) -> Dict[str, Any]:
        """Get current status of a manufacturing workflow."""
        if system_id not in self.workflows:
            return {"error": "Workflow not found"}
            
        # Calculate overall progress
        total_stages = len(ManufacturingStage)
        completed_stages = sum(1 for stage, completion 
                             in self.stage_completion[system_id].items() 
                             if completion >= 1.0)
        current_stage_progress = self.stage_completion[system_id][self.current_stage[system_id]]
        
        overall_progress = (completed_stages + current_stage_progress) / total_stages
        
        # Count open issues
        open_issues = sum(1 for issue in self.workflows[system_id]["issues"] if not issue["resolved"])
        
        return {
            "system_id": system_id,
            "propulsion_type": self.workflows[system_id]["propulsion_type"],
            "current_stage": self.current_stage[system_id].name,
            "stage_progress": current_stage_progress,
            "overall_progress": overall_progress,
            "quality_metrics": self.quality_metrics[system_id],
            "open_issues": open_issues,
            "total_issues": len(self.workflows[system_id]["issues"]),
            "completed": self.workflows[system_id]["completed"]
        }
    
    def estimate_completion_time(self, system_id: str) -> Dict[str, Any]:
        """Estimate remaining manufacturing time."""
        if system_id not in self.workflows:
            return {"error": "Workflow not found"}
            
        spec = self.manufacturing_specs[system_id]
        current = self.current_stage[system_id]
        current_progress = self.stage_completion[system_id][current]
        
        # Calculate remaining time based on manufacturing specs
        remaining_hours = 0.0
        
        # Current stage remaining time
        if current == ManufacturingStage.DESIGN:
            stage_total = spec.complexity * 8  # 8 hours per complexity point
            remaining_hours += stage_total * (1 - current_progress)
        elif current == ManufacturingStage.COMPONENT_FABRICATION:
            remaining_hours += spec.fabrication_time * (1 - current_progress)
        elif current == ManufacturingStage.ASSEMBLY:
            remaining_hours += spec.assembly_time * (1 - current_progress)
        elif current == ManufacturingStage.TESTING:
            remaining_hours += spec.testing_duration * (1 - current_progress)
        else:
            # Other stages - estimate based on complexity
            stage_total = spec.complexity * 2  # 2 hours per complexity point
            remaining_hours += stage_total * (1 - current_progress)
            
        # Add time for future stages
        future_stages = [stage for stage in ManufacturingStage if stage.value > current.value]
        for stage in future_stages:
            if stage == ManufacturingStage.COMPONENT_FABRICATION:
                remaining_hours += spec.fabrication_time
            elif stage == ManufacturingStage.ASSEMBLY:
                remaining_hours += spec.assembly_time
            elif stage == ManufacturingStage.TESTING:
                remaining_hours += spec.testing_duration
            else:
                # Other stages - estimate based on complexity
                stage_total = spec.complexity * 2  # 2 hours per complexity point
                remaining_hours += stage_total
                
        # Add time for open issues
        open_issues = sum(1 for issue in self.workflows[system_id]["issues"] if not issue["resolved"])
        issue_time = open_issues * spec.complexity * 0.5  # 0.5 hours per issue per complexity point
        remaining_hours += issue_time
        
        return {
            "system_id": system_id,
            "estimated_hours": remaining_hours,
            "estimated_days": remaining_hours / 24,
            "complexity_factor": spec.complexity,
            "open_issues": open_issues
        }
    
    def generate_manufacturing_report(self, system_id: str) -> Dict[str, Any]:
        """Generate comprehensive manufacturing report."""
        if system_id not in self.workflows:
            return {"error": "Workflow not found"}
            
        workflow = self.workflows[system_id]
        spec = self.manufacturing_specs[system_id]
        
        # Check quality against thresholds
        quality_status = {}
        for metric, value in self.quality_metrics[system_id].items():
            threshold = spec.quality_thresholds.get(metric, 0.7)  # Default threshold
            quality_status[metric] = {
                "value": value,
                "threshold": threshold,
                "passed": value >= threshold
            }
            
        # Calculate material usage
        material_usage = {
            material: {
                "required": quantity,
                "used": quantity * min(1.0, self.stage_completion[system_id][ManufacturingStage.COMPONENT_FABRICATION])
            }
            for material, quantity in spec.material_requirements.items()
        }
        
        return {
            "system_id": system_id,
            "report_generated_at": time.time(),
            "propulsion_type": workflow["propulsion_type"],
            "thrust_rating": workflow["thrust_rating"],
            "manufacturing_status": "Completed" if workflow["completed"] else "In Progress",
            "current_stage": self.current_stage[system_id].name,
            "stage_history": workflow["stage_history"],
            "quality_metrics": quality_status,
            "material_usage": material_usage,
            "issues_summary": {
                "total": len(workflow["issues"]),
                "open": sum(1 for issue in workflow["issues"] if not issue["resolved"]),
                "resolved": sum(1 for issue in workflow["issues"] if issue["resolved"]),
                "by_severity": {
                    severity: sum(1 for issue in workflow["issues"] if issue["severity"] == severity)
                    for severity in range(1, 6)  # Severity 1-5
                }
            },
            "estimated_completion": self.estimate_completion_time(system_id)
        }
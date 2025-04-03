import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.utils.logging_framework import get_logger

logger = get_logger("quality_control")

class QAStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    NEEDS_REVIEW = "needs_review"

@dataclass
class QACheckpoint:
    name: str
    description: str
    required_measurements: List[str]
    tolerance_ranges: Dict[str, tuple]
    is_critical: bool = False

class QualityController:
    def __init__(self):
        self.checkpoints: Dict[str, Dict[str, QACheckpoint]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # Initialize standard QA checkpoints
        self._initialize_standard_checkpoints()
    
    def _initialize_standard_checkpoints(self):
        """Initialize standard QA checkpoints for UCAV manufacturing."""
        wing_checkpoints = {
            "structural": QACheckpoint(
                name="Wing Structural Integrity",
                description="Verify wing structural components",
                required_measurements=["stress_test", "weight_distribution"],
                tolerance_ranges={"stress_test": (0.95, 1.05), "weight_distribution": (0.9, 1.1)},
                is_critical=True
            ),
            "assembly": QACheckpoint(
                name="Wing Assembly Quality",
                description="Check wing assembly precision",
                required_measurements=["alignment", "joint_strength"],
                tolerance_ranges={"alignment": (0.98, 1.02), "joint_strength": (0.95, 1.05)},
                is_critical=True
            )
        }
        
        fuselage_checkpoints = {
            "composite": QACheckpoint(
                name="Composite Layup Quality",
                description="Verify composite material integrity",
                required_measurements=["thickness", "void_content"],
                tolerance_ranges={"thickness": (0.97, 1.03), "void_content": (0, 0.02)},
                is_critical=True
            )
        }
        
        self.checkpoints["wing_assembly"] = wing_checkpoints
        self.checkpoints["fuselage_layup"] = fuselage_checkpoints
    
    def perform_qa_check(self, task_id: str, checkpoint_id: str, 
                        measurements: Dict[str, float]) -> Dict[str, Any]:
        """Perform quality assurance check."""
        if task_id not in self.checkpoints:
            return {"status": QAStatus.FAILED, "error": "No QA checkpoints defined for task"}
            
        if checkpoint_id not in self.checkpoints[task_id]:
            return {"status": QAStatus.FAILED, "error": "Checkpoint not found"}
            
        checkpoint = self.checkpoints[task_id][checkpoint_id]
        
        # Validate all required measurements are present
        missing_measurements = set(checkpoint.required_measurements) - set(measurements.keys())
        if missing_measurements:
            return {
                "status": QAStatus.FAILED,
                "error": f"Missing measurements: {missing_measurements}"
            }
        
        # Check measurements against tolerance ranges
        issues = []
        for measure, value in measurements.items():
            if measure in checkpoint.tolerance_ranges:
                min_val, max_val = checkpoint.tolerance_ranges[measure]
                if not min_val <= value <= max_val:
                    issues.append(f"{measure} out of tolerance range")
        
        # Store and return results
        result = {
            "status": QAStatus.FAILED if issues else QAStatus.PASSED,
            "timestamp": datetime.now(),
            "measurements": measurements,
            "issues": issues,
            "checkpoint": checkpoint.name
        }
        
        if task_id not in self.results:
            self.results[task_id] = {}
        self.results[task_id][checkpoint_id] = result
        
        return result
    
    def get_qa_results(self, task_id: str) -> Dict[str, Any]:
        """Get QA results for a task."""
        return self.results.get(task_id, {})
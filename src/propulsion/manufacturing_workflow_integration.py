"""
Manufacturing workflow integration system for propulsion components.
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
from src.propulsion.manufacturing_workflow import (
    ManufacturingWorkflow, 
    ManufacturingSpec, 
    ManufacturingStage
)
from src.propulsion.propulsion_optimization_integration import PropulsionOptimizationIntegrator


@dataclass
class ManufacturingIntegrationConfig:
    """Configuration for manufacturing workflow integration."""
    auto_quality_checks: bool = True
    auto_stage_advancement: bool = False
    quality_threshold: float = 0.75
    optimization_feedback: bool = True
    material_tracking: bool = True
    issue_notification: bool = True


class ManufacturingWorkflowIntegrator:
    """Integrates manufacturing workflows with propulsion systems."""
    
    def __init__(self, 
                 config: ManufacturingIntegrationConfig,
                 workflow_manager: Optional[ManufacturingWorkflow] = None,
                 optimization_integrator: Optional[PropulsionOptimizationIntegrator] = None):
        """Initialize manufacturing workflow integrator."""
        self.config = config
        self.workflow_manager = workflow_manager or ManufacturingWorkflow()
        self.optimization_integrator = optimization_integrator
        
        # Track manufacturing data
        self.active_systems: Dict[str, PropulsionInterface] = {}
        self.material_inventory: Dict[str, float] = {}
        self.quality_feedback: Dict[str, List[Dict[str, Any]]] = {}
        self.optimization_feedback: Dict[str, List[Dict[str, Any]]] = {}
        
    def register_system(self, 
                      system_id: str, 
                      system: PropulsionInterface,
                      manufacturing_spec: ManufacturingSpec) -> bool:
        """Register a propulsion system for manufacturing integration."""
        # Register with workflow manager
        success = self.workflow_manager.create_workflow(
            system_id, system, manufacturing_spec
        )
        
        if not success:
            return False
            
        # Track in integrator
        self.active_systems[system_id] = system
        self.quality_feedback[system_id] = []
        self.optimization_feedback[system_id] = []
        
        # Initialize material tracking
        if self.config.material_tracking:
            for material, quantity in manufacturing_spec.material_requirements.items():
                if material not in self.material_inventory:
                    self.material_inventory[material] = 0.0
        
        return True
    
    def update_material_inventory(self, material: str, quantity: float) -> float:
        """Update material inventory levels."""
        if material not in self.material_inventory:
            self.material_inventory[material] = 0.0
            
        self.material_inventory[material] += quantity
        return self.material_inventory[material]
    
    def check_material_availability(self, system_id: str) -> Dict[str, Any]:
        """Check if materials are available for manufacturing."""
        if system_id not in self.workflow_manager.manufacturing_specs:
            return {"available": False, "error": "System not found"}
            
        spec = self.workflow_manager.manufacturing_specs[system_id]
        
        # Check each required material
        availability = {}
        all_available = True
        
        for material, required in spec.material_requirements.items():
            available = self.material_inventory.get(material, 0.0)
            availability[material] = {
                "required": required,
                "available": available,
                "sufficient": available >= required
            }
            
            if available < required:
                all_available = False
        
        return {
            "available": all_available,
            "materials": availability
        }
    
    def consume_materials(self, system_id: str, stage: ManufacturingStage) -> Dict[str, Any]:
        """Consume materials for a manufacturing stage."""
        if system_id not in self.workflow_manager.manufacturing_specs:
            return {"success": False, "error": "System not found"}
            
        if not self.config.material_tracking:
            return {"success": True, "tracking_disabled": True}
            
        spec = self.workflow_manager.manufacturing_specs[system_id]
        current_stage = self.workflow_manager.current_stage[system_id]
        
        # Only consume materials during component fabrication
        if stage != ManufacturingStage.COMPONENT_FABRICATION:
            return {"success": True, "no_consumption": True}
            
        # Check availability first
        availability = self.check_material_availability(system_id)
        if not availability["available"]:
            return {"success": False, "error": "Insufficient materials"}
            
        # Consume materials
        consumed = {}
        for material, required in spec.material_requirements.items():
            self.material_inventory[material] -= required
            consumed[material] = required
            
        return {
            "success": True,
            "consumed": consumed
        }
    
    def perform_quality_check(self, system_id: str) -> Dict[str, Any]:
        """Perform quality check on manufacturing process."""
        if system_id not in self.workflow_manager.quality_metrics:
            return {"success": False, "error": "System not found"}
            
        metrics = self.workflow_manager.quality_metrics[system_id]
        spec = self.workflow_manager.manufacturing_specs[system_id]
        current_stage = self.workflow_manager.current_stage[system_id]
        
        # Check each quality metric against thresholds
        results = {}
        passed = True
        
        for metric, value in metrics.items():
            threshold = spec.quality_thresholds.get(metric, self.config.quality_threshold)
            metric_passed = value >= threshold
            
            results[metric] = {
                "value": value,
                "threshold": threshold,
                "passed": metric_passed
            }
            
            if not metric_passed:
                passed = False
                
        # Record quality feedback
        self.quality_feedback[system_id].append({
            "timestamp": time.time(),
            "stage": current_stage.name,
            "passed": passed,
            "metrics": results
        })
        
        return {
            "success": True,
            "passed": passed,
            "results": results,
            "stage": current_stage.name
        }
    
    def apply_optimization_feedback(self, system_id: str) -> Dict[str, Any]:
        """Apply optimization feedback to manufacturing process."""
        if not self.optimization_integrator or not self.config.optimization_feedback:
            return {"success": False, "error": "Optimization feedback not enabled"}
            
        if system_id not in self.active_systems:
            return {"success": False, "error": "System not found"}
            
        # Get optimization report
        opt_report = self.optimization_integrator.get_optimization_report(system_id)
        
        # Extract relevant metrics for manufacturing
        if "current_status" not in opt_report:
            return {"success": False, "error": "No optimization data available"}
            
        # Apply feedback to quality metrics
        quality_updates = {}
        
        # Map optimization metrics to quality metrics
        if "efficiency" in opt_report.get("current_status", {}):
            quality_updates["power_efficiency"] = opt_report["current_status"]["efficiency"]
            
        if "temperature" in opt_report.get("current_status", {}):
            # Normalize temperature to quality metric (0-1)
            temp = opt_report["current_status"]["temperature"]
            max_temp = 1000.0  # Example max temperature
            quality_updates["thermal_resistance"] = max(0.0, 1.0 - (temp / max_temp))
            
        # Update quality metrics if we have data
        if quality_updates:
            self.workflow_manager.update_progress(
                system_id, 0.0, quality_updates
            )
            
            # Record feedback application
            self.optimization_feedback[system_id].append({
                "timestamp": time.time(),
                "applied_updates": quality_updates,
                "source_metrics": opt_report.get("current_status", {})
            })
            
        return {
            "success": True,
            "updates_applied": len(quality_updates) > 0,
            "quality_updates": quality_updates
        }
    
    def process_manufacturing_step(self, 
                                 system_id: str, 
                                 progress_increment: float = 0.1) -> Dict[str, Any]:
        """Process a single manufacturing step with all integrations."""
        if system_id not in self.active_systems:
            return {"success": False, "error": "System not found"}
            
        current_stage = self.workflow_manager.current_stage[system_id]
        
        # 1. Update progress
        self.workflow_manager.update_progress(system_id, progress_increment)
        
        # 2. Consume materials if needed
        material_result = self.consume_materials(system_id, current_stage)
        
        # 3. Apply optimization feedback if available
        opt_feedback = {}
        if self.optimization_integrator and self.config.optimization_feedback:
            opt_feedback = self.apply_optimization_feedback(system_id)
        
        # 4. Perform quality check if enabled
        quality_result = {}
        if self.config.auto_quality_checks:
            quality_result = self.perform_quality_check(system_id)
        
        # 5. Auto-advance stage if enabled and stage is complete
        advance_result = {"advanced": False}
        current_progress = self.workflow_manager.stage_completion[system_id][current_stage]
        
        if (self.config.auto_stage_advancement and 
            current_progress >= 1.0 and
            (not self.config.auto_quality_checks or quality_result.get("passed", False))):
            
            advanced = self.workflow_manager.advance_stage(system_id)
            advance_result = {"advanced": advanced, "to_stage": self.workflow_manager.current_stage[system_id].name}
        
        # 6. Get current status
        status = self.workflow_manager.get_workflow_status(system_id)
        
        return {
            "success": True,
            "system_id": system_id,
            "current_stage": current_stage.name,
            "progress": current_progress,
            "materials": material_result,
            "quality_check": quality_result,
            "optimization_feedback": opt_feedback,
            "stage_advanced": advance_result,
            "status": status
        }
    
    def generate_integrated_report(self, system_id: str) -> Dict[str, Any]:
        """Generate comprehensive integrated manufacturing report."""
        if system_id not in self.active_systems:
            return {"error": "System not found"}
            
        # Get base manufacturing report
        base_report = self.workflow_manager.generate_manufacturing_report(system_id)
        
        # Add optimization data if available
        optimization_data = {}
        if self.optimization_integrator:
            try:
                optimization_data = self.optimization_integrator.get_optimization_report(system_id)
            except:
                optimization_data = {"error": "Failed to get optimization data"}
        
        # Add material tracking data
        material_data = {}
        if self.config.material_tracking:
            material_data = self.check_material_availability(system_id)
        
        # Add quality feedback history
        quality_history = self.quality_feedback.get(system_id, [])
        
        return {
            **base_report,
            "optimization_data": optimization_data,
            "material_data": material_data,
            "quality_history": quality_history,
            "optimization_feedback_history": self.optimization_feedback.get(system_id, []),
            "integration_config": {
                "auto_quality_checks": self.config.auto_quality_checks,
                "auto_stage_advancement": self.config.auto_stage_advancement,
                "quality_threshold": self.config.quality_threshold,
                "optimization_feedback": self.config.optimization_feedback,
                "material_tracking": self.config.material_tracking
            }
        }
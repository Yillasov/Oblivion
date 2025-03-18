"""
Simplified manufacturing workflow for propulsion systems.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import time

from src.propulsion.base import PropulsionInterface


class ManufacturingStage(Enum):
    """Basic manufacturing stages for propulsion systems."""
    DESIGN = 0
    FABRICATION = 1
    ASSEMBLY = 2
    TESTING = 3
    QUALITY_CHECK = 4


@dataclass
class ManufacturingConfig:
    """Simple configuration for manufacturing workflow."""
    auto_advance: bool = False
    quality_threshold: float = 0.8
    record_metrics: bool = True


class PropulsionManufacturingWorkflow:
    """Simplified manufacturing workflow for propulsion systems."""
    
    def __init__(self, config: Optional[ManufacturingConfig] = None):
        """Initialize manufacturing workflow."""
        self.config = config or ManufacturingConfig()
        self.systems: Dict[str, PropulsionInterface] = {}
        self.stages: Dict[str, ManufacturingStage] = {}
        self.progress: Dict[str, Dict[ManufacturingStage, float]] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        
    def register_system(self, system_id: str, system: PropulsionInterface) -> bool:
        """Register a propulsion system for manufacturing."""
        if system_id in self.systems:
            return False
            
        self.systems[system_id] = system
        self.stages[system_id] = ManufacturingStage.DESIGN
        self.progress[system_id] = {stage: 0.0 for stage in ManufacturingStage}
        self.metrics[system_id] = {"quality": 0.0, "efficiency": 0.0}
        self.history[system_id] = []
        
        return True
        
    def update_progress(self, 
                       system_id: str, 
                       increment: float, 
                       quality_update: Optional[float] = None) -> Dict[str, Any]:
        """Update manufacturing progress for a system."""
        if system_id not in self.systems:
            return {"success": False, "error": "System not found"}
            
        current_stage = self.stages[system_id]
        
        # Update progress
        self.progress[system_id][current_stage] += increment
        self.progress[system_id][current_stage] = min(1.0, self.progress[system_id][current_stage])
        
        # Update quality if provided
        if quality_update is not None:
            self.metrics[system_id]["quality"] = quality_update
            
        # Record history
        self.history[system_id].append({
            "timestamp": time.time(),
            "stage": current_stage.name,
            "progress": self.progress[system_id][current_stage],
            "metrics": self.metrics[system_id].copy()
        })
        
        # Auto-advance if enabled and stage complete
        if (self.config.auto_advance and 
            self.progress[system_id][current_stage] >= 1.0 and
            self.metrics[system_id]["quality"] >= self.config.quality_threshold):
            self.advance_stage(system_id)
            
        return {
            "success": True,
            "system_id": system_id,
            "stage": self.stages[system_id].name,
            "progress": self.progress[system_id][current_stage],
            "metrics": self.metrics[system_id]
        }
        
    def advance_stage(self, system_id: str) -> bool:
        """Advance to next manufacturing stage."""
        if system_id not in self.systems:
            return False
            
        current_stage = self.stages[system_id]
        
        # Check if already at final stage
        if current_stage == ManufacturingStage.QUALITY_CHECK:
            return False
            
        # Advance to next stage
        next_stage = ManufacturingStage(current_stage.value + 1)
        self.stages[system_id] = next_stage
        
        return True
        
    def get_status(self, system_id: str) -> Dict[str, Any]:
        """Get manufacturing status for a system."""
        if system_id not in self.systems:
            return {"error": "System not found"}
            
        # Calculate overall progress
        total_stages = len(ManufacturingStage)
        completed_stages = sum(1 for stage, progress in self.progress[system_id].items() 
                             if progress >= 1.0 and stage.value < self.stages[system_id].value)
        current_progress = self.progress[system_id][self.stages[system_id]]
        
        overall_progress = (completed_stages + current_progress) / total_stages
        
        return {
            "system_id": system_id,
            "current_stage": self.stages[system_id].name,
            "stage_progress": current_progress,
            "overall_progress": overall_progress,
            "quality": self.metrics[system_id]["quality"],
            "completed": self.stages[system_id] == ManufacturingStage.QUALITY_CHECK and current_progress >= 1.0
        }
        
    def integrate_with_simulation(self, system_id: str, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate manufacturing with simulation results."""
        if system_id not in self.systems:
            return {"success": False, "error": "System not found"}
            
        # Extract relevant metrics from simulation
        if "system_results" in simulation_results and system_id in simulation_results["system_results"]:
            system_results = simulation_results["system_results"][system_id]
            
            # Update manufacturing quality based on simulation performance
            if "average_performance" in system_results:
                perf = system_results["average_performance"]
                
                # Calculate quality score from performance metrics
                efficiency = perf.get("efficiency", 0.0)
                reliability = 1.0 - min(1.0, perf.get("failures", 0) / 10.0)
                
                # Update quality metric
                quality_score = (efficiency + reliability) / 2.0
                self.metrics[system_id]["quality"] = quality_score
                self.metrics[system_id]["efficiency"] = efficiency
                
                return {
                    "success": True,
                    "system_id": system_id,
                    "quality_updated": True,
                    "new_quality": quality_score
                }
                
        return {"success": False, "error": "No relevant simulation data"}
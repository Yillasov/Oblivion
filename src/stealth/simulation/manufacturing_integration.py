"""
Integration between stealth systems and manufacturing workflows.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from src.stealth.base.config import StealthMaterialConfig
from src.stealth.materials.ram.ram_system import RAMMaterial
from src.stealth.simulation.material_sim import MaterialPropertiesSimulator


@dataclass
class StealthManufacturingSpec:
    """Manufacturing specifications for stealth materials."""
    material_id: str
    material_type: str
    thickness_mm: float
    coverage_percentage: float
    estimated_production_time_hours: float
    required_materials: Dict[str, float]  # material name -> quantity
    quality_checks: List[str]


class StealthManufacturingStage(Enum):
    """Manufacturing stages for stealth systems."""
    DESIGN = 0
    MATERIAL_PREPARATION = 1
    APPLICATION = 2
    CURING = 3
    TESTING = 4
    QUALITY_CONTROL = 5


class StealthManufacturingIntegration:
    """Integrates stealth systems with manufacturing workflows."""
    
    def __init__(self, material_simulator: Optional[MaterialPropertiesSimulator] = None):
        """Initialize stealth manufacturing integration."""
        self.material_simulator = material_simulator or MaterialPropertiesSimulator()
        self.registered_materials: Dict[str, StealthManufacturingSpec] = {}
        self.manufacturing_progress: Dict[str, Dict[StealthManufacturingStage, float]] = {}
        self.quality_metrics: Dict[str, Dict[str, float]] = {}
        
    def register_material(self, material_id: str, spec: StealthManufacturingSpec) -> bool:
        """Register a stealth material for manufacturing."""
        if material_id in self.registered_materials:
            return False
            
        self.registered_materials[material_id] = spec
        self.manufacturing_progress[material_id] = {stage: 0.0 for stage in StealthManufacturingStage}
        self.quality_metrics[material_id] = {
            "surface_uniformity": 0.0,
            "adhesion_quality": 0.0,
            "thickness_consistency": 0.0,
            "radar_absorption": 0.0
        }
        
        return True
        
    def update_progress(self, 
                      material_id: str, 
                      stage: StealthManufacturingStage, 
                      increment: float) -> Dict[str, Any]:
        """Update manufacturing progress for a stealth material."""
        if material_id not in self.registered_materials:
            return {"success": False, "error": "Material not registered"}
            
        self.manufacturing_progress[material_id][stage] += increment
        self.manufacturing_progress[material_id][stage] = min(1.0, self.manufacturing_progress[material_id][stage])
        
        return {
            "success": True,
            "material_id": material_id,
            "stage": stage.name,
            "progress": self.manufacturing_progress[material_id][stage]
        }
        
    def get_manufacturing_status(self, material_id: str) -> Dict[str, Any]:
        """Get manufacturing status for a stealth material."""
        if material_id not in self.registered_materials:
            return {"success": False, "error": "Material not registered"}
            
        # Calculate overall progress
        total_stages = len(StealthManufacturingStage)
        current_stage = next((stage for stage in StealthManufacturingStage 
                           if self.manufacturing_progress[material_id][stage] < 1.0), 
                           StealthManufacturingStage.QUALITY_CONTROL)
        
        completed_stages = sum(1 for stage in StealthManufacturingStage 
                             if stage.value < current_stage.value)
        
        current_progress = self.manufacturing_progress[material_id][current_stage]
        overall_progress = (completed_stages + current_progress) / total_stages
        
        return {
            "success": True,
            "material_id": material_id,
            "material_type": self.registered_materials[material_id].material_type,
            "current_stage": current_stage.name,
            "stage_progress": current_progress,
            "overall_progress": overall_progress,
            "quality_metrics": self.quality_metrics[material_id]
        }
        
    def generate_manufacturing_report(self, material_id: str) -> Dict[str, Any]:
        """Generate a manufacturing report for a stealth material."""
        if material_id not in self.registered_materials:
            return {"success": False, "error": "Material not registered"}
            
        spec = self.registered_materials[material_id]
        status = self.get_manufacturing_status(material_id)
        
        # Simulate material properties for quality assessment
        material_config = StealthMaterialConfig(
            material_type=spec.material_type,
            thickness_mm=spec.thickness_mm,
            coverage_percentage=spec.coverage_percentage
        )
        
        simulated_properties = self.material_simulator.simulate_material(
            material_config, 
            {"temperature": 22.0, "humidity": 50.0}
        )
        
        return {
            "success": True,
            "material_id": material_id,
            "manufacturing_status": status,
            "material_specs": {
                "type": spec.material_type,
                "thickness_mm": spec.thickness_mm,
                "coverage_percentage": spec.coverage_percentage
            },
            "production_metrics": {
                "estimated_time_hours": spec.estimated_production_time_hours,
                "required_materials": spec.required_materials,
                "quality_checks": spec.quality_checks
            },
            "simulated_properties": simulated_properties
        }
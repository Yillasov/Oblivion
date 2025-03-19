from typing import Dict, List, Any, Optional, Union, cast
from enum import Enum

from ..base.interfaces import StealthInterface
from ..base.config import StealthSystemConfig, StealthType
from ..design.parametric_design import ParametricStealthDesigner, CADIntegration
from ..effectiveness.stealth_effectiveness import StealthEffectivenessEvaluator
# Fix the import by creating a proper path
from ..neuromorphic.neuromorphic_integration import NeuromorphicIntegration
# Add import for EquipmentManager
from src.manufacturing.equipment.manager import EquipmentManager


class StealthAPI:
    """Unified API for accessing stealth capabilities across the system."""
    
    def __init__(self):
        """Initialize the stealth API."""
        self.stealth_systems: Dict[str, StealthInterface] = {}
        self.designer = ParametricStealthDesigner()
        self.cad_integration = CADIntegration(self.designer)
        self.effectiveness_evaluator = StealthEffectivenessEvaluator()
        self.neuromorphic_integration = NeuromorphicIntegration()
        # Add equipment manager
        self.equipment_manager = EquipmentManager()
        
    def register_stealth_system(self, system_id: str, system: StealthInterface) -> bool:
        """Register a stealth system with the API."""
        if system_id in self.stealth_systems:
            return False
            
        self.stealth_systems[system_id] = system
        self.effectiveness_evaluator.register_stealth_system(system_id, system)
        self.neuromorphic_integration.register_stealth_system(system_id, system)
        return True
        
    def get_stealth_system(self, system_id: str) -> Optional[StealthInterface]:
        """Get a registered stealth system by ID."""
        return self.stealth_systems.get(system_id)
        
    def list_stealth_systems(self) -> List[str]:
        """List all registered stealth system IDs."""
        return list(self.stealth_systems.keys())
        
    def create_stealth_design(self, platform_type: str, dimensions: Dict[str, float], 
                             primary_goal: str) -> Dict[str, Any]:
        """Create a new stealth design."""
        from ..design.parametric_design import DesignOptimizationGoal
        goal = DesignOptimizationGoal[primary_goal]
        return self.designer.create_new_design(platform_type, dimensions, goal)
        
    def generate_manufacturing_specs(self) -> Dict[str, Any]:
        """Generate manufacturing specifications for the current design."""
        return self.cad_integration.generate_manufacturing_specs()
        
    def evaluate_stealth_effectiveness(self, system_id: str, 
                                     threat_data: Dict[str, Any],
                                     environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate stealth effectiveness against specific threats."""
        system = self.get_stealth_system(system_id)
        if not system:
            # Fix the return type to match Dict[str, Any] instead of Dict[str, float]
            return {"error": "System not found"}
            
        return system.calculate_effectiveness(threat_data, environmental_conditions)
        
    def activate_stealth_system(self, system_id: str, 
                              activation_params: Dict[str, Any] = {}) -> bool:
        """Activate a stealth system."""
        system = self.get_stealth_system(system_id)
        if not system:
            return False
            
        return system.activate(activation_params or {})
        
    def deactivate_stealth_system(self, system_id: str) -> bool:
        """Deactivate a stealth system."""
        system = self.get_stealth_system(system_id)
        if not system:
            return False
            
        return system.deactivate()
        
    def get_system_status(self, system_id: str) -> Dict[str, Any]:
        """Get the status of a stealth system."""
        system = self.get_stealth_system(system_id)
        if not system:
            return {"error": "System not found"}
            
        return system.get_status()
    
    # Add new method to connect with manufacturing equipment
    def get_required_manufacturing_equipment(self) -> List[str]:
        """Get required manufacturing equipment for current stealth design."""
        if not self.designer.current_design:
            return []
            
        # Use CAD integration to determine required equipment
        return self.cad_integration._get_required_manufacturing_equipment()
    
    def check_equipment_availability(self) -> Dict[str, bool]:
        """Check availability of required manufacturing equipment."""
        required_equipment = self.get_required_manufacturing_equipment()
        availability = {}
        
        for equipment_id in required_equipment:
            # Fix: Use equipment dictionary directly instead of has_equipment method
            availability[equipment_id] = equipment_id in self.equipment_manager.equipment
            
        return availability
    
    def reserve_manufacturing_equipment(self, start_time: str, duration_hours: float) -> Dict[str, bool]:
        """Reserve required manufacturing equipment for production."""
        required_equipment = self.get_required_manufacturing_equipment()
        reservation_results = {}
        
        for equipment_id in required_equipment:
            # Fix: Access equipment directly from the dictionary
            equipment = self.equipment_manager.equipment.get(equipment_id)
            if equipment:
                # Instead of using a non-existent reserve method, use execute_operation
                # which is part of the ManufacturingEquipment interface
                operation_result = equipment.execute_operation(
                    "reserve", 
                    {
                        "start_time": start_time,
                        "duration_hours": duration_hours
                    }
                )
                reservation_results[equipment_id] = operation_result.get("success", False)
            else:
                reservation_results[equipment_id] = False
                
        return reservation_results
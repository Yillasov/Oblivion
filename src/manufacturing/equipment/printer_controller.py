import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, Optional
from datetime import datetime
from src.manufacturing.equipment.base import ManufacturingEquipment, EquipmentStatus

class ThreeDPrinterController(ManufacturingEquipment):
    """Controller for 3D printer manufacturing equipment."""
    
    def initialize(self) -> bool:
        """Initialize 3D printer."""
        try:
            self.status = EquipmentStatus(
                operational=True,
                temperature=25.0,
                power_state="standby",
                last_maintenance=datetime.now()
            )
            return True
        except Exception:
            self.status.error_code = 1
            return False
    
    def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute 3D printer operation."""
        if not self.status.operational:
            return {"success": False, "error": "Equipment not operational"}
        
        operations = {
            "heat_bed": self._heat_bed,
            "extrude": self._extrude_material,
            "move_head": self._move_print_head,
            "set_layer": self._configure_layer
        }
        
        if operation not in operations:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        
        return operations[operation](params)
    
    def _heat_bed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Heat printer bed."""
        target_temp = params.get("temperature", 60.0)
        return {
            "success": True,
            "bed_temperature": target_temp,
            "heating_time": 120  # seconds
        }
    
    def _extrude_material(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Control material extrusion."""
        return {
            "success": True,
            "flow_rate": params.get("flow_rate", 100),
            "material_type": params.get("material", "PLA")
        }
    
    def _move_print_head(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Move print head to position."""
        return {
            "success": True,
            "position": {
                "x": params.get("x", 0),
                "y": params.get("y", 0),
                "z": params.get("z", 0)
            }
        }
    
    def _configure_layer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure layer parameters."""
        return {
            "success": True,
            "layer_height": params.get("height", 0.2),
            "layer_number": params.get("number", 1)
        }
    
    def shutdown(self) -> bool:
        """Safely shutdown 3D printer."""
        try:
            # Cool down components
            self._heat_bed({"temperature": 0})
            self._extrude_material({"flow_rate": 0})
            
            self.status.power_state = "off"
            self.status.operational = False
            return True
        except Exception:
            self.status.error_code = 2
            return False
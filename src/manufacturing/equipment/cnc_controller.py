import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, Optional
from datetime import datetime
from src.manufacturing.equipment.base import ManufacturingEquipment, EquipmentStatus

class CNCController(ManufacturingEquipment):
    """Controller for CNC manufacturing equipment."""
    
    def initialize(self) -> bool:
        """Initialize CNC machine."""
        try:
            # Simulate hardware initialization
            self.status = EquipmentStatus(
                operational=True,
                temperature=25.0,
                power_state="standby",
                last_maintenance=datetime.now()
            )
            return True
        except Exception as e:
            self.status.error_code = 1
            return False
    
    def shutdown(self) -> bool:
        """Safely shutdown CNC machine."""
        try:
            self.status.power_state = "off"
            self.status.operational = False
            return True
        except Exception:
            return False
    
    def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CNC manufacturing operation."""
        if not self.status.operational:
            return {"success": False, "error": "Equipment not operational"}
        
        operations = {
            "move": self._execute_movement,
            "cut": self._execute_cutting,
            "tool_change": self._change_tool,
            "calibrate": self._calibrate_axes
        }
        
        if operation not in operations:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        
        return operations[operation](params)
    
    def _execute_movement(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute movement operation."""
        required = ["x", "y", "z", "speed"]
        if not all(key in params for key in required):
            return {"success": False, "error": "Missing required parameters"}
            
        return {
            "success": True,
            "position": {
                "x": params["x"],
                "y": params["y"],
                "z": params["z"]
            }
        }
    
    def _execute_cutting(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cutting operation."""
        required = ["depth", "feed_rate", "speed"]
        if not all(key in params for key in required):
            return {"success": False, "error": "Missing required parameters"}
            
        return {
            "success": True,
            "cut_depth": params["depth"],
            "feed_rate": params["feed_rate"]
        }
    
    def _change_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool change operation."""
        if "tool_id" not in params:
            return {"success": False, "error": "Tool ID required"}
            
        return {
            "success": True,
            "tool_id": params["tool_id"],
            "tool_type": params.get("tool_type", "generic")
        }
    
    def _calibrate_axes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrate machine axes."""
        axes = params.get("axes", ["x", "y", "z"])
        return {
            "success": True,
            "calibrated_axes": axes,
            "accuracy": 0.01  # mm
        }
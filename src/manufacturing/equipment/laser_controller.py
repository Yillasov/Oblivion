import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, Optional
from datetime import datetime
from src.manufacturing.equipment.base import ManufacturingEquipment, EquipmentStatus

class LaserCutterController(ManufacturingEquipment):
    """Controller for laser cutter manufacturing equipment."""
    
    def initialize(self) -> bool:
        """Initialize laser cutter."""
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
        """Execute laser cutter operation."""
        if not self.status.operational:
            return {"success": False, "error": "Equipment not operational"}
        
        operations = {
            "set_power": self._set_laser_power,
            "set_speed": self._set_cutting_speed,
            "move_laser": self._move_laser_head,
            "air_assist": self._control_air_assist
        }
        
        if operation not in operations:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        
        return operations[operation](params)
    
    def _set_laser_power(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set laser power level."""
        power = params.get("power", 50.0)
        return {
            "success": True,
            "power_level": power,
            "mode": "continuous" if power > 80 else "pulsed"
        }
    
    def _set_cutting_speed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set cutting speed."""
        return {
            "success": True,
            "speed": params.get("speed", 100),
            "acceleration": params.get("acceleration", 500)
        }
    
    def _move_laser_head(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Move laser head to position."""
        return {
            "success": True,
            "position": {
                "x": params.get("x", 0),
                "y": params.get("y", 0)
            }
        }
    
    def _control_air_assist(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Control air assist system."""
        return {
            "success": True,
            "enabled": params.get("enabled", True),
            "pressure": params.get("pressure", 2.5)  # bar
        }
    
    def shutdown(self) -> bool:
        """Safely shutdown laser cutter."""
        try:
            # Disable laser and cooling
            self._set_laser_power({"power": 0})
            self._control_air_assist({"enabled": False})
            
            self.status.power_state = "off"
            self.status.operational = False
            return True
        except Exception:
            self.status.error_code = 2
            return False
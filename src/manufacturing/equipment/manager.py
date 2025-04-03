import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, Optional
from src.manufacturing.equipment.base import ManufacturingEquipment
from src.manufacturing.equipment.cnc_controller import CNCController
from src.manufacturing.equipment.printer_controller import ThreeDPrinterController
from src.manufacturing.equipment.laser_controller import LaserCutterController

class EquipmentManager:
    """Manager for manufacturing equipment."""
    
    def __init__(self):
        self.equipment: Dict[str, ManufacturingEquipment] = {}
    
    def add_equipment(self, equipment_id: str, equipment_type: str, 
                     config: Optional[Dict[str, Any]] = None) -> bool:
        """Add new equipment to manager."""
        if equipment_id in self.equipment:
            return False
            
        equipment_types = {
            "cnc": lambda: CNCController(equipment_id, config),
            "3d_printer": lambda: ThreeDPrinterController(equipment_id, config),
            "laser_cutter": lambda: LaserCutterController(equipment_id, config)
        }
        
        if equipment_type not in equipment_types:
            return False
            
        self.equipment[equipment_id] = equipment_types[equipment_type]()
        return True
    
    def initialize_all(self) -> Dict[str, bool]:
        """Initialize all equipment."""
        results = {}
        for equipment_id, equipment in self.equipment.items():
            results[equipment_id] = equipment.initialize()
        return results
    
    def shutdown_all(self) -> Dict[str, bool]:
        """Shutdown all equipment."""
        results = {}
        for equipment_id, equipment in self.equipment.items():
            results[equipment_id] = equipment.shutdown()
        return results
    
    def get_equipment_status(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific equipment."""
        if equipment_id not in self.equipment:
            return None
            
        status = self.equipment[equipment_id].get_status()
        return {
            "operational": status.operational,
            "temperature": status.temperature,
            "power_state": status.power_state,
            "error_code": status.error_code,
            "last_maintenance": status.last_maintenance.isoformat() if status.last_maintenance else None
        }
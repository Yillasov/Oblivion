from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EquipmentStatus:
    operational: bool
    temperature: float
    power_state: str
    error_code: Optional[int] = None
    last_maintenance: Optional[datetime] = None

class ManufacturingEquipment(ABC):
    """Base class for manufacturing equipment control."""
    
    def __init__(self, equipment_id: str, config: Optional[Dict[str, Any]] = None):
        self.equipment_id = equipment_id
        self.config = config or {}
        self.status = EquipmentStatus(
            operational=False,
            temperature=0.0,
            power_state="off"
        )
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize equipment."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Safely shutdown equipment."""
        pass
    
    @abstractmethod
    def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manufacturing operation."""
        pass
    
    def get_status(self) -> EquipmentStatus:
        """Get current equipment status."""
        return self.status
import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

from src.core.utils.logging_framework import get_logger
from src.core.hardware.resource_sharing import ResourceSharingManager

logger = get_logger("manufacturing_resources")

class ManufacturingResourceType(Enum):
    CNC_MACHINE = "cnc_machine"
    COMPOSITE_LAYUP = "composite_layup"
    ASSEMBLY_STATION = "assembly_station"
    QUALITY_CONTROL = "quality_control"
    TESTING_EQUIPMENT = "testing_equipment"

@dataclass
class ResourceRequest:
    resource_type: ManufacturingResourceType
    quantity: int
    duration: timedelta
    priority: int
    task_id: str

class ManufacturingResourceAllocator:
    """Manages manufacturing resource allocation."""
    
    def __init__(self):
        self.resource_limits = {
            ManufacturingResourceType.CNC_MACHINE: 2,
            ManufacturingResourceType.COMPOSITE_LAYUP: 3,
            ManufacturingResourceType.ASSEMBLY_STATION: 4,
            ManufacturingResourceType.QUALITY_CONTROL: 2,
            ManufacturingResourceType.TESTING_EQUIPMENT: 2
        }
        self.resource_manager = ResourceSharingManager.get_instance()
        self.allocations: Dict[str, List[ResourceRequest]] = {}
        self._lock = threading.Lock()
        
    def request_resources(self, request: ResourceRequest) -> bool:
        """Request resources for a manufacturing task."""
        with self._lock:
            if not self._validate_request(request):
                return False
                
            # Check resource availability
            if not self._check_availability(request):
                logger.warning(f"Resources not available for {request.task_id}")
                return False
            
            # Allocate resources
            if request.task_id not in self.allocations:
                self.allocations[request.task_id] = []
            
            self.allocations[request.task_id].append(request)
            logger.info(f"Resources allocated for task {request.task_id}")
            
            return True
    
    def release_resources(self, task_id: str) -> None:
        """Release all resources allocated to a task."""
        with self._lock:
            if task_id in self.allocations:
                del self.allocations[task_id]
                logger.info(f"Resources released for task {task_id}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource allocation status."""
        with self._lock:
            status = {}
            for resource_type in ManufacturingResourceType:
                used = self._get_used_resources(resource_type)
                total = self.resource_limits[resource_type]
                status[resource_type.value] = {
                    "used": used,
                    "available": total - used,
                    "total": total
                }
            return status
    
    def _validate_request(self, request: ResourceRequest) -> bool:
        """Validate resource request."""
        if request.quantity <= 0:
            logger.error("Invalid resource quantity requested")
            return False
            
        if request.resource_type not in self.resource_limits:
            logger.error(f"Unknown resource type: {request.resource_type}")
            return False
            
        return True
    
    def _check_availability(self, request: ResourceRequest) -> bool:
        """Check if requested resources are available."""
        used_resources = self._get_used_resources(request.resource_type)
        return (used_resources + request.quantity) <= self.resource_limits[request.resource_type]
    
    def _get_used_resources(self, resource_type: ManufacturingResourceType) -> int:
        """Get total used resources of a specific type."""
        return sum(
            req.quantity for reqs in self.allocations.values()
            for req in reqs if req.resource_type == resource_type
        )
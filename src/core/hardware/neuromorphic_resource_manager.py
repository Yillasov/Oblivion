#!/usr/bin/env python3
"""
Centralized resource manager for neuromorphic hardware allocation.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING
from enum import Enum, auto
import logging
import threading
from dataclasses import dataclass, field

# Only import type definitions to avoid circular imports
if TYPE_CHECKING:
    from src.core.hardware.resource_sharing_manager import ResourceSharingManager

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of neuromorphic resources that can be allocated."""
    NEURON = auto()
    SYNAPSE = auto()
    MEMORY = auto()
    COMPUTE = auto()
    IO_BANDWIDTH = auto()

@dataclass
class ResourceAllocation:
    """Represents a resource allocation."""
    allocation_id: str
    resource_type: ResourceType
    amount: int
    owner_id: str
    hardware_type: str
    timestamp: float
    priority: int = 0

@dataclass
class ResourcePool:
    """Resource pool for a specific hardware type."""
    hardware_type: str
    total_resources: Dict[ResourceType, int]
    allocated_resources: Dict[ResourceType, int] = field(default_factory=dict)
    allocations: Dict[str, ResourceAllocation] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize allocated resources."""
        self.allocated_resources = {rtype: 0 for rtype in self.total_resources.keys()}
    
    def available(self, resource_type: ResourceType) -> int:
        """Get available amount of a resource type."""
        if resource_type not in self.total_resources:
            return 0
        return self.total_resources[resource_type] - self.allocated_resources.get(resource_type, 0)
    
    def utilization(self, resource_type: ResourceType) -> float:
        """Get utilization percentage of a resource type."""
        if resource_type not in self.total_resources or self.total_resources[resource_type] == 0:
            return 0.0
        return self.allocated_resources.get(resource_type, 0) / self.total_resources[resource_type]


class NeuromorphicResourceManager:
    """Centralized manager for neuromorphic hardware resources."""
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'NeuromorphicResourceManager':
        """Get singleton instance of the resource manager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """Initialize the resource manager."""
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.registered_processes: Set[str] = set()
        self.allocation_counter = 0
    
    def register_hardware(self, hardware_type: str, resources: Dict[ResourceType, int]) -> bool:
        """Register a neuromorphic hardware with its available resources."""
        with self._lock:
            if hardware_type in self.resource_pools:
                logger.warning(f"Hardware type {hardware_type} already registered")
                return False
            
            self.resource_pools[hardware_type] = ResourcePool(
                hardware_type=hardware_type,
                total_resources=resources
            )
            logger.info(f"Registered neuromorphic hardware: {hardware_type}")
            return True
    
    def register_process(self, process_id: str) -> bool:
        """Register a process that will request resources."""
        with self._lock:
            if process_id in self.registered_processes:
                return False
            self.registered_processes.add(process_id)
            return True
    
    def allocate(self, 
                process_id: str, 
                hardware_type: str, 
                resource_type: ResourceType, 
                amount: int,
                priority: int = 0) -> Optional[str]:
        """
        Allocate resources to a process.
        
        Args:
            process_id: ID of the requesting process
            hardware_type: Type of neuromorphic hardware
            resource_type: Type of resource to allocate
            amount: Amount to allocate
            priority: Priority level (higher numbers = higher priority)
            
        Returns:
            Allocation ID if successful, None otherwise
        """
        with self._lock:
            # Validate request
            if process_id not in self.registered_processes:
                logger.error(f"Process {process_id} not registered")
                return None
                
            if hardware_type not in self.resource_pools:
                logger.error(f"Hardware type {hardware_type} not registered")
                return None
            
            pool = self.resource_pools[hardware_type]
            
            # Check if enough resources are available
            if pool.available(resource_type) < amount:
                logger.warning(f"Not enough {resource_type} resources available in {hardware_type}")
                return None
            
            # Create allocation
            self.allocation_counter += 1
            allocation_id = f"alloc_{self.allocation_counter}"
            
            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                resource_type=resource_type,
                amount=amount,
                owner_id=process_id,
                hardware_type=hardware_type,
                timestamp=0.0,  # Could use time.time() here
                priority=priority
            )
            
            # Update pool
            pool.allocations[allocation_id] = allocation
            pool.allocated_resources[resource_type] += amount
            
            logger.info(f"Allocated {amount} {resource_type} resources to {process_id}")
            return allocation_id
    
    def release(self, allocation_id: str) -> bool:
        """Release a resource allocation."""
        with self._lock:
            # Find the allocation
            for hardware_type, pool in self.resource_pools.items():
                if allocation_id in pool.allocations:
                    allocation = pool.allocations[allocation_id]
                    
                    # Update pool
                    pool.allocated_resources[allocation.resource_type] -= allocation.amount
                    del pool.allocations[allocation_id]
                    
                    logger.info(f"Released allocation {allocation_id}")
                    return True
            
            logger.warning(f"Allocation {allocation_id} not found")
            return False
    
    def get_utilization(self, hardware_type: str) -> Dict[str, float]:
        """Get resource utilization for a hardware type."""
        with self._lock:
            if hardware_type not in self.resource_pools:
                return {}
            
            pool = self.resource_pools[hardware_type]
            return {
                resource_type.name: pool.utilization(resource_type)
                for resource_type in pool.total_resources.keys()
            }
    
    def get_process_allocations(self, process_id: str) -> List[ResourceAllocation]:
        """Get all allocations for a specific process."""
        with self._lock:
            allocations = []
            for pool in self.resource_pools.values():
                for allocation in pool.allocations.values():
                    if allocation.owner_id == process_id:
                        allocations.append(allocation)
            return allocations
    
    def get_pool(self, hardware_type: str) -> Optional[ResourcePool]:
        """
        Get resource pool for a hardware type.
        
        Args:
            hardware_type: Type of neuromorphic hardware
            
        Returns:
            ResourcePool if found, None otherwise
        """
        with self._lock:
            return self.resource_pools.get(hardware_type)
    
    def get_sharing_manager(self) -> 'ResourceSharingManager':
        """
        Get the resource sharing manager.
        
        Returns:
            ResourceSharingManager: Resource sharing manager instance
        """
        # Import here to avoid circular imports
        from src.core.hardware.resource_sharing_manager import ResourceSharingManager
        return ResourceSharingManager.get_instance()
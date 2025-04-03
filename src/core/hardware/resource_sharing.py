#!/usr/bin/env python3
"""
Hardware Resource Sharing

Provides mechanisms for sharing neuromorphic hardware resources between multiple processes.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Set, Tuple
import threading
import time
import uuid
from enum import Enum
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("resource_sharing")


class ResourceSharingError(Exception):
    """Exception raised for errors in hardware resource sharing."""
    pass


class ResourceType(Enum):
    """Types of shareable hardware resources."""
    NEURON = "neuron"
    CORE = "core"
    CHIP = "chip"
    MEMORY = "memory"
    IO = "io"


@dataclass
class ResourceAllocation:
    """Resource allocation details."""
    resource_id: str
    resource_type: ResourceType
    owner_id: str
    quantity: int
    timestamp: float
    expiry: Optional[float] = None


class ResourcePool:
    """Manages a pool of shareable hardware resources."""
    
    def __init__(self, hardware_type: str, resource_limits: Dict[ResourceType, int]):
        """Initialize resource pool."""
        self.hardware_type = hardware_type
        self.resource_limits = resource_limits
        self.allocations: List[ResourceAllocation] = []
        self.lock = threading.RLock()
        
    def allocate(self, owner_id: str, resource_type: ResourceType, 
                quantity: int, timeout: Optional[float] = None) -> str:
        """
        Allocate resources from the pool.
        
        Args:
            owner_id: ID of the resource owner
            resource_type: Type of resource to allocate
            quantity: Amount to allocate
            timeout: Optional timeout in seconds
            
        Returns:
            str: Resource allocation ID
            
        Raises:
            ResourceSharingError: If allocation fails
        """
        with self.lock:
            # Check if enough resources are available
            used = self._get_used_resources(resource_type)
            available = self.resource_limits.get(resource_type, 0) - used
            
            if quantity > available:
                raise ResourceSharingError(
                    f"Not enough {resource_type.value} resources available. "
                    f"Requested: {quantity}, Available: {available}")
            
            # Create allocation
            allocation_id = str(uuid.uuid4())
            allocation = ResourceAllocation(
                resource_id=allocation_id,
                resource_type=resource_type,
                owner_id=owner_id,
                quantity=quantity,
                timestamp=time.time(),
                expiry=time.time() + timeout if timeout else None
            )
            
            self.allocations.append(allocation)
            logger.info(f"Allocated {quantity} {resource_type.value} resources to {owner_id}")
            
            return allocation_id
    
    def release(self, allocation_id: str) -> bool:
        """
        Release allocated resources.
        
        Args:
            allocation_id: Resource allocation ID
            
        Returns:
            bool: Success status
        """
        with self.lock:
            for i, allocation in enumerate(self.allocations):
                if allocation.resource_id == allocation_id:
                    self.allocations.pop(i)
                    logger.info(
                        f"Released {allocation.quantity} {allocation.resource_type.value} "
                        f"resources from {allocation.owner_id}")
                    return True
            
            return False
    
    def release_by_owner(self, owner_id: str) -> int:
        """
        Release all resources allocated to an owner.
        
        Args:
            owner_id: Owner ID
            
        Returns:
            int: Number of allocations released
        """
        with self.lock:
            to_remove = [a for a in self.allocations if a.owner_id == owner_id]
            for allocation in to_remove:
                self.allocations.remove(allocation)
            
            if to_remove:
                logger.info(f"Released all resources owned by {owner_id}")
                
            return len(to_remove)
    
    def _get_used_resources(self, resource_type: ResourceType) -> int:
        """Get total used resources of a specific type."""
        # Clean expired allocations first
        self._clean_expired_allocations()
        
        return sum(a.quantity for a in self.allocations 
                  if a.resource_type == resource_type)
    
    def _clean_expired_allocations(self) -> None:
        """Remove expired allocations."""
        now = time.time()
        expired = [a for a in self.allocations 
                  if a.expiry is not None and a.expiry < now]
        
        for allocation in expired:
            self.allocations.remove(allocation)
            logger.info(
                f"Expired allocation of {allocation.quantity} "
                f"{allocation.resource_type.value} resources from {allocation.owner_id}")


class ResourceSharingManager:
    """Manages resource sharing across hardware platforms."""
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'ResourceSharingManager':
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = ResourceSharingManager()
            return cls._instance
    
    def __init__(self):
        """Initialize resource sharing manager."""
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.process_resources: Dict[str, Set[str]] = {}
        
    def create_pool(self, hardware_type: str, 
                   resource_limits: Dict[ResourceType, int]) -> ResourcePool:
        """
        Create a resource pool for a hardware type.
        
        Args:
            hardware_type: Hardware type
            resource_limits: Resource limits by type
            
        Returns:
            ResourcePool: Created resource pool
        """
        with self._lock:
            if hardware_type in self.resource_pools:
                logger.warning(f"Resource pool for {hardware_type} already exists")
                return self.resource_pools[hardware_type]
            
            pool = ResourcePool(hardware_type, resource_limits)
            self.resource_pools[hardware_type] = pool
            logger.info(f"Created resource pool for {hardware_type}")
            
            return pool
    
    def get_pool(self, hardware_type: str) -> Optional[ResourcePool]:
        """Get resource pool for a hardware type."""
        return self.resource_pools.get(hardware_type)
    
    def register_process(self, process_id: str) -> None:
        """Register a process for resource tracking."""
        with self._lock:
            if process_id not in self.process_resources:
                self.process_resources[process_id] = set()
    
    def allocate_resource(self, process_id: str, hardware_type: str,
                         resource_type: ResourceType, quantity: int,
                         timeout: Optional[float] = None) -> str:
        """
        Allocate resources for a process.
        
        Args:
            process_id: Process ID
            hardware_type: Hardware type
            resource_type: Resource type
            quantity: Amount to allocate
            timeout: Optional timeout in seconds
            
        Returns:
            str: Resource allocation ID
        """
        with self._lock:
            # Register process if needed
            self.register_process(process_id)
            
            # Get or create pool
            pool = self.get_pool(hardware_type)
            if not pool:
                raise ResourceSharingError(f"No resource pool for {hardware_type}")
            
            # Allocate resource
            allocation_id = pool.allocate(process_id, resource_type, quantity, timeout)
            
            # Track allocation
            self.process_resources[process_id].add(allocation_id)
            
            return allocation_id
    
    def release_resource(self, allocation_id: str, hardware_type: str) -> bool:
        """
        Release allocated resources.
        
        Args:
            allocation_id: Resource allocation ID
            hardware_type: Hardware type
            
        Returns:
            bool: Success status
        """
        with self._lock:
            pool = self.get_pool(hardware_type)
            if not pool:
                return False
            
            success = pool.release(allocation_id)
            
            # Update process tracking
            if success:
                for process_id, allocations in self.process_resources.items():
                    if allocation_id in allocations:
                        allocations.remove(allocation_id)
                        break
            
            return success
    
    def cleanup_process(self, process_id: str) -> None:
        """
        Clean up all resources allocated to a process.
        
        Args:
            process_id: Process ID
        """
        with self._lock:
            if process_id not in self.process_resources:
                return
            
            # Release all resources in all pools
            for pool in self.resource_pools.values():
                pool.release_by_owner(process_id)
            
            # Remove process tracking
            del self.process_resources[process_id]
            logger.info(f"Cleaned up resources for process {process_id}")


# Create global instance
resource_manager = ResourceSharingManager.get_instance()


def get_resource_manager() -> ResourceSharingManager:
    """Get the global resource sharing manager."""
    return resource_manager
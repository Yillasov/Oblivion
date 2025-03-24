"""
Resource Sharing Manager for Multi-Process Neuromorphic Applications

Provides advanced resource sharing mechanisms for neuromorphic hardware
when multiple processes need to access the same resources.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import threading
import time
import logging
from enum import Enum
import uuid
import heapq

from src.core.utils.logging_framework import get_logger
from src.core.hardware.neuromorphic_resource_manager import (
    NeuromorphicResourceManager, ResourceType, ResourceAllocation
)

logger = get_logger("resource_sharing")

class SharingMode(Enum):
    """Resource sharing modes."""
    EXCLUSIVE = "exclusive"  # Resource can only be used by one process at a time
    TIME_SLICED = "time_sliced"  # Resource is time-sliced between processes
    PROPORTIONAL = "proportional"  # Resource is divided proportionally
    PRIORITY_BASED = "priority_based"  # Higher priority processes get resources first


class ResourceRequest:
    """Represents a request for shared resources."""
    
    def __init__(self, 
                 process_id: str,
                 hardware_type: str,
                 resource_type: ResourceType,
                 amount: int,
                 priority: int = 0,
                 sharing_mode: SharingMode = SharingMode.EXCLUSIVE,
                 time_slice_ms: int = 100,
                 min_allocation: Optional[int] = None):
        """
        Initialize a resource request.
        
        Args:
            process_id: ID of the requesting process
            hardware_type: Type of neuromorphic hardware
            resource_type: Type of resource requested
            amount: Amount of resource requested
            priority: Priority level (higher = more important)
            sharing_mode: How the resource should be shared
            time_slice_ms: For TIME_SLICED mode, milliseconds per slice
            min_allocation: Minimum acceptable allocation (for PROPORTIONAL mode)
        """
        self.request_id = str(uuid.uuid4())
        self.process_id = process_id
        self.hardware_type = hardware_type
        self.resource_type = resource_type
        self.amount = amount
        self.priority = priority
        self.sharing_mode = sharing_mode
        self.time_slice_ms = time_slice_ms
        self.min_allocation = min_allocation or amount
        self.timestamp = time.time()
        
    def __lt__(self, other):
        """Compare requests by priority (for priority queue)."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.timestamp < other.timestamp  # Earlier request first


class SharedResourceGroup:
    """Manages a group of shared resources of the same type."""
    
    def __init__(self, hardware_type: str, resource_type: ResourceType):
        """
        Initialize a shared resource group.
        
        Args:
            hardware_type: Type of neuromorphic hardware
            resource_type: Type of resource in this group
        """
        self.hardware_type = hardware_type
        self.resource_type = resource_type
        self.total_allocated = 0
        self.allocation_id = None
        self.sharing_mode = SharingMode.EXCLUSIVE
        self.active_process = None
        self.time_slice_start = 0
        self.process_allocations: Dict[str, int] = {}
        self.waiting_requests: List[ResourceRequest] = []
        self.lock = threading.RLock()
        
    def add_request(self, request: ResourceRequest) -> bool:
        """
        Add a request to this resource group.
        
        Args:
            request: Resource request
            
        Returns:
            bool: True if request was accepted
        """
        with self.lock:
            # If first request, set sharing mode
            if not self.waiting_requests and not self.process_allocations:
                self.sharing_mode = request.sharing_mode
            
            # If incompatible sharing mode, reject
            if request.sharing_mode != self.sharing_mode:
                logger.warning(f"Request {request.request_id} has incompatible sharing mode")
                return False
            
            # Add to waiting requests
            self.waiting_requests.append(request)
            
            # Sort by priority
            self.waiting_requests.sort()
            
            logger.info(f"Added request {request.request_id} to resource group")
            return True
    
    def allocate_resources(self, resource_manager: NeuromorphicResourceManager) -> Dict[str, Any]:
        """
        Allocate resources based on sharing mode.
        
        Args:
            resource_manager: Resource manager to use for allocation
            
        Returns:
            Dict[str, Any]: Allocation results
        """
        with self.lock:
            if not self.waiting_requests and not self.process_allocations:
                return {}
            
            if self.sharing_mode == SharingMode.EXCLUSIVE:
                return self._allocate_exclusive(resource_manager)
            elif self.sharing_mode == SharingMode.TIME_SLICED:
                return self._allocate_time_sliced(resource_manager)
            elif self.sharing_mode == SharingMode.PROPORTIONAL:
                return self._allocate_proportional(resource_manager)
            elif self.sharing_mode == SharingMode.PRIORITY_BASED:
                return self._allocate_priority_based(resource_manager)
            
            return {}
    
    def _allocate_exclusive(self, resource_manager: NeuromorphicResourceManager) -> Dict[str, Any]:
        """Allocate resources in exclusive mode."""
        # If no active process, get highest priority request
        if not self.active_process and self.waiting_requests:
            request = self.waiting_requests[0]
            
            # Try to allocate
            allocation_id = resource_manager.allocate(
                request.process_id,
                request.hardware_type,
                request.resource_type,
                request.amount,
                request.priority
            )
            
            if allocation_id:
                # Success - update state
                self.active_process = request.process_id
                self.allocation_id = allocation_id
                self.total_allocated = request.amount
                self.process_allocations[request.process_id] = request.amount
                self.waiting_requests.pop(0)
                
                logger.info(f"Allocated {request.amount} {request.resource_type} to {request.process_id} (exclusive)")
                return {
                    "process_id": request.process_id,
                    "allocation_id": allocation_id,
                    "amount": request.amount,
                    "sharing_mode": SharingMode.EXCLUSIVE.value
                }
        
        return {}
    
    def _allocate_time_sliced(self, resource_manager: NeuromorphicResourceManager) -> Dict[str, Any]:
        """Allocate resources in time-sliced mode."""
        current_time = time.time() * 1000  # Convert to ms
        
        # If no active process or time slice expired, switch processes
        if (not self.active_process or 
            current_time - self.time_slice_start > self.waiting_requests[0].time_slice_ms):
            
            # Release current allocation if any
            if self.allocation_id:
                resource_manager.release(self.allocation_id)
                self.allocation_id = None
                self.total_allocated = 0
                
                if self.active_process:
                    # Move current process to end of queue
                    for i, req in enumerate(self.waiting_requests):
                        if req.process_id == self.active_process:
                            request = self.waiting_requests.pop(i)
                            self.waiting_requests.append(request)
                            break
            
            # Get next process
            if self.waiting_requests:
                request = self.waiting_requests[0]
                
                # Try to allocate
                allocation_id = resource_manager.allocate(
                    request.process_id,
                    request.hardware_type,
                    request.resource_type,
                    request.amount,
                    request.priority
                )
                
                if allocation_id:
                    # Success - update state
                    self.active_process = request.process_id
                    self.allocation_id = allocation_id
                    self.total_allocated = request.amount
                    self.time_slice_start = current_time
                    self.process_allocations = {request.process_id: request.amount}
                    
                    logger.info(f"Allocated {request.amount} {request.resource_type} to {request.process_id} (time-sliced)")
                    return {
                        "process_id": request.process_id,
                        "allocation_id": allocation_id,
                        "amount": request.amount,
                        "sharing_mode": SharingMode.TIME_SLICED.value,
                        "time_slice_ms": request.time_slice_ms
                    }
        
        return {}
    
    def _allocate_proportional(self, resource_manager: NeuromorphicResourceManager) -> Dict[str, Any]:
        """Allocate resources proportionally among processes."""
        # Release current allocation if any
        if self.allocation_id:
            resource_manager.release(self.allocation_id)
            self.allocation_id = None
            self.total_allocated = 0
        
        # Calculate total requested resources
        total_requested = sum(req.amount for req in self.waiting_requests)
        
        # Get available resources
        pool = resource_manager.get_pool(self.hardware_type)
        if not pool:
            return {}
            
        available = pool.available(self.resource_type)
        
        # If enough resources for everyone, allocate as requested
        if available >= total_requested:
            allocations = {}
            for req in self.waiting_requests:
                allocation_id = resource_manager.allocate(
                    req.process_id,
                    req.hardware_type,
                    req.resource_type,
                    req.amount,
                    req.priority
                )
                
                if allocation_id:
                    allocations[req.process_id] = {
                        "allocation_id": allocation_id,
                        "amount": req.amount
                    }
            
            self.process_allocations = {pid: alloc["amount"] for pid, alloc in allocations.items()}
            self.total_allocated = sum(self.process_allocations.values())
            
            logger.info(f"Allocated resources proportionally to {len(allocations)} processes")
            return {
                "allocations": allocations,
                "sharing_mode": SharingMode.PROPORTIONAL.value
            }
        
        # Not enough for everyone - allocate proportionally
        # Sort by priority first
        self.waiting_requests.sort()
        
        # Calculate proportional allocation
        allocations = {}
        remaining = available
        
        for req in self.waiting_requests:
            # Calculate proportional share
            share = min(req.amount, int(req.amount * available / total_requested))
            
            # Ensure minimum allocation is met
            if share < req.min_allocation:
                share = min(remaining, req.min_allocation)
            
            if share > 0 and share <= remaining:
                allocation_id = resource_manager.allocate(
                    req.process_id,
                    req.hardware_type,
                    req.resource_type,
                    share,
                    req.priority
                )
                
                if allocation_id:
                    allocations[req.process_id] = {
                        "allocation_id": allocation_id,
                        "amount": share
                    }
                    remaining -= share
        
        self.process_allocations = {pid: alloc["amount"] for pid, alloc in allocations.items()}
        self.total_allocated = sum(self.process_allocations.values())
        
        logger.info(f"Allocated resources proportionally to {len(allocations)} processes")
        return {
            "allocations": allocations,
            "sharing_mode": SharingMode.PROPORTIONAL.value
        }
    
    def _allocate_priority_based(self, resource_manager: NeuromorphicResourceManager) -> Dict[str, Any]:
        """Allocate resources based on priority."""
        # Release current allocation if any
        if self.allocation_id:
            resource_manager.release(self.allocation_id)
            self.allocation_id = None
            self.total_allocated = 0
        
        # Sort by priority
        self.waiting_requests.sort()
        
        # Get available resources
        pool = resource_manager.get_pool(self.hardware_type)
        if not pool:
            return {}
            
        available = pool.available(self.resource_type)
        
        # Allocate to highest priority first
        allocations = {}
        remaining = available
        
        for req in self.waiting_requests:
            # Allocate as much as possible
            amount = min(req.amount, remaining)
            
            if amount > 0:
                allocation_id = resource_manager.allocate(
                    req.process_id,
                    req.hardware_type,
                    req.resource_type,
                    amount,
                    req.priority
                )
                
                if allocation_id:
                    allocations[req.process_id] = {
                        "allocation_id": allocation_id,
                        "amount": amount
                    }
                    remaining -= amount
            
            # Stop if no more resources
            if remaining <= 0:
                break
        
        self.process_allocations = {pid: alloc["amount"] for pid, alloc in allocations.items()}
        self.total_allocated = sum(self.process_allocations.values())
        
        logger.info(f"Allocated resources by priority to {len(allocations)} processes")
        return {
            "allocations": allocations,
            "sharing_mode": SharingMode.PRIORITY_BASED.value
        }
    
    def release_process_resources(self, process_id: str, resource_manager: NeuromorphicResourceManager) -> bool:
        """
        Release resources allocated to a process.
        
        Args:
            process_id: Process ID
            resource_manager: Resource manager
            
        Returns:
            bool: True if resources were released
        """
        with self.lock:
            # Remove from waiting requests
            self.waiting_requests = [req for req in self.waiting_requests if req.process_id != process_id]
            
            # If active process, release allocation
            if self.active_process == process_id and self.allocation_id:
                resource_manager.release(self.allocation_id)
                self.allocation_id = None
                self.active_process = None
                self.total_allocated = 0
                
                if process_id in self.process_allocations:
                    del self.process_allocations[process_id]
                
                logger.info(f"Released resources for process {process_id}")
                return True
            
            # For proportional/priority modes, need to reallocate
            if process_id in self.process_allocations:
                del self.process_allocations[process_id]
                
                # Find and release the specific allocation
                process_allocations = resource_manager.get_process_allocations(process_id)
                for alloc in process_allocations:
                    if (alloc.hardware_type == self.hardware_type and 
                        alloc.resource_type == self.resource_type):
                        resource_manager.release(alloc.allocation_id)
                        logger.info(f"Released resources for process {process_id}")
                        return True
            
            return False


class ResourceSharingManager:
    """
    Manager for shared neuromorphic resources.
    
    Provides mechanisms for multiple processes to share neuromorphic
    hardware resources using different sharing strategies.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'ResourceSharingManager':
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """Initialize the resource sharing manager."""
        self.resource_groups: Dict[Tuple[str, ResourceType], SharedResourceGroup] = {}
        self.process_requests: Dict[str, List[str]] = {}  # process_id -> list of request_ids
        self.request_map: Dict[str, ResourceRequest] = {}  # request_id -> request
        self.resource_manager = NeuromorphicResourceManager.get_instance()
        self.allocation_thread = None
        self.running = False
        self.allocation_interval = 0.1  # seconds
    
    def request_resources(self, 
                         process_id: str, 
                         hardware_type: str,
                         resource_type: ResourceType,
                         amount: int,
                         priority: int = 0,
                         sharing_mode: SharingMode = SharingMode.EXCLUSIVE,
                         time_slice_ms: int = 100,
                         min_allocation: Optional[int] = None) -> Optional[str]:
        """
        Request shared resources.
        
        Args:
            process_id: ID of the requesting process
            hardware_type: Type of neuromorphic hardware
            resource_type: Type of resource requested
            amount: Amount of resource requested
            priority: Priority level (higher = more important)
            sharing_mode: How the resource should be shared
            time_slice_ms: For TIME_SLICED mode, milliseconds per slice
            min_allocation: Minimum acceptable allocation (for PROPORTIONAL mode)
            
        Returns:
            Optional[str]: Request ID if successful, None otherwise
        """
        with self._lock:
            # Ensure process is registered
            if not self.resource_manager.register_process(process_id):
                logger.warning(f"Process {process_id} not registered with resource manager")
            
            # Create request
            request = ResourceRequest(
                process_id=process_id,
                hardware_type=hardware_type,
                resource_type=resource_type,
                amount=amount,
                priority=priority,
                sharing_mode=sharing_mode,
                time_slice_ms=time_slice_ms,
                min_allocation=min_allocation
            )
            
            # Get or create resource group
            group_key = (hardware_type, resource_type)
            if group_key not in self.resource_groups:
                self.resource_groups[group_key] = SharedResourceGroup(hardware_type, resource_type)
            
            # Add request to group
            if not self.resource_groups[group_key].add_request(request):
                logger.warning(f"Request {request.request_id} rejected by resource group")
                return None
            
            # Track request
            if process_id not in self.process_requests:
                self.process_requests[process_id] = []
            self.process_requests[process_id].append(request.request_id)
            self.request_map[request.request_id] = request
            
            # Start allocation thread if not running
            self._ensure_allocation_thread()
            
            logger.info(f"Resource request {request.request_id} accepted for {process_id}")
            return request.request_id
    
    def release_resources(self, process_id: str, request_id: Optional[str] = None) -> bool:
        """
        Release resources allocated to a process.
        
        Args:
            process_id: Process ID
            request_id: Optional specific request ID to release
            
        Returns:
            bool: True if resources were released
        """
        with self._lock:
            if process_id not in self.process_requests:
                logger.warning(f"Process {process_id} has no resource requests")
                return False
            
            # If specific request_id provided, release just that request
            if request_id:
                if request_id not in self.request_map:
                    logger.warning(f"Request {request_id} not found")
                    return False
                
                request = self.request_map[request_id]
                group_key = (request.hardware_type, request.resource_type)
                
                if group_key in self.resource_groups:
                    self.resource_groups[group_key].release_process_resources(
                        process_id, self.resource_manager
                    )
                
                # Remove from tracking
                self.process_requests[process_id].remove(request_id)
                del self.request_map[request_id]
                
                logger.info(f"Released resources for request {request_id}")
                return True
            
            # Release all resources for the process
            released = False
            request_ids = self.process_requests[process_id].copy()
            
            for req_id in request_ids:
                request = self.request_map[req_id]
                group_key = (request.hardware_type, request.resource_type)
                
                if group_key in self.resource_groups:
                    if self.resource_groups[group_key].release_process_resources(
                        process_id, self.resource_manager
                    ):
                        released = True
                
                # Remove from tracking
                self.process_requests[process_id].remove(req_id)
                del self.request_map[req_id]
            
            # Clean up if no more requests
            if not self.process_requests[process_id]:
                del self.process_requests[process_id]
            
            logger.info(f"Released all resources for process {process_id}")
            return released
    
    def get_allocation_status(self, process_id: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get allocation status for a process.
        
        Args:
            process_id: Process ID
            request_id: Optional specific request ID
            
        Returns:
            Dict[str, Any]: Allocation status
        """
        with self._lock:
            if process_id not in self.process_requests:
                return {"status": "no_requests"}
            
            # If specific request_id provided, get just that status
            if request_id:
                if request_id not in self.request_map:
                    return {"status": "request_not_found"}
                
                request = self.request_map[request_id]
                group_key = (request.hardware_type, request.resource_type)
                
                if group_key in self.resource_groups:
                    group = self.resource_groups[group_key]
                    
                    # Check if process has an allocation
                    if process_id in group.process_allocations:
                        return {
                            "status": "allocated",
                            "amount": group.process_allocations[process_id],
                            "sharing_mode": group.sharing_mode.value,
                            "resource_type": request.resource_type.name,
                            "hardware_type": request.hardware_type
                        }
                    
                    # Check if process is waiting
                    for i, req in enumerate(group.waiting_requests):
                        if req.request_id == request_id:
                            return {
                                "status": "waiting",
                                "position": i,
                                "sharing_mode": group.sharing_mode.value,
                                "resource_type": request.resource_type.name,
                                "hardware_type": request.hardware_type
                            }
                
                return {"status": "unknown"}
            
            # Get status for all requests
            statuses = {}
            for req_id in self.process_requests[process_id]:
                statuses[req_id] = self.get_allocation_status(process_id, req_id)
            
            return {
                "status": "multiple_requests",
                "requests": statuses
            }
    
    def _ensure_allocation_thread(self):
        """Ensure the allocation thread is running."""
        if self.allocation_thread is None or not self.allocation_thread.is_alive():
            self.running = True
            self.allocation_thread = threading.Thread(
                target=self._allocation_loop,
                daemon=True
            )
            self.allocation_thread.start()
    
    def _allocation_loop(self):
        """Main allocation loop."""
        while self.running:
            try:
                with self._lock:
                    # Process each resource group
                    for group_key, group in self.resource_groups.items():
                        group.allocate_resources(self.resource_manager)
            
            except Exception as e:
                logger.error(f"Error in allocation loop: {str(e)}")
            
            time.sleep(self.allocation_interval)
    
    def stop(self):
        """Stop the allocation thread."""
        self.running = False
        if self.allocation_thread:
            self.allocation_thread.join(timeout=2.0)
            self.allocation_thread = None
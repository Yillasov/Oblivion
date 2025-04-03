#!/usr/bin/env python3
"""
Dynamic Resource Allocation

Provides adaptive resource allocation capabilities for neuromorphic hardware.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import threading
import time
import logging
from enum import Enum

from src.core.utils.logging_framework import get_logger
from src.core.hardware.exceptions import HardwareAllocationError

logger = get_logger("dynamic_allocation")

class AllocationPriority(Enum):
    """Priority levels for resource allocation."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class DynamicResourceAllocator:
    """Provides dynamic resource allocation capabilities."""
    
    def __init__(self, resource_manager):
        """
        Initialize the dynamic allocator.
        
        Args:
            resource_manager: The resource manager to use for allocation
        """
        self.resource_manager = resource_manager
        self.workloads = {}
        self.allocation_map = {}
        self.lock = threading.RLock()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.reallocation_threshold = 0.15  # 15% change triggers reallocation
        
    def register_workload(self, workload_id: str, initial_resources: Dict[str, int], 
                          priority: AllocationPriority = AllocationPriority.MEDIUM) -> bool:
        """
        Register a workload for dynamic allocation.
        
        Args:
            workload_id: Unique identifier for the workload
            initial_resources: Initial resource requirements
            priority: Allocation priority
            
        Returns:
            bool: Success status
        """
        with self.lock:
            if workload_id in self.workloads:
                logger.warning(f"Workload {workload_id} already registered")
                return False
                
            self.workloads[workload_id] = {
                "resources": initial_resources,
                "priority": priority,
                "last_usage": {},
                "trend": {},
                "allocated": {}
            }
            
            # Perform initial allocation
            try:
                allocation_result = self._allocate_for_workload(workload_id)
                self.workloads[workload_id]["allocated"] = allocation_result
                return True
            except HardwareAllocationError as e:
                logger.error(f"Failed to allocate resources for workload {workload_id}: {str(e)}")
                return False
    
    def update_workload_demand(self, workload_id: str, new_demand: Dict[str, int]) -> bool:
        """
        Update resource demand for a workload.
        
        Args:
            workload_id: Workload identifier
            new_demand: New resource requirements
            
        Returns:
            bool: Success status
        """
        with self.lock:
            if workload_id not in self.workloads:
                logger.warning(f"Workload {workload_id} not registered")
                return False
                
            # Calculate change percentage
            current = self.workloads[workload_id]["resources"]
            change_percentage = self._calculate_change_percentage(current, new_demand)
            
            # Update demand
            self.workloads[workload_id]["resources"] = new_demand
            
            # Reallocate if change exceeds threshold
            if change_percentage > self.reallocation_threshold:
                logger.info(f"Significant demand change ({change_percentage:.2%}) for {workload_id}, reallocating")
                return self._reallocate_resources()
            
            return True
    
    def start_monitoring(self, interval: float = 5.0) -> bool:
        """
        Start monitoring and dynamic reallocation.
        
        Args:
            interval: Monitoring interval in seconds
            
        Returns:
            bool: Success status
        """
        if self.monitoring_thread is not None:
            logger.warning("Monitoring already active")
            return False
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started dynamic allocation monitoring with interval {interval}s")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop monitoring and dynamic reallocation.
        
        Returns:
            bool: Success status
        """
        if self.monitoring_thread is None:
            logger.warning("Monitoring not active")
            return False
            
        self.monitoring_active = False
        self.monitoring_thread.join(timeout=2.0)
        self.monitoring_thread = None
        logger.info("Stopped dynamic allocation monitoring")
        return True
    
    def _monitoring_loop(self, interval: float) -> None:
        """
        Resource monitoring and reallocation loop.
        
        Args:
            interval: Monitoring interval in seconds
        """
        while self.monitoring_active:
            try:
                # Collect usage data
                self._collect_usage_data()
                
                # Analyze trends and predict future needs
                self._analyze_trends()
                
                # Reallocate resources if needed
                self._reallocate_resources()
                
            except Exception as e:
                logger.error(f"Error in dynamic allocation monitoring: {str(e)}")
                
            time.sleep(interval)
    
    def _collect_usage_data(self) -> None:
        """Collect resource usage data for all workloads."""
        with self.lock:
            for workload_id, workload in self.workloads.items():
                # In a real implementation, this would collect actual usage metrics
                # For now, we'll use a simplified approach
                allocated = workload["allocated"]
                workload["last_usage"] = {
                    resource: self._simulate_usage(allocated.get(resource, 0))
                    for resource in allocated
                }
    
    def _simulate_usage(self, allocated: int) -> int:
        """Simulate resource usage for demonstration purposes."""
        import random
        # Simulate 60-95% usage of allocated resources
        return int(allocated * (0.6 + random.random() * 0.35))
    
    def _analyze_trends(self) -> None:
        """Analyze usage trends and predict future needs."""
        with self.lock:
            for workload_id, workload in self.workloads.items():
                usage = workload["last_usage"]
                allocated = workload["allocated"]
                
                # Calculate utilization percentage for each resource
                for resource, used in usage.items():
                    if resource in allocated and allocated[resource] > 0:
                        utilization = used / allocated[resource]
                        
                        # Update trend based on utilization
                        if utilization > 0.9:  # High utilization
                            workload["trend"][resource] = 1.2  # Suggest 20% increase
                        elif utilization < 0.5:  # Low utilization
                            workload["trend"][resource] = 0.8  # Suggest 20% decrease
                        else:
                            workload["trend"][resource] = 1.0  # No change
    
    def _reallocate_resources(self) -> bool:
        """
        Reallocate resources based on current demands and trends.
        
        Returns:
            bool: Success status
        """
        with self.lock:
            try:
                # Calculate new resource allocations based on trends
                new_allocations = {}
                
                for workload_id, workload in self.workloads.items():
                    resources = workload["resources"]
                    trends = workload.get("trend", {})
                    
                    # Apply trends to adjust resource requests
                    adjusted_resources = {
                        resource: int(amount * trends.get(resource, 1.0))
                        for resource, amount in resources.items()
                    }
                    
                    new_allocations[workload_id] = adjusted_resources
                
                # Sort workloads by priority for allocation
                sorted_workloads = sorted(
                    self.workloads.items(),
                    key=lambda x: x[1]["priority"].value,
                    reverse=True
                )
                
                # Allocate resources in priority order
                for workload_id, _ in sorted_workloads:
                    self._allocate_for_workload(
                        workload_id, 
                        new_allocations.get(workload_id, {})
                    )
                
                return True
                
            except Exception as e:
                logger.error(f"Error during resource reallocation: {str(e)}")
                return False
    
    def _allocate_for_workload(self, workload_id: str, resources: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Allocate resources for a specific workload.
        
        Args:
            workload_id: Workload identifier
            resources: Resource requirements (uses stored requirements if None)
            
        Returns:
            Dict[str, Any]: Allocation result
        """
        if resources is None:
            resources = self.workloads[workload_id]["resources"]
        
        result = {}
        
        # Release existing allocations if any
        self._release_workload_resources(workload_id)
        
        # Allocate neurons if needed
        if resources is not None and "neurons" in resources and resources["neurons"] > 0:
            neuron_result = self.resource_manager.allocate_neurons(
                resources["neurons"],
                {"workload_id": workload_id}
            )
            result["neurons"] = neuron_result.get("allocated_neurons", 0)
        
        # Store allocation result
        self.workloads[workload_id]["allocated"] = result
        return result
    
    def _release_workload_resources(self, workload_id: str) -> None:
        """
        Release resources allocated to a workload.
        
        Args:
            workload_id: Workload identifier
        """
        # In a real implementation, this would release specific resources
        # For now, we'll just reset the allocation in our tracking
        if workload_id in self.workloads:
            self.workloads[workload_id]["allocated"] = {}
    
    def _calculate_change_percentage(self, current: Dict[str, int], new: Dict[str, int]) -> float:
        """
        Calculate percentage change between resource allocations.
        
        Args:
            current: Current resource allocation
            new: New resource allocation
            
        Returns:
            float: Change percentage (0.0-1.0)
        """
        # Calculate total resources in each allocation
        total_current = sum(current.values())
        total_new = sum(new.values())
        
        if total_current == 0:
            return 1.0 if total_new > 0 else 0.0
            
        return abs(total_new - total_current) / total_current
"""
Resource management for power supply systems.

This module provides classes and utilities for managing power resources,
including distribution, allocation, monitoring, and optimization.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
import time
import threading
import numpy as np

from src.core.hardware.resource_manager import ResourceManager
from src.core.hardware.resource_sharing import get_resource_manager
from src.power.base import PowerSupplyType, PowerSupplySpecs, NeuromorphicPowerSupply
from src.core.utils.logging_framework import get_logger

logger = get_logger("power_resource_management")


class PowerPriority(Enum):
    """Priority levels for power systems."""
    CRITICAL = 3
    HIGH = 2
    MEDIUM = 1
    LOW = 0


@dataclass
class PowerResource:
    """Represents a power resource in the system."""
    id: str
    type: PowerSupplyType
    max_output: float  # Maximum power output in kW
    current_output: float = 0.0  # Current power output in kW
    priority: PowerPriority = PowerPriority.MEDIUM
    efficiency: float = 0.9
    health: float = 1.0  # Health factor (0-1)
    temperature: float = 20.0  # Temperature in Celsius
    allocation_percentage: float = 100.0  # Percentage of power allocated
    consumers: Set[str] = field(default_factory=set)  # Systems consuming power


class PowerResourceManager:
    """Manages power resources across the UCAV platform."""
    
    def __init__(self, hardware_resource_manager: Optional[ResourceManager] = None):
        """
        Initialize the power resource manager.
        
        Args:
            hardware_resource_manager: Hardware resource manager
        """
        self.hardware_rm = hardware_resource_manager or get_resource_manager()
        self.power_resources: Dict[str, PowerResource] = {}
        self.power_consumers: Dict[str, Dict[str, float]] = {}  # Consumer ID -> {resource_id: requirement}
        self.allocation_map: Dict[str, Dict[str, float]] = {}  # Consumer ID -> {resource_id: allocation}
        self.total_available_power = 0.0
        self.total_allocated_power = 0.0
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 1.0  # seconds
        self.power_history: List[Dict[str, Any]] = []
        self.initialized = False
        
        # Power distribution strategy
        self.distribution_strategy = "priority_based"  # Options: "priority_based", "proportional", "efficiency_based"
        
        logger.info("Power resource manager initialized")
    
    def register_power_resource(self, 
                               power_system: NeuromorphicPowerSupply, 
                               resource_id: str,
                               priority: PowerPriority = PowerPriority.MEDIUM) -> bool:
        """
        Register a power system as a resource.
        
        Args:
            power_system: Power supply system
            resource_id: Unique identifier for the resource
            priority: Priority level
            
        Returns:
            Success status
        """
        if resource_id in self.power_resources:
            logger.warning(f"Power resource '{resource_id}' already registered")
            return False
        
        try:
            specs = power_system.get_specifications()
            status = power_system.get_status()
            
            # Create power resource
            resource = PowerResource(
                id=resource_id,
                type=getattr(power_system, 'type', PowerSupplyType.SOLID_STATE_BATTERY),
                max_output=specs.power_output,
                current_output=status.get("current_output", 0.0),
                priority=priority,
                efficiency=specs.efficiency,
                health=status.get("health", 1.0),
                temperature=status.get("temperature", 20.0)
            )
            
            self.power_resources[resource_id] = resource
            self.total_available_power += specs.power_output
            
            logger.info(f"Registered power resource '{resource_id}' with {specs.power_output} kW capacity")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register power resource: {str(e)}")
            return False
    
    def register_power_consumer(self, 
                               consumer_id: str, 
                               power_requirements: Dict[str, float]) -> bool:
        """
        Register a system as a power consumer.
        
        Args:
            consumer_id: Consumer identifier
            power_requirements: Power requirements in kW by resource ID
            
        Returns:
            Success status
        """
        if consumer_id in self.power_consumers:
            logger.warning(f"Power consumer '{consumer_id}' already registered")
            return False
        
        # Validate requirements
        for resource_id, requirement in power_requirements.items():
            if resource_id not in self.power_resources:
                logger.warning(f"Unknown power resource '{resource_id}'")
                return False
        
        self.power_consumers[consumer_id] = power_requirements
        self.allocation_map[consumer_id] = {resource_id: 0.0 for resource_id in power_requirements}
        
        logger.info(f"Registered power consumer '{consumer_id}'")
        return True
    
    def allocate_power(self) -> Dict[str, Dict[str, float]]:
        """
        Allocate power to consumers based on current strategy.
        
        Returns:
            Allocation map (consumer_id -> {resource_id: allocation})
        """
        if self.distribution_strategy == "priority_based":
            return self._allocate_power_priority_based()
        elif self.distribution_strategy == "proportional":
            return self._allocate_power_proportional()
        elif self.distribution_strategy == "efficiency_based":
            return self._allocate_power_efficiency_based()
        else:
            logger.warning(f"Unknown distribution strategy: {self.distribution_strategy}")
            return self._allocate_power_priority_based()
    
    def _allocate_power_priority_based(self) -> Dict[str, Dict[str, float]]:
        """Allocate power based on consumer and resource priorities."""
        # Reset allocations
        for consumer_id in self.allocation_map:
            for resource_id in self.allocation_map[consumer_id]:
                self.allocation_map[consumer_id][resource_id] = 0.0
        
        # Sort consumers by priority (assuming consumers have priority attributes)
        # For simplicity, we'll use a fixed priority order for now
        consumer_priorities = {
            consumer_id: PowerPriority.CRITICAL if "propulsion" in consumer_id else
                         PowerPriority.HIGH if "navigation" in consumer_id else
                         PowerPriority.MEDIUM
            for consumer_id in self.power_consumers
        }
        
        # Sort consumers by priority (highest first)
        sorted_consumers = sorted(
            self.power_consumers.keys(),
            key=lambda cid: consumer_priorities.get(cid, PowerPriority.MEDIUM).value,
            reverse=True
        )
        
        # Track remaining capacity for each resource
        remaining_capacity = {
            resource_id: resource.max_output
            for resource_id, resource in self.power_resources.items()
        }
        
        # Allocate power to consumers in priority order
        for consumer_id in sorted_consumers:
            requirements = self.power_consumers[consumer_id]
            
            for resource_id, requirement in requirements.items():
                if resource_id in remaining_capacity:
                    # Allocate up to the requirement or remaining capacity
                    allocation = min(requirement, remaining_capacity[resource_id])
                    self.allocation_map[consumer_id][resource_id] = allocation
                    remaining_capacity[resource_id] -= allocation
                    
                    # Update resource consumers set
                    if resource_id in self.power_resources and allocation > 0:
                        self.power_resources[resource_id].consumers.add(consumer_id)
        
        # Update resource allocation percentages
        for resource_id, resource in self.power_resources.items():
            used_capacity = resource.max_output - remaining_capacity.get(resource_id, 0)
            if resource.max_output > 0:
                resource.allocation_percentage = (used_capacity / resource.max_output) * 100
        
        # Calculate total allocated power
        self.total_allocated_power = sum(
            sum(allocations.values())
            for allocations in self.allocation_map.values()
        )
        
        logger.info(f"Allocated {self.total_allocated_power:.2f} kW of power using priority-based strategy")
        return self.allocation_map
    
    def _allocate_power_proportional(self) -> Dict[str, Dict[str, float]]:
        """Allocate power proportionally to all consumers."""
        # Implementation similar to priority-based but with proportional distribution
        # For brevity, this is a simplified version
        
        # Calculate total requirements per resource
        total_requirements = {}
        for consumer_id, requirements in self.power_consumers.items():
            for resource_id, requirement in requirements.items():
                if resource_id not in total_requirements:
                    total_requirements[resource_id] = 0
                total_requirements[resource_id] += requirement
        
        # Calculate allocation factors
        allocation_factors = {}
        for resource_id, total_req in total_requirements.items():
            if resource_id in self.power_resources:
                resource = self.power_resources[resource_id]
                if total_req > 0 and total_req > resource.max_output:
                    allocation_factors[resource_id] = resource.max_output / total_req
                else:
                    allocation_factors[resource_id] = 1.0
        
        # Apply allocation factors
        for consumer_id, requirements in self.power_consumers.items():
            for resource_id, requirement in requirements.items():
                if resource_id in allocation_factors:
                    factor = allocation_factors[resource_id]
                    self.allocation_map[consumer_id][resource_id] = requirement * factor
                    
                    # Update resource consumers set
                    if resource_id in self.power_resources and requirement * factor > 0:
                        self.power_resources[resource_id].consumers.add(consumer_id)
        
        # Update resource allocation percentages
        for resource_id, resource in self.power_resources.items():
            if resource_id in total_requirements and resource.max_output > 0:
                allocated = min(total_requirements[resource_id], resource.max_output)
                resource.allocation_percentage = (allocated / resource.max_output) * 100
        
        # Calculate total allocated power
        self.total_allocated_power = sum(
            sum(allocations.values())
            for allocations in self.allocation_map.values()
        )
        
        logger.info(f"Allocated {self.total_allocated_power:.2f} kW of power using proportional strategy")
        return self.allocation_map
    
    def _allocate_power_efficiency_based(self) -> Dict[str, Dict[str, float]]:
        """Allocate power based on resource efficiency."""
        # Simplified implementation for brevity
        logger.info("Using efficiency-based power allocation")
        return self._allocate_power_priority_based()
    
    def start_monitoring(self) -> bool:
        """
        Start monitoring power resources.
        
        Returns:
            Success status
        """
        if self.monitoring_active:
            logger.warning("Power monitoring already active")
            return False
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Started power resource monitoring")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop monitoring power resources.
        
        Returns:
            Success status
        """
        if not self.monitoring_active:
            logger.warning("Power monitoring not active")
            return False
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            self.monitoring_thread = None
        
        logger.info("Stopped power resource monitoring")
        return True
    
    def _monitoring_loop(self) -> None:
        """Monitoring loop for power resources."""
        while self.monitoring_active:
            try:
                # Update resource status
                self._update_resource_status()
                
                # Record power usage
                self._record_power_usage()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in power monitoring: {str(e)}")
                time.sleep(1.0)  # Sleep on error to prevent tight loop
    
    def _update_resource_status(self) -> None:
        """Update status of power resources."""
        # In a real system, this would query the actual power systems
        # For simulation, we'll update based on allocation
        for resource_id, resource in self.power_resources.items():
            # Calculate current output based on allocations
            current_output = sum(
                allocations.get(resource_id, 0)
                for allocations in self.allocation_map.values()
            )
            
            # Update resource
            resource.current_output = current_output
            
            # Simulate temperature based on load
            load_factor = current_output / resource.max_output if resource.max_output > 0 else 0
            resource.temperature = 20.0 + (40.0 * load_factor)  # 20°C idle, up to 60°C at full load
    
    def _record_power_usage(self) -> None:
        """Record power usage for analysis."""
        timestamp = time.time()
        
        # Create snapshot of current power usage
        snapshot = {
            "timestamp": timestamp,
            "total_available": self.total_available_power,
            "total_allocated": self.total_allocated_power,
            "resources": {
                resource_id: {
                    "current_output": resource.current_output,
                    "max_output": resource.max_output,
                    "temperature": resource.temperature,
                    "allocation_percentage": resource.allocation_percentage,
                    "efficiency": resource.efficiency,
                    "health": resource.health
                }
                for resource_id, resource in self.power_resources.items()
            },
            "consumers": {
                consumer_id: {
                    "total_allocation": sum(allocations.values()),
                    "allocations": allocations
                }
                for consumer_id, allocations in self.allocation_map.items()
            }
        }
        
        # Add to history
        self.power_history.append(snapshot)
        
        # Limit history size
        if len(self.power_history) > 1000:
            self.power_history = self.power_history[-1000:]
    
    def optimize_power_usage(self) -> Dict[str, Any]:
        """
        Optimize power usage across the system.
        
        Returns:
            Optimization results
        """
        # Implement power optimization logic
        try:
            # Get current power status
            current_status = {
                "allocation_percentage": (self.total_allocated_power / self.total_available_power * 100) if self.total_available_power > 0 else 0,
                "consumers": {
                    consumer_id: {
                        "total_allocation": sum(allocations.values()),
                        "allocations": allocations
                    }
                    for consumer_id, allocations in self.allocation_map.items()
                }
            }
            
            # Check if optimization is needed
            if current_status["allocation_percentage"] < 90:
                logger.info("Power usage already optimal")
                return {"optimized": False, "reason": "usage_below_threshold"}
            
            # Identify high-consumption consumers
            high_consumers = {}
            for consumer_id, data in current_status["consumers"].items():
                if data["total_allocation"] > 0.1 * self.total_available_power:
                    high_consumers[consumer_id] = data["total_allocation"]
            
            # Apply optimization strategy
            optimizations = {}
            
            # 1. Redistribute from low-priority to high-priority consumers
            if self.distribution_strategy == "priority_based":
                # Re-run allocation with stricter priority enforcement
                original_allocation = self.total_allocated_power
                self._allocate_power_priority_based()
                new_allocation = self.total_allocated_power
                
                optimizations["priority_redistribution"] = {
                    "original": original_allocation,
                    "optimized": new_allocation,
                    "savings": original_allocation - new_allocation
                }
            
            # 2. Identify and use more efficient resources
            if len(self.power_resources) > 1:
                # Sort resources by efficiency
                sorted_resources = sorted(
                    self.power_resources.items(),
                    key=lambda x: x[1].efficiency,
                    reverse=True
                )
                
                # Try to shift load to more efficient resources
                optimizations["efficiency_shift"] = {
                    "from_resources": [],
                    "to_resources": []
                }
                
                # Implementation simplified for brevity
            
            logger.info(f"Power optimization completed with {len(optimizations)} strategies")
            return {
                "optimized": True,
                "strategies": optimizations,
                "status": {
                    "total_available": self.total_available_power,
                    "total_allocated": self.total_allocated_power,
                    "resources": {
                        rid: {
                            "current_output": r.current_output,
                            "max_output": r.max_output,
                            "efficiency": r.efficiency,
                            "health": r.health
                        } for rid, r in self.power_resources.items()
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error optimizing power usage: {str(e)}")
            return {"optimized": False, "error": str(e)}
    
    def predict_power_needs(self, time_horizon: float = 60.0) -> Dict[str, Any]:
        """
        Predict power needs for the specified time horizon.
        
        Args:
            time_horizon: Time horizon in seconds
            
        Returns:
            Prediction results
        """
        if len(self.power_history) < 10:
            return {"predicted": False, "reason": "insufficient_history"}
        
        try:
            # Extract recent history
            recent_history = self.power_history[-min(100, len(self.power_history)):]
            
            # Calculate trends
            total_allocated = [entry["total_allocated"] for entry in recent_history]
            
            # Simple linear trend
            if len(total_allocated) > 1:
                slope = (total_allocated[-1] - total_allocated[0]) / len(total_allocated)
                predicted_allocation = total_allocated[-1] + slope * (time_horizon / self.monitoring_interval)
            else:
                predicted_allocation = total_allocated[-1]
            
            # Resource-specific predictions
            resource_predictions = {}
            for resource_id in self.power_resources:
                resource_outputs = [
                    entry["resources"].get(resource_id, {}).get("current_output", 0)
                    for entry in recent_history if "resources" in entry
                ]
                
                if resource_outputs:
                    if len(resource_outputs) > 1:
                        resource_slope = (resource_outputs[-1] - resource_outputs[0]) / len(resource_outputs)
                        predicted_output = resource_outputs[-1] + resource_slope * (time_horizon / self.monitoring_interval)
                    else:
                        predicted_output = resource_outputs[-1]
                    
                    resource_predictions[resource_id] = {
                        "current": resource_outputs[-1] if resource_outputs else 0,
                        "predicted": predicted_output,
                        "change": predicted_output - (resource_outputs[-1] if resource_outputs else 0)
                    }
            
            return {
                "predicted": True,
                "current_allocation": total_allocated[-1] if total_allocated else 0,
                "predicted_allocation": predicted_allocation,
                "change_percentage": ((predicted_allocation / total_allocated[-1]) - 1) * 100 if total_allocated and total_allocated[-1] > 0 else 0,
                "resources": resource_predictions,
                "time_horizon": time_horizon
            }
            
        except Exception as e:
            logger.error(f"Error predicting power needs: {str(e)}")
            return {"predicted": False, "error": str(e)}
    
    def get_power_history(self, 
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         resource_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get power usage history.
        
        Args:
            start_time: Start time (timestamp)
            end_time: End time (timestamp)
            resource_ids: List of resource IDs to include
            
        Returns:
            Filtered power history
        """
        if not self.power_history:
            return []
        
        # Apply time filters
        filtered_history = self.power_history
        
        if start_time is not None:
            filtered_history = [
                entry for entry in filtered_history
                if entry["timestamp"] >= start_time
            ]
        
        if end_time is not None:
            filtered_history = [
                entry for entry in filtered_history
                if entry["timestamp"] <= end_time
            ]
        
        # Apply resource filters
        if resource_ids:
            filtered_history = [
                {
                    **entry,
                    "resources": {
                        resource_id: data
                        for resource_id, data in entry.get("resources", {}).items()
                        if resource_id in resource_ids
                    }
                }
                for entry in filtered_history
            ]
        
        return filtered_history
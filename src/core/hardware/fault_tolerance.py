#!/usr/bin/env python3
"""
Fault Tolerance Manager for Neuromorphic Hardware

Provides fault detection, redundancy management, and graceful degradation
for critical neuromorphic components.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import threading
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass
import logging

from src.core.utils.logging_framework import get_logger

logger = get_logger("fault_tolerance")

class FaultType(Enum):
    """Types of faults that can occur in neuromorphic hardware."""
    COMMUNICATION = "communication"
    COMPUTATION = "computation"
    MEMORY = "memory"
    POWER = "power"
    THERMAL = "thermal"
    TIMING = "timing"
    UNKNOWN = "unknown"

class FaultSeverity(Enum):
    """Severity levels for faults."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FaultEvent:
    """Represents a fault event in the system."""
    component_id: str
    fault_type: FaultType
    severity: FaultSeverity
    timestamp: float
    details: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[float] = None
    resolution_strategy: Optional[str] = None

class RedundancyStrategy(Enum):
    """Strategies for component redundancy."""
    NONE = "none"
    ACTIVE = "active"  # All redundant components active simultaneously
    PASSIVE = "passive"  # Redundant components activated on failure
    VOTING = "voting"  # Multiple components active, voting on results

class FaultToleranceManager:
    """
    Manages fault tolerance for neuromorphic components.
    
    Provides:
    - Fault detection and logging
    - Component redundancy management
    - Graceful degradation strategies
    - Health monitoring
    """
    
    def __init__(self):
        """Initialize the fault tolerance manager."""
        self.components: Dict[str, Dict[str, Any]] = {}
        self.redundancy_groups: Dict[str, List[str]] = {}
        self.active_faults: Dict[str, FaultEvent] = {}
        self.fault_history: List[FaultEvent] = []
        self.health_monitors: Dict[str, Callable] = {}
        self.degradation_strategies: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.critical_components: Set[str] = set()
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds
        
        logger.info("Fault tolerance manager initialized")
    
    def register_component(self, 
                          component_id: str, 
                          component: Any, 
                          is_critical: bool = False,
                          redundancy_group: Optional[str] = None,
                          redundancy_strategy: RedundancyStrategy = RedundancyStrategy.PASSIVE) -> bool:
        """
        Register a component for fault tolerance management.
        
        Args:
            component_id: Unique identifier for the component
            component: The component object
            is_critical: Whether this is a critical component
            redundancy_group: Optional group name for redundant components
            redundancy_strategy: Strategy for redundancy management
            
        Returns:
            bool: Success status
        """
        if component_id in self.components:
            logger.warning(f"Component {component_id} already registered")
            return False
        
        self.components[component_id] = {
            "component": component,
            "is_critical": is_critical,
            "redundancy_group": redundancy_group,
            "redundancy_strategy": redundancy_strategy,
            "active": True,
            "health": 1.0,
            "last_checked": time.time()
        }
        
        if is_critical:
            self.critical_components.add(component_id)
        
        if redundancy_group:
            if redundancy_group not in self.redundancy_groups:
                self.redundancy_groups[redundancy_group] = []
            self.redundancy_groups[redundancy_group].append(component_id)
        
        logger.info(f"Registered component {component_id} for fault tolerance")
        return True
    
    def register_health_monitor(self, component_id: str, monitor_func: Callable) -> bool:
        """
        Register a health monitoring function for a component.
        
        Args:
            component_id: Component identifier
            monitor_func: Function that returns health status (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        if component_id not in self.components:
            logger.warning(f"Cannot register health monitor: Component {component_id} not found")
            return False
        
        self.health_monitors[component_id] = monitor_func
        logger.info(f"Registered health monitor for component {component_id}")
        return True
    
    def register_degradation_strategy(self, component_id: str, strategy_func: Callable) -> bool:
        """
        Register a degradation strategy for a component.
        
        Args:
            component_id: Component identifier
            strategy_func: Function that implements degradation
            
        Returns:
            bool: Success status
        """
        if component_id not in self.components:
            logger.warning(f"Cannot register degradation strategy: Component {component_id} not found")
            return False
        
        self.degradation_strategies[component_id] = strategy_func
        logger.info(f"Registered degradation strategy for component {component_id}")
        return True
    
    def register_recovery_strategy(self, component_id: str, strategy_func: Callable) -> bool:
        """
        Register a recovery strategy for a component.
        
        Args:
            component_id: Component identifier
            strategy_func: Function that implements recovery
            
        Returns:
            bool: Success status
        """
        if component_id not in self.components:
            logger.warning(f"Cannot register recovery strategy: Component {component_id} not found")
            return False
        
        self.recovery_strategies[component_id] = strategy_func
        logger.info(f"Registered recovery strategy for component {component_id}")
        return True
    
    def report_fault(self, 
                    component_id: str, 
                    fault_type: FaultType, 
                    severity: FaultSeverity,
                    details: Dict[str, Any] = {}) -> Optional[str]:
        """
        Report a fault in a component.
        
        Args:
            component_id: Component identifier
            fault_type: Type of fault
            severity: Severity level
            details: Additional details about the fault
            
        Returns:
            Optional[str]: Fault ID if successfully reported
        """
        if component_id not in self.components:
            logger.warning(f"Cannot report fault: Component {component_id} not found")
            return None
        
        # Create fault event
        fault_id = f"{component_id}_{int(time.time())}"
        fault_event = FaultEvent(
            component_id=component_id,
            fault_type=fault_type,
            severity=severity,
            timestamp=time.time(),
            details=details or {}
        )
        
        # Store fault
        self.active_faults[fault_id] = fault_event
        self.fault_history.append(fault_event)
        
        # Log fault
        logger.warning(f"Fault reported: {component_id} - {fault_type.value} - {severity.value}")
        
        # Handle fault based on severity
        if severity in [FaultSeverity.HIGH, FaultSeverity.CRITICAL]:
            self._handle_critical_fault(fault_id, fault_event)
        else:
            self._handle_non_critical_fault(fault_id, fault_event)
        
        return fault_id
    
    def _handle_critical_fault(self, fault_id: str, fault_event: FaultEvent):
        """Handle a critical fault."""
        component_id = fault_event.component_id
        
        # Verify component exists before proceeding
        if component_id not in self.components:
            logger.error(f"Cannot handle fault for unknown component: {component_id}")
            return
            
        component_info = self.components[component_id]
        
        # Check if component is in a redundancy group
        redundancy_group = component_info.get("redundancy_group")
        if redundancy_group and redundancy_group in self.redundancy_groups:
            # Activate redundant component
            success = self._activate_redundant_component(component_id, redundancy_group)
            if not success:
                logger.warning(f"Failed to activate redundant component for {component_id}")
        
        # Apply degradation strategy if available
        if component_id in self.degradation_strategies:
            try:
                self.degradation_strategies[component_id](component_info["component"], fault_event)
                logger.info(f"Applied degradation strategy for {component_id}")
            except Exception as e:
                logger.error(f"Error applying degradation strategy for {component_id}: {str(e)}")
        
        # Mark component as inactive if critical fault
        if fault_event.severity == FaultSeverity.CRITICAL:
            component_info["active"] = False
            # Notify system about critical component failure
            self._notify_critical_component_failure(component_id)
            logger.warning(f"Component {component_id} marked as inactive due to critical fault")
    
    def _notify_critical_component_failure(self, component_id: str) -> None:
        """Notify system about critical component failure."""
        if component_id in self.critical_components:
            logger.critical(f"Critical component failure: {component_id}")
            # Implement notification mechanism (e.g., event system)
    
    def _handle_non_critical_fault(self, fault_id: str, fault_event: FaultEvent):
        """Handle a non-critical fault."""
        component_id = fault_event.component_id
        component_info = self.components[component_id]
        
        # Update health status
        if fault_event.severity == FaultSeverity.MEDIUM:
            component_info["health"] = max(0.5, component_info["health"] - 0.2)
        else:  # LOW severity
            component_info["health"] = max(0.8, component_info["health"] - 0.1)
        
        logger.info(f"Component {component_id} health updated to {component_info['health']}")
    
    def _activate_redundant_component(self, failed_component_id: str, redundancy_group: str):
        """Activate a redundant component to replace a failed one."""
        # Find available redundant components
        available_components = [
            cid for cid in self.redundancy_groups[redundancy_group]
            if cid != failed_component_id and self.components[cid]["active"]
        ]
        
        if not available_components:
            logger.error(f"No available redundant components in group {redundancy_group}")
            return
        
        # Select the first available component
        replacement_id = available_components[0]
        
        # Activate the replacement
        logger.info(f"Activating redundant component {replacement_id} to replace {failed_component_id}")
        
        # In a real implementation, we would transfer state and activate the component
        # For now, we just log the action
    
    def resolve_fault(self, fault_id: str, resolution_strategy: str) -> bool:
        """
        Mark a fault as resolved.
        
        Args:
            fault_id: Fault identifier
            resolution_strategy: Description of how the fault was resolved
            
        Returns:
            bool: Success status
        """
        if fault_id not in self.active_faults:
            logger.warning(f"Cannot resolve fault: Fault {fault_id} not found")
            return False
        
        fault_event = self.active_faults[fault_id]
        fault_event.resolved = True
        fault_event.resolution_time = time.time()
        fault_event.resolution_strategy = resolution_strategy
        
        # Remove from active faults
        del self.active_faults[fault_id]
        
        # Apply recovery strategy if available
        component_id = fault_event.component_id
        if component_id in self.recovery_strategies:
            try:
                component_info = self.components[component_id]
                self.recovery_strategies[component_id](component_info["component"], fault_event)
                logger.info(f"Applied recovery strategy for {component_id}")
                
                # Reset health status
                component_info["health"] = 1.0
                component_info["active"] = True
            except Exception as e:
                logger.error(f"Error applying recovery strategy for {component_id}: {str(e)}")
        
        logger.info(f"Fault {fault_id} resolved: {resolution_strategy}")
        return True
    
    def start_monitoring(self):
        """Start the health monitoring thread."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop the health monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_component_health()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(1.0)  # Sleep on error to avoid tight loop
    
    def _check_component_health(self):
        """Check health of all registered components."""
        current_time = time.time()
        
        for component_id, component_info in self.components.items():
            # Skip inactive components
            if not component_info["active"]:
                continue
            
            # Check if we have a health monitor for this component
            if component_id in self.health_monitors:
                try:
                    health = self.health_monitors[component_id](component_info["component"])
                    
                    # Update health status
                    component_info["health"] = health
                    component_info["last_checked"] = current_time
                    
                    # Check for health issues
                    if health < 0.3:
                        self.report_fault(
                            component_id,
                            FaultType.UNKNOWN,
                            FaultSeverity.CRITICAL,
                            {"health": health}
                        )
                    elif health < 0.6:
                        self.report_fault(
                            component_id,
                            FaultType.UNKNOWN,
                            FaultSeverity.HIGH,
                            {"health": health}
                        )
                    elif health < 0.8:
                        self.report_fault(
                            component_id,
                            FaultType.UNKNOWN,
                            FaultSeverity.MEDIUM,
                            {"health": health}
                        )
                except Exception as e:
                    logger.error(f"Error checking health for {component_id}: {str(e)}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dict[str, Any]: System health information
        """
        # Calculate overall health metrics
        total_components = len(self.components)
        active_components = sum(1 for info in self.components.values() if info["active"])
        
        # Calculate average health of active components
        active_health = [
            info["health"] for info in self.components.values() 
            if info["active"]
        ]
        avg_health = sum(active_health) / len(active_health) if active_health else 0.0
        
        # Calculate critical component health
        critical_health = [
            self.components[cid]["health"] for cid in self.critical_components
            if cid in self.components and self.components[cid]["active"]
        ]
        critical_avg_health = sum(critical_health) / len(critical_health) if critical_health else 0.0
        
        # Count active faults by severity
        fault_counts = {severity.value: 0 for severity in FaultSeverity}
        for fault in self.active_faults.values():
            fault_counts[fault.severity.value] += 1
        
        return {
            "timestamp": time.time(),
            "total_components": total_components,
            "active_components": active_components,
            "overall_health": avg_health,
            "critical_health": critical_avg_health,
            "active_faults": fault_counts,
            "total_active_faults": len(self.active_faults),
            "total_fault_history": len(self.fault_history)
        }
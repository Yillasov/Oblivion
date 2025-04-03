#!/usr/bin/env python3
"""
Neuromorphic Hardware Monitor

Monitors neuromorphic hardware health and performance,
integrating with the fault tolerance manager.
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
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging

from src.core.hardware.fault_tolerance import (
    FaultToleranceManager, FaultType, FaultSeverity, RedundancyStrategy
)
from src.core.utils.logging_framework import get_logger

logger = get_logger("hardware_monitor")

class HardwareMetricType(Enum):
    """Types of hardware metrics to monitor."""
    TEMPERATURE = "temperature"
    POWER = "power"
    UTILIZATION = "utilization"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    SPIKE_RATE = "spike_rate"

class NeuromorphicHardwareMonitor:
    """
    Monitors neuromorphic hardware health and performance.
    
    Integrates with the fault tolerance manager to report issues
    and trigger appropriate responses.
    """
    
    def __init__(self, fault_manager: FaultToleranceManager):
        """
        Initialize the hardware monitor.
        
        Args:
            fault_manager: Fault tolerance manager instance
        """
        self.fault_manager = fault_manager
        self.hardware_components: Dict[str, Dict[str, Any]] = {}
        self.metric_thresholds: Dict[str, Dict[HardwareMetricType, Dict[str, float]]] = {}
        self.metric_history: Dict[str, Dict[HardwareMetricType, List[Dict[str, Any]]]] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 0.5  # seconds
        
        logger.info("Neuromorphic hardware monitor initialized")
    
    def register_hardware(self, 
                         hardware_id: str, 
                         hardware_component: Any,
                         hardware_type: str,
                         is_critical: bool = False,
                         redundancy_group: Optional[str] = None) -> bool:
        """
        Register hardware for monitoring.
        
        Args:
            hardware_id: Unique identifier for the hardware
            hardware_component: The hardware component object
            hardware_type: Type of hardware (e.g., "loihi", "spinnaker")
            is_critical: Whether this is a critical component
            redundancy_group: Optional group name for redundant components
            
        Returns:
            bool: Success status
        """
        if hardware_id in self.hardware_components:
            logger.warning(f"Hardware {hardware_id} already registered")
            return False
        
        # Register with fault manager
        self.fault_manager.register_component(
            hardware_id,
            hardware_component,
            is_critical,
            redundancy_group,
            RedundancyStrategy.PASSIVE
        )
        
        # Register health monitor
        self.fault_manager.register_health_monitor(
            hardware_id,
            lambda component: self._check_hardware_health(hardware_id)
        )
        
        # Store hardware info
        self.hardware_components[hardware_id] = {
            "component": hardware_component,
            "type": hardware_type,
            "is_critical": is_critical,
            "redundancy_group": redundancy_group,
            "last_checked": time.time(),
            "metrics": {}
        }
        
        # Initialize metric history
        self.metric_history[hardware_id] = {
            metric_type: [] for metric_type in HardwareMetricType
        }
        
        # Set default thresholds based on hardware type
        self._set_default_thresholds(hardware_id, hardware_type)
        
        logger.info(f"Registered hardware {hardware_id} ({hardware_type}) for monitoring")
        return True
    
    def _set_default_thresholds(self, hardware_id: str, hardware_type: str):
        """Set default metric thresholds based on hardware type."""
        # Base thresholds
        base_thresholds = {
            HardwareMetricType.TEMPERATURE: {
                "warning": 70.0,  # degrees C
                "critical": 85.0
            },
            HardwareMetricType.POWER: {
                "warning": 0.8,  # normalized to max power
                "critical": 0.95
            },
            HardwareMetricType.UTILIZATION: {
                "warning": 0.9,  # 90% utilization
                "critical": 0.98
            },
            HardwareMetricType.ERROR_RATE: {
                "warning": 0.05,  # 5% error rate
                "critical": 0.15
            },
            HardwareMetricType.RESPONSE_TIME: {
                "warning": 0.8,  # normalized to expected response time
                "critical": 0.95
            },
            HardwareMetricType.MEMORY_USAGE: {
                "warning": 0.85,  # 85% memory usage
                "critical": 0.95
            },
            HardwareMetricType.SPIKE_RATE: {
                "warning": 0.9,  # 90% of max spike rate
                "critical": 0.98
            }
        }
        
        # Adjust thresholds based on hardware type
        if hardware_type == "loihi":
            # Loihi has better thermal characteristics
            base_thresholds[HardwareMetricType.TEMPERATURE]["warning"] = 75.0
            base_thresholds[HardwareMetricType.TEMPERATURE]["critical"] = 90.0
        elif hardware_type == "spinnaker":
            # SpiNNaker has different memory constraints
            base_thresholds[HardwareMetricType.MEMORY_USAGE]["warning"] = 0.8
            base_thresholds[HardwareMetricType.MEMORY_USAGE]["critical"] = 0.9
        elif hardware_type == "truenorth":
            # TrueNorth has different power characteristics
            base_thresholds[HardwareMetricType.POWER]["warning"] = 0.7
            base_thresholds[HardwareMetricType.POWER]["critical"] = 0.85
        
        self.metric_thresholds[hardware_id] = base_thresholds
    
    def set_metric_threshold(self, 
                            hardware_id: str, 
                            metric_type: HardwareMetricType,
                            warning_threshold: float,
                            critical_threshold: float) -> bool:
        """
        Set custom thresholds for a specific metric.
        
        Args:
            hardware_id: Hardware identifier
            metric_type: Type of metric
            warning_threshold: Threshold for warning level
            critical_threshold: Threshold for critical level
            
        Returns:
            bool: Success status
        """
        if hardware_id not in self.hardware_components:
            logger.warning(f"Cannot set threshold: Hardware {hardware_id} not found")
            return False
        
        if hardware_id not in self.metric_thresholds:
            self.metric_thresholds[hardware_id] = {}
        
        if metric_type not in self.metric_thresholds[hardware_id]:
            self.metric_thresholds[hardware_id][metric_type] = {}
        
        self.metric_thresholds[hardware_id][metric_type]["warning"] = warning_threshold
        self.metric_thresholds[hardware_id][metric_type]["critical"] = critical_threshold
        
        logger.info(f"Set {metric_type.value} thresholds for {hardware_id}: "
                   f"warning={warning_threshold}, critical={critical_threshold}")
        
        return True
    
    def update_metrics(self, hardware_id: str, metrics: Dict[HardwareMetricType, float]) -> bool:
        """
        Update metrics for a hardware component.
        
        Args:
            hardware_id: Hardware identifier
            metrics: Dictionary of metric values
            
        Returns:
            bool: Success status
        """
        if hardware_id not in self.hardware_components:
            logger.warning(f"Cannot update metrics: Hardware {hardware_id} not found")
            return False
        
        hardware_info = self.hardware_components[hardware_id]
        current_time = time.time()
        
        # Update metrics
        for metric_type, value in metrics.items():
            # Store current value
            hardware_info["metrics"][metric_type] = value
            
            # Add to history
            if hardware_id in self.metric_history and metric_type in self.metric_history[hardware_id]:
                self.metric_history[hardware_id][metric_type].append({
                    "timestamp": current_time,
                    "value": value
                })
                
                # Limit history size
                if len(self.metric_history[hardware_id][metric_type]) > 1000:
                    self.metric_history[hardware_id][metric_type] = self.metric_history[hardware_id][metric_type][-1000:]
            
            # Check thresholds
            self._check_metric_threshold(hardware_id, metric_type, value)
        
        # Update last checked time
        hardware_info["last_checked"] = current_time
        
        return True
    
    def _check_metric_threshold(self, hardware_id: str, metric_type: HardwareMetricType, value: float):
        """Check if a metric exceeds thresholds and report faults if needed."""
        if hardware_id not in self.metric_thresholds:
            return
        
        if metric_type not in self.metric_thresholds[hardware_id]:
            return
        
        thresholds = self.metric_thresholds[hardware_id][metric_type]
        
        # Check critical threshold
        if "critical" in thresholds and value >= thresholds["critical"]:
            # Report critical fault
            self.fault_manager.report_fault(
                hardware_id,
                self._metric_to_fault_type(metric_type),
                FaultSeverity.CRITICAL,
                {
                    "metric_type": metric_type.value,
                    "value": value,
                    "threshold": thresholds["critical"]
                }
            )
        # Check warning threshold
        elif "warning" in thresholds and value >= thresholds["warning"]:
            # Report warning fault
            self.fault_manager.report_fault(
                hardware_id,
                self._metric_to_fault_type(metric_type),
                FaultSeverity.MEDIUM,
                {
                    "metric_type": metric_type.value,
                    "value": value,
                    "threshold": thresholds["warning"]
                }
            )
    
    def _metric_to_fault_type(self, metric_type: HardwareMetricType) -> FaultType:
        """Convert metric type to fault type."""
        if metric_type == HardwareMetricType.TEMPERATURE:
            return FaultType.THERMAL
        elif metric_type == HardwareMetricType.POWER:
            return FaultType.POWER
        elif metric_type == HardwareMetricType.RESPONSE_TIME:
            return FaultType.TIMING
        elif metric_type == HardwareMetricType.MEMORY_USAGE:
            return FaultType.MEMORY
        elif metric_type == HardwareMetricType.ERROR_RATE:
            return FaultType.COMPUTATION
        else:
            return FaultType.UNKNOWN
    
    def _check_hardware_health(self, hardware_id: str) -> float:
        """
        Calculate overall health score for hardware component.
        
        Args:
            hardware_id: Hardware identifier
            
        Returns:
            float: Health score (0.0-1.0)
        """
        if hardware_id not in self.hardware_components:
            return 0.0
        
        hardware_info = self.hardware_components[hardware_id]
        metrics = hardware_info.get("metrics", {})
        
        if not metrics:
            return 1.0  # No metrics yet, assume healthy
        
        # Calculate health based on metrics and thresholds
        health_scores = []
        
        for metric_type, value in metrics.items():
            if hardware_id in self.metric_thresholds and metric_type in self.metric_thresholds[hardware_id]:
                thresholds = self.metric_thresholds[hardware_id][metric_type]
                
                if "critical" in thresholds:
                    # Calculate normalized health score
                    if value >= thresholds["critical"]:
                        health_scores.append(0.0)  # Critical threshold exceeded
                    elif "warning" in thresholds and value >= thresholds["warning"]:
                        # Linear interpolation between warning and critical
                        warning = thresholds["warning"]
                        critical = thresholds["critical"]
                        health = 0.5 - 0.5 * (value - warning) / (critical - warning)
                        health_scores.append(max(0.0, health))
                    else:
                        # Below warning threshold
                        if "warning" in thresholds:
                            # Linear interpolation between perfect and warning
                            warning = thresholds["warning"]
                            health = 1.0 - 0.5 * (value / warning)
                            health_scores.append(max(0.5, health))
                        else:
                            health_scores.append(1.0)
        
        # Calculate overall health as average of individual scores
        if health_scores:
            return sum(health_scores) / len(health_scores)
        else:
            return 1.0  # No health scores calculated, assume healthy
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start fault manager monitoring as well
        self.fault_manager.start_monitoring()
        
        logger.info("Started hardware monitoring")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Stop fault manager monitoring as well
        self.fault_manager.stop_monitoring()
        
        logger.info("Stopped hardware monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # In a real implementation, we would poll hardware for metrics
                # For now, we just check if any components haven't been updated recently
                self._check_stale_components()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(1.0)  # Sleep on error to avoid tight loop
    
    def _check_stale_components(self):
        """Check for components that haven't been updated recently."""
        current_time = time.time()
        stale_threshold = 5.0  # seconds
        
        for hardware_id, hardware_info in self.hardware_components.items():
            last_checked = hardware_info.get("last_checked", 0)
            
            # Check if component hasn't been updated recently
            if current_time - last_checked > stale_threshold:
                # Report communication fault
                self.fault_manager.report_fault(
                    hardware_id,
                    FaultType.COMMUNICATION,
                    FaultSeverity.MEDIUM,
                    {
                        "last_checked": last_checked,
                        "current_time": current_time,
                        "time_difference": current_time - last_checked
                    }
                )
    
    def get_hardware_status(self, hardware_id: str) -> Dict[str, Any]:
        """
        Get status information for a hardware component.
        
        Args:
            hardware_id: Hardware identifier
            
        Returns:
            Dict[str, Any]: Status information
        """
        if hardware_id not in self.hardware_components:
            return {"error": f"Hardware {hardware_id} not found"}
        
        hardware_info = self.hardware_components[hardware_id]
        
        # Calculate health score
        health_score = self._check_hardware_health(hardware_id)
        
        # Get recent metric history (last 10 entries for each metric)
        recent_metrics = {}
        if hardware_id in self.metric_history:
            for metric_type, history in self.metric_history[hardware_id].items():
                if history:
                    recent_metrics[metric_type.value] = history[-10:]
        
        # Get active faults for this component
        active_faults = [
            {
                "fault_id": fault_id,
                "fault_type": fault.fault_type.value,
                "severity": fault.severity.value,
                "timestamp": fault.timestamp,
                "details": fault.details
            }
            for fault_id, fault in self.fault_manager.active_faults.items()
            if fault.component_id == hardware_id
        ]
        
        return {
            "hardware_id": hardware_id,
            "hardware_type": hardware_info["type"],
            "is_critical": hardware_info["is_critical"],
            "redundancy_group": hardware_info["redundancy_group"],
            "health_score": health_score,
            "current_metrics": {
                metric_type.value: value 
                for metric_type, value in hardware_info.get("metrics", {}).items()
            },
            "recent_metrics": recent_metrics,
            "active_faults": active_faults,
            "last_checked": hardware_info["last_checked"]
        }
    
    def get_all_hardware_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all hardware components.
        
        Returns:
            Dict[str, Dict[str, Any]]: Status information by hardware ID
        """
        return {
            hardware_id: self.get_hardware_status(hardware_id)
            for hardware_id in self.hardware_components
        }
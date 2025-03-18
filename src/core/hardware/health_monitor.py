"""
Hardware Health Monitoring System

Provides real-time monitoring of neuromorphic hardware health metrics
and alerts for potential issues.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta

from src.core.utils.logging_framework import get_logger
from src.core.hardware.hardware_registry import hardware_registry
from src.core.hardware.error_codes import HardwareErrorCode
from src.core.hardware.recovery_strategies import attempt_recovery

logger = get_logger("hardware_health")

# Health status constants
HEALTH_CRITICAL = "CRITICAL"
HEALTH_WARNING = "WARNING"
HEALTH_NORMAL = "NORMAL"
HEALTH_UNKNOWN = "UNKNOWN"


class HealthMetric:
    """Hardware health metric definition."""
    
    def __init__(self, 
                 name: str, 
                 description: str,
                 warning_threshold: float,
                 critical_threshold: float,
                 unit: str = "",
                 lower_is_better: bool = False):
        """
        Initialize health metric.
        
        Args:
            name: Metric name
            description: Metric description
            warning_threshold: Warning threshold value
            critical_threshold: Critical threshold value
            unit: Unit of measurement
            lower_is_better: True if lower values are better
        """
        self.name = name
        self.description = description
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.unit = unit
        self.lower_is_better = lower_is_better
        
    def evaluate(self, value: float) -> str:
        """
        Evaluate metric value against thresholds.
        
        Args:
            value: Metric value
            
        Returns:
            str: Health status (CRITICAL, WARNING, or NORMAL)
        """
        if self.lower_is_better:
            if value >= self.critical_threshold:
                return HEALTH_CRITICAL
            elif value >= self.warning_threshold:
                return HEALTH_WARNING
            else:
                return HEALTH_NORMAL
        else:
            if value <= self.critical_threshold:
                return HEALTH_CRITICAL
            elif value <= self.warning_threshold:
                return HEALTH_WARNING
            else:
                return HEALTH_NORMAL


class HardwareHealthMonitor:
    """Hardware health monitoring system."""
    
    def __init__(self, 
                 check_interval: float = 60.0,
                 auto_recovery: bool = True):
        """
        Initialize hardware health monitor.
        
        Args:
            check_interval: Interval between health checks in seconds
            auto_recovery: Enable automatic recovery attempts
        """
        self.check_interval = check_interval
        self.auto_recovery = auto_recovery
        self.is_monitoring = False
        self.monitor_thread = None
        self.health_metrics = {}
        self.health_history = []
        self.max_history = 100
        self.alert_callbacks = []
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default health metrics."""
        # Temperature metric (higher is worse)
        self.add_metric(HealthMetric(
            name="temperature",
            description="Hardware temperature",
            warning_threshold=70.0,
            critical_threshold=85.0,
            unit="°C",
            lower_is_better=True
        ))
        
        # Power consumption metric (higher is worse)
        self.add_metric(HealthMetric(
            name="power_consumption",
            description="Power consumption",
            warning_threshold=80.0,
            critical_threshold=95.0,
            unit="W",
            lower_is_better=True
        ))
        
        # Error rate metric (higher is worse)
        self.add_metric(HealthMetric(
            name="error_rate",
            description="Hardware error rate",
            warning_threshold=0.01,
            critical_threshold=0.05,
            unit="%",
            lower_is_better=True
        ))
        
        # Response time metric (higher is worse)
        self.add_metric(HealthMetric(
            name="response_time",
            description="Hardware response time",
            warning_threshold=100.0,
            critical_threshold=500.0,
            unit="ms",
            lower_is_better=True
        ))
        
        # Resource utilization metric (higher is worse)
        self.add_metric(HealthMetric(
            name="resource_utilization",
            description="Resource utilization",
            warning_threshold=80.0,
            critical_threshold=95.0,
            unit="%",
            lower_is_better=True
        ))
    
    def add_metric(self, metric: HealthMetric) -> None:
        """
        Add a health metric to monitor.
        
        Args:
            metric: Health metric to add
        """
        self.health_metrics[metric.name] = metric
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add alert callback function.
        
        Args:
            callback: Function to call when health issues are detected
        """
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self) -> bool:
        """
        Start health monitoring.
        
        Returns:
            bool: Success status
        """
        if self.is_monitoring:
            logger.warning("Health monitoring already started")
            return False
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"Started hardware health monitoring with interval {self.check_interval}s")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop health monitoring.
        
        Returns:
            bool: Success status
        """
        if not self.is_monitoring:
            logger.warning("Health monitoring not started")
            return False
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Stopped hardware health monitoring")
        return True
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check health for all registered hardware
                for hw_type in hardware_registry.get_available_hardware_types():
                    hw = hardware_registry.get_hardware(hw_type)
                    if not hw:
                        continue
                    
                    # Get hardware metrics
                    metrics = self._get_hardware_metrics(hw)
                    
                    # Evaluate health
                    health_status = self._evaluate_health(metrics)
                    
                    # Add to history
                    self._add_to_history(hw_type, metrics, health_status)
                    
                    # Handle critical issues
                    self._handle_health_issues(hw_type, health_status)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
            
            # Sleep until next check
            time.sleep(self.check_interval)
    
    def _get_hardware_metrics(self, hardware) -> Dict[str, float]:
        """
        Get hardware metrics.
        
        Args:
            hardware: Hardware instance
            
        Returns:
            Dict[str, float]: Hardware metrics
        """
        # In a real implementation, this would get actual metrics from hardware
        # For now, we'll use simulated metrics or metrics from the hardware interface
        
        metrics = {}
        
        # Try to get metrics from hardware interface
        if hasattr(hardware, "get_metrics"):
            hw_metrics = hardware.get_metrics()
            metrics.update(hw_metrics)
        
        # Fill in missing metrics with simulated values
        for metric_name in self.health_metrics:
            if metric_name not in metrics:
                # Simulate metric value (normally distributed around 50%)
                import random
                metrics[metric_name] = random.normalvariate(50, 20)
        
        return metrics
    
    def _evaluate_health(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Evaluate hardware health based on metrics.
        
        Args:
            metrics: Hardware metrics
            
        Returns:
            Dict[str, str]: Health status for each metric
        """
        health_status = {}
        
        for metric_name, value in metrics.items():
            if metric_name in self.health_metrics:
                metric = self.health_metrics[metric_name]
                health_status[metric_name] = metric.evaluate(value)
            else:
                health_status[metric_name] = HEALTH_UNKNOWN
        
        return health_status
    
    def _add_to_history(self, 
                       hardware_type: str, 
                       metrics: Dict[str, float],
                       health_status: Dict[str, str]) -> None:
        """
        Add health check to history.
        
        Args:
            hardware_type: Hardware type
            metrics: Hardware metrics
            health_status: Health status
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "hardware_type": hardware_type,
            "metrics": metrics,
            "health_status": health_status,
            "overall_health": self._calculate_overall_health(health_status)
        }
        
        self.health_history.append(entry)
        
        # Trim history if needed
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]
    
    def _calculate_overall_health(self, health_status: Dict[str, str]) -> str:
        """
        Calculate overall health status.
        
        Args:
            health_status: Health status for each metric
            
        Returns:
            str: Overall health status
        """
        if HEALTH_CRITICAL in health_status.values():
            return HEALTH_CRITICAL
        elif HEALTH_WARNING in health_status.values():
            return HEALTH_WARNING
        elif all(status == HEALTH_NORMAL for status in health_status.values()):
            return HEALTH_NORMAL
        else:
            return HEALTH_UNKNOWN
    
    def _handle_health_issues(self, hardware_type: str, health_status: Dict[str, str]) -> None:
        """
        Handle health issues.
        
        Args:
            hardware_type: Hardware type
            health_status: Health status
        """
        # Check for critical issues
        critical_issues = [
            metric for metric, status in health_status.items() 
            if status == HEALTH_CRITICAL
        ]
        
        if critical_issues:
            logger.warning(f"Critical health issues detected for {hardware_type}: {', '.join(critical_issues)}")
            
            # Create alert
            alert = {
                "timestamp": datetime.now().isoformat(),
                "hardware_type": hardware_type,
                "level": HEALTH_CRITICAL,
                "issues": critical_issues,
                "message": f"Critical health issues detected for {hardware_type}"
            }
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {str(e)}")
            
            # Attempt recovery if enabled
            if self.auto_recovery:
                for issue in critical_issues:
                    error_code = self._map_issue_to_error_code(issue)
                    if error_code:
                        logger.info(f"Attempting recovery for {issue} on {hardware_type}")
                        attempt_recovery(hardware_type, error_code)
    
    def _map_issue_to_error_code(self, issue: str) -> Optional[HardwareErrorCode]:
        """
        Map health issue to error code.
        
        Args:
            issue: Health issue
            
        Returns:
            Optional[HardwareErrorCode]: Corresponding error code or None
        """
        # Map common issues to error codes
        mapping = {
            "temperature": HardwareErrorCode.HARDWARE_SWITCHING_FAILED,
            "power_consumption": HardwareErrorCode.RESOURCE_ALLOCATION_FAILED,
            "error_rate": HardwareErrorCode.SIMULATION_FAILED,
            "response_time": HardwareErrorCode.COMMUNICATION_FAILED,
            "resource_utilization": HardwareErrorCode.RESOURCE_ALLOCATION_FAILED
        }
        
        return mapping.get(issue)
    
    def get_hardware_health(self, hardware_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current hardware health status.
        
        Args:
            hardware_type: Hardware type (optional)
            
        Returns:
            Dict[str, Any]: Hardware health status
        """
        if not self.health_history:
            return {"status": "No health data available"}
        
        # Filter by hardware type if specified
        if hardware_type:
            history = [entry for entry in self.health_history if entry["hardware_type"] == hardware_type]
            if not history:
                return {"status": f"No health data available for {hardware_type}"}
        else:
            history = self.health_history
        
        # Get most recent entry for each hardware type
        latest_entries = {}
        for entry in reversed(history):
            hw_type = entry["hardware_type"]
            if hw_type not in latest_entries:
                latest_entries[hw_type] = entry
        
        return {
            "timestamp": datetime.now().isoformat(),
            "hardware_health": list(latest_entries.values()),
            "overall_health": self._calculate_overall_health_from_entries(latest_entries.values())
        }
    
    def _calculate_overall_health_from_entries(self, entries) -> str:
        """
        Calculate overall health from entries.
        
        Args:
            entries: Health history entries
            
        Returns:
            str: Overall health status
        """
        overall_health = [entry.get("overall_health", HEALTH_UNKNOWN) for entry in entries]
        
        if HEALTH_CRITICAL in overall_health:
            return HEALTH_CRITICAL
        elif HEALTH_WARNING in overall_health:
            return HEALTH_WARNING
        elif all(health == HEALTH_NORMAL for health in overall_health):
            return HEALTH_NORMAL
        else:
            return HEALTH_UNKNOWN
    
    def get_health_history(self, 
                          hardware_type: Optional[str] = None,
                          hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get hardware health history.
        
        Args:
            hardware_type: Hardware type (optional)
            hours: Number of hours to include (optional)
            
        Returns:
            List[Dict[str, Any]]: Health history
        """
        # Filter by hardware type if specified
        if hardware_type:
            history = [entry for entry in self.health_history if entry["hardware_type"] == hardware_type]
        else:
            history = self.health_history
        
        # Filter by time if specified
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            history = [
                entry for entry in history 
                if datetime.fromisoformat(entry["timestamp"]) >= cutoff
            ]
        
        return history


# Global health monitor instance
health_monitor = HardwareHealthMonitor()


def start_health_monitoring(check_interval: float = 60.0, auto_recovery: bool = True) -> bool:
    """
    Start hardware health monitoring.
    
    Args:
        check_interval: Interval between health checks in seconds
        auto_recovery: Enable automatic recovery attempts
        
    Returns:
        bool: Success status
    """
    global health_monitor
    health_monitor = HardwareHealthMonitor(check_interval, auto_recovery)
    return health_monitor.start_monitoring()


def stop_health_monitoring() -> bool:
    """
    Stop hardware health monitoring.
    
    Returns:
        bool: Success status
    """
    return health_monitor.stop_monitoring()


def get_hardware_health(hardware_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current hardware health status.
    
    Args:
        hardware_type: Hardware type (optional)
        
    Returns:
        Dict[str, Any]: Hardware health status
    """
    return health_monitor.get_hardware_health(hardware_type)


def register_health_alert_callback(callback: Callable[[Dict[str, Any]], None]) -> None:
    """
    Register callback for health alerts.
    
    Args:
        callback: Function to call when health issues are detected
    """
    health_monitor.add_alert_callback(callback)
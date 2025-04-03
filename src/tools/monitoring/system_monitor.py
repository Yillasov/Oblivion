#!/usr/bin/env python3
"""
Simple System Monitoring and Diagnostics

Provides basic monitoring and diagnostics for neuromorphic hardware systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import sys
import time
import json
import psutil
import platform
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append("/Users/yessine/Oblivion")

# Import the logger instance and get_logger function
from src.core.utils.logging_framework import get_logger, neuromorphic_logger # Import the instance
from src.core.hardware.config_manager import HardwareConfigManager

logger = get_logger("system_monitor")


class SystemMonitor:
    """Simple system monitor for neuromorphic hardware."""
    
    def __init__(self, 
                 monitor_interval: float = 5.0,
                 log_dir: str = "/Users/yessine/Oblivion/logs/monitoring"):
        """
        Initialize system monitor.
        
        Args:
            monitor_interval: Interval between monitoring checks (seconds)
            log_dir: Directory to store monitoring logs
        """
        self.monitor_interval = monitor_interval
        self.log_dir = log_dir
        self.is_monitoring = False
        self.monitor_thread = None
        self.hardware_configs = HardwareConfigManager()
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "system": {},
            "hardware": {},
            "processes": {}
        }
        
        logger.info("Initialized system monitor")
    
    def start_monitoring(self) -> bool:
        """Start monitoring system."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return False
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"Started monitoring with interval {self.monitor_interval}s")
        return True
    
    def stop_monitoring(self) -> bool:
        """Stop monitoring system."""
        if not self.is_monitoring:
            logger.warning("Monitoring not started")
            return False
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Stopped monitoring")
        return True
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                self._collect_system_metrics()
                self._collect_hardware_metrics()
                self._collect_process_metrics()
                
                # Log metrics
                self._log_metrics()
                
                # Check for issues
                issues = self._check_for_issues()
                if issues:
                    logger.warning(f"Detected {len(issues)} issues")
                    for issue in issues:
                        logger.warning(f"  - {issue}")
                
                # Sleep until next check
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        self.metrics["system"] = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "boot_time": psutil.boot_time(),
            "platform": platform.platform(),
            "python_version": platform.python_version()
        }
    
    def _collect_hardware_metrics(self) -> None:
        """Collect hardware-specific metrics."""
        # This would connect to actual hardware in a real implementation
        # For now, we'll just simulate some metrics
        self.metrics["hardware"] = {
            "timestamp": datetime.now().isoformat(),
            "temperature": 45.2,  # Simulated temperature in Celsius
            "power_consumption": 15.7,  # Simulated power in Watts
            "active_neurons": 1024,  # Simulated active neurons
            "active_synapses": 8192,  # Simulated active synapses
            "error_count": 0,  # Simulated error count
            "uptime": 3600  # Simulated uptime in seconds
        }
    
    def _collect_process_metrics(self) -> None:
        """Collect process metrics."""
        processes = {}
        
        # Get neuromorphic-related processes
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
            try:
                # Filter for relevant processes (customize as needed)
                if any(x in proc.info['name'].lower() for x in ['python', 'neuro', 'sim']):
                    processes[proc.info['pid']] = {
                        "name": proc.info['name'],
                        "username": proc.info['username'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": proc.info['memory_percent'],
                        "create_time": datetime.fromtimestamp(proc.create_time()).isoformat()
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        self.metrics["processes"] = processes
    
    def _log_metrics(self) -> None:
        """Log metrics to file."""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"metrics_{timestamp}.json")
        
        # Append metrics to log file
        with open(log_file, 'a') as f:
            f.write(json.dumps(self.metrics) + '\n')
    
    def _check_for_issues(self) -> List[str]:
        """Check for potential issues."""
        issues: List[str] = []
        
        try:
            # Check system metrics
            system = self.metrics.get("system", {})
            if not isinstance(system, dict):
                logger.warning("System metrics not available or invalid format")
                system = {}
                
            if system.get("cpu_percent", 0) > 90:
                issues.append("High CPU usage (>90%)")
            
            if system.get("memory_percent", 0) > 90:
                issues.append("High memory usage (>90%)")
            
            if system.get("disk_percent", 0) > 90:
                issues.append("Low disk space (<10% free)")
            
            # Check hardware metrics
            hardware = self.metrics.get("hardware", {})
            if not isinstance(hardware, dict):
                logger.warning("Hardware metrics not available or invalid format")
                hardware = {}
                
            if hardware.get("temperature", 0) > 80:
                issues.append(f"High hardware temperature: {hardware.get('temperature')}°C")
            
            if hardware.get("error_count", 0) > 0:
                issues.append(f"Hardware errors detected: {hardware.get('error_count')}")
            
            # Check for network connectivity issues
            network = self.metrics.get("network", {})
            if isinstance(network, dict) and network.get("packet_loss", 0) > 5:
                issues.append(f"Network packet loss detected: {network.get('packet_loss')}%")
                
            # Check for process-specific issues
            processes = self.metrics.get("processes", [])
            for process in processes:
                if isinstance(process, dict) and process.get("cpu_percent", 0) > 80:
                    issues.append(f"High CPU usage by process {process.get('name', 'unknown')}: {process.get('cpu_percent')}%")
        
        except Exception as e:
            logger.error(f"Error checking for issues: {str(e)}")
            issues.append(f"Monitoring error: {str(e)}")
        
        return issues
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        # Calculate health score (0-100)
        system = self.metrics["system"]
        hardware = self.metrics["hardware"]
        
        # Simple health calculation
        cpu_health = 100 - system.get("cpu_percent", 0)
        memory_health = 100 - system.get("memory_percent", 0)
        disk_health = 100 - system.get("disk_percent", 0)
        
        # Hardware health (temperature-based)
        temp = hardware.get("temperature", 0)
        if temp > 80:
            temp_health = 0
        elif temp > 60:
            temp_health = 50
        else:
            temp_health = 100
        
        # Overall health
        overall_health = (cpu_health + memory_health + disk_health + temp_health) / 4
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": overall_health,
            "cpu_health": cpu_health,
            "memory_health": memory_health,
            "disk_health": disk_health,
            "temperature_health": temp_health,
            "issues": self._check_for_issues()
        }
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a diagnostic report.
        
        Args:
            output_file: Path to output file (optional)
            
        Returns:
            str: Path to report file
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"diagnostic_report_{timestamp}.txt")
        
        with open(output_file, 'w') as f:
            # System information
            f.write("=== SYSTEM DIAGNOSTIC REPORT ===\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("--- System Information ---\n")
            for key, value in self.metrics["system"].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n--- Hardware Information ---\n")
            for key, value in self.metrics["hardware"].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n--- Process Information ---\n")
            for pid, proc_info in self.metrics["processes"].items():
                f.write(f"PID {pid}: {proc_info['name']} (CPU: {proc_info['cpu_percent']}%, Mem: {proc_info['memory_percent']}%)\n")
            
            # Health information
            f.write("\n--- Health Information ---\n")
            health = self.get_system_health()
            f.write(f"Overall Health: {health['overall_health']:.1f}%\n")
            f.write(f"CPU Health: {health['cpu_health']:.1f}%\n")
            f.write(f"Memory Health: {health['memory_health']:.1f}%\n")
            f.write(f"Disk Health: {health['disk_health']:.1f}%\n")
            f.write(f"Temperature Health: {health['temperature_health']:.1f}%\n")
            
            # Issues
            f.write("\n--- Detected Issues ---\n")
            issues = health["issues"]
            if issues:
                for issue in issues:
                    f.write(f"- {issue}\n")
            else:
                f.write("No issues detected\n")
        
        logger.info(f"Generated diagnostic report: {output_file}")
        return output_file


def main():
    """Main entry point for system monitor."""
    import argparse
    import logging # Import standard logging to access level constants

    parser = argparse.ArgumentParser(description="Neuromorphic System Monitor")
    parser.add_argument("--interval", type=float, default=5.0, help="Monitoring interval in seconds")
    parser.add_argument("--log-dir", default="/Users/yessine/Oblivion/logs/monitoring", help="Log directory")
    # Add log level argument
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    parser.add_argument("--report", action="store_true", help="Generate diagnostic report")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring")

    args = parser.parse_args()

    # Configure the global logging level using the custom framework instance
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    neuromorphic_logger.set_global_level(log_level) # Call the method on the instance
    logger.info(f"Set global logging level to: {args.log_level.upper()}")


    # Create system monitor
    monitor = SystemMonitor(
        monitor_interval=args.interval,
        log_dir=args.log_dir
    )
    
    if args.report:
        # Generate report
        report_path = monitor.generate_report()
        print(f"Generated diagnostic report: {report_path}")
    
    if args.monitor:
        # Start monitoring
        try:
            monitor.start_monitoring()
            print(f"Started monitoring with interval {args.interval}s. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while True:
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("Stopping monitoring...")
            monitor.stop_monitoring()
    
    if not args.report and not args.monitor:
        # Show health by default
        health = monitor.get_system_health()
        logger.debug(f"System health details: {health}") # Example debug log
        print(f"System Health: {health['overall_health']:.1f}%")

        if health["issues"]:
            print("\nDetected Issues:")
            for issue in health["issues"]:
                print(f"- {issue}")

# Example usage if run directly
if __name__ == "__main__":
    main()
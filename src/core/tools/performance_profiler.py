"""
Performance Profiling Tools for Neuromorphic Hardware

This module provides tools for profiling and optimizing performance
of neuromorphic hardware operations.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class OperationProfiler:
    """
    Profiles the execution time and resource usage of neuromorphic operations.
    """
    
    def __init__(self):
        """Initialize the operation profiler."""
        self.operation_times = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.resource_usage = defaultdict(list)
    
    def profile_operation(self, operation_name: str):
        """
        Decorator for profiling an operation.
        
        Args:
            operation_name: Name of the operation to profile
            
        Returns:
            Callable: Decorated function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Record start time
                start_time = time.time()
                
                # Execute the operation
                result = func(*args, **kwargs)
                
                # Record end time and calculate duration
                end_time = time.time()
                duration = end_time - start_time
                
                # Store timing information
                self.operation_times[operation_name].append(duration)
                self.operation_counts[operation_name] += 1
                
                logger.debug(f"Operation '{operation_name}' took {duration:.6f} seconds")
                
                return result
            return wrapper
        return decorator
    
    def record_resource_usage(self, operation_name: str, resources: Dict[str, float]):
        """
        Record resource usage for an operation.
        
        Args:
            operation_name: Name of the operation
            resources: Dictionary of resource usage metrics
        """
        self.resource_usage[operation_name].append(resources)
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operations.
        
        Args:
            operation_name: Optional name of specific operation to get stats for
            
        Returns:
            Dict[str, Any]: Dictionary of operation statistics
        """
        if operation_name:
            # Return stats for specific operation
            if operation_name not in self.operation_times:
                return {}
            
            times = self.operation_times[operation_name]
            count = self.operation_counts[operation_name]
            
            return {
                'count': count,
                'total_time': sum(times),
                'average_time': sum(times) / count if count > 0 else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'std_dev': np.std(times) if times else 0
            }
        else:
            # Return stats for all operations
            stats = {}
            
            for op_name in self.operation_times:
                stats[op_name] = self.get_operation_stats(op_name)
            
            return stats
    
    def get_resource_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get resource usage statistics.
        
        Args:
            operation_name: Optional name of specific operation to get stats for
            
        Returns:
            Dict[str, Any]: Dictionary of resource usage statistics
        """
        if operation_name:
            # Return stats for specific operation
            if operation_name not in self.resource_usage:
                return {}
            
            resources = self.resource_usage[operation_name]
            
            # Aggregate resource stats
            stats = {}
            
            if not resources:
                return stats
            
            # Get all resource types
            resource_types = set()
            for r in resources:
                resource_types.update(r.keys())
            
            # Calculate stats for each resource type
            for res_type in resource_types:
                values = [r.get(res_type, 0) for r in resources if res_type in r]
                
                if not values:
                    continue
                
                stats[res_type] = {
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std_dev': np.std(values)
                }
            
            return stats
        else:
            # Return stats for all operations
            stats = {}
            
            for op_name in self.resource_usage:
                stats[op_name] = self.get_resource_stats(op_name)
            
            return stats
    
    def reset(self):
        """Reset all profiling data."""
        self.operation_times.clear()
        self.operation_counts.clear()
        self.resource_usage.clear()
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dict[str, Any]: Performance report
        """
        report = {
            'operations': self.get_operation_stats(),
            'resources': self.get_resource_stats(),
            'summary': {
                'total_operations': sum(self.operation_counts.values()),
                'total_time': sum(sum(times) for times in self.operation_times.values()),
                'operation_types': len(self.operation_times),
                'slowest_operation': None,
                'fastest_operation': None
            }
        }
        
        # Find slowest and fastest operations
        if self.operation_times:
            avg_times = {op: sum(times) / self.operation_counts[op] 
                        for op, times in self.operation_times.items()}
            
            slowest_op = max(avg_times.items(), key=lambda x: x[1])
            fastest_op = min(avg_times.items(), key=lambda x: x[1])
            
            report['summary']['slowest_operation'] = {
                'name': slowest_op[0],
                'average_time': slowest_op[1]
            }
            
            report['summary']['fastest_operation'] = {
                'name': fastest_op[0],
                'average_time': fastest_op[1]
            }
        
        return report


class HardwareProfiler:
    """
    Profiles neuromorphic hardware performance and resource utilization.
    """
    
    def __init__(self, hardware_interface):
        """
        Initialize the hardware profiler.
        
        Args:
            hardware_interface: Interface to the neuromorphic hardware
        """
        self.hardware = hardware_interface
        self.operation_profiler = OperationProfiler()
        self.neuron_utilization = []
        self.synapse_utilization = []
        self.power_measurements = []
        self.timing_measurements = []
    
    def measure_neuron_utilization(self) -> Dict[str, float]:
        """
        Measure neuron utilization on the hardware.
        
        Returns:
            Dict[str, float]: Neuron utilization metrics
        """
        # In a real implementation, this would query the hardware
        # For this simple version, we'll generate placeholder data
        
        # Get hardware info
        hw_info = self.hardware.get_hardware_info()
        
        # Calculate utilization
        neurons_allocated = hw_info.get('neurons_allocated', 0)
        neurons_available = hw_info.get('cores_available', 1) * hw_info.get('neurons_per_core', 1024)
        
        utilization = neurons_allocated / neurons_available if neurons_available > 0 else 0
        
        metrics = {
            'neurons_allocated': neurons_allocated,
            'neurons_available': neurons_available,
            'utilization': utilization
        }
        
        self.neuron_utilization.append(metrics)
        return metrics
    
    def measure_synapse_utilization(self) -> Dict[str, float]:
        """
        Measure synapse utilization on the hardware.
        
        Returns:
            Dict[str, float]: Synapse utilization metrics
        """
        # In a real implementation, this would query the hardware
        # For this simple version, we'll generate placeholder data
        
        # Get hardware info
        hw_info = self.hardware.get_hardware_info()
        
        # Calculate utilization
        synapses_allocated = hw_info.get('synapses_allocated', 0)
        synapses_available = hw_info.get('cores_available', 1) * 1024 * 1024  # Placeholder
        
        utilization = synapses_allocated / synapses_available if synapses_available > 0 else 0
        
        metrics = {
            'synapses_allocated': synapses_allocated,
            'synapses_available': synapses_available,
            'utilization': utilization
        }
        
        self.synapse_utilization.append(metrics)
        return metrics
    
    def measure_power_consumption(self) -> Dict[str, float]:
        """
        Measure power consumption of the hardware.
        
        Returns:
            Dict[str, float]: Power consumption metrics
        """
        # In a real implementation, this would query the hardware
        # For this simple version, we'll generate placeholder data
        
        metrics = {
            'total_power_mw': 100.0,  # Placeholder
            'core_power_mw': 80.0,    # Placeholder
            'io_power_mw': 20.0       # Placeholder
        }
        
        self.power_measurements.append(metrics)
        return metrics
    
    def measure_execution_time(self, operation_func, *args, **kwargs) -> float:
        """
        Measure execution time of an operation on the hardware.
        
        Args:
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            float: Execution time in seconds
        """
        start_time = time.time()
        result = operation_func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        self.timing_measurements.append({
            'operation': operation_func.__name__,
            'execution_time': execution_time
        })
        
        return execution_time
    
    def generate_optimization_recommendations(self) -> List[str]:
        """
        Generate recommendations for hardware optimization.
        
        Returns:
            List[str]: List of optimization recommendations
        """
        recommendations = []
        
        # Check neuron utilization
        if self.neuron_utilization:
            latest = self.neuron_utilization[-1]
            if latest['utilization'] < 0.3:
                recommendations.append("Low neuron utilization detected. Consider reducing the number of allocated neurons.")
            elif latest['utilization'] > 0.9:
                recommendations.append("High neuron utilization detected. Consider distributing computation across more cores.")
        
        # Check synapse utilization
        if self.synapse_utilization:
            latest = self.synapse_utilization[-1]
            if latest['utilization'] < 0.3:
                recommendations.append("Low synapse utilization detected. Consider optimizing network connectivity.")
            elif latest['utilization'] > 0.9:
                recommendations.append("High synapse utilization detected. Consider pruning less important connections.")
        
        # Check operation performance
        op_stats = self.operation_profiler.get_operation_stats()
        slow_ops = [op for op, stats in op_stats.items() if stats.get('average_time', 0) > 0.1]
        
        if slow_ops:
            recommendations.append(f"Slow operations detected: {', '.join(slow_ops)}. Consider optimizing these operations.")
        
        return recommendations
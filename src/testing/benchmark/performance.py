from typing import Dict, Any, List, Optional, Callable
import numpy as np
import time
import json
import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BenchmarkMetric:
    """Represents a performance benchmark metric."""
    name: str
    value: float
    unit: str
    timestamp: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp
        }

class PerformanceBenchmark:
    """Performance benchmarking tools for neuromorphic systems."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics: List[BenchmarkMetric] = []
        self.start_time = time.time()
        self.output_dir = config.get("output_dir", "/Users/yessine/Oblivion/benchmark_results")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def record_metric(self, name: str, value: float, unit: str) -> None:
        """Record a performance metric."""
        metric = BenchmarkMetric(name, value, unit)
        self.metrics.append(metric)
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> float:
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        self.record_metric(f"{func.__name__}_execution_time", execution_time, "seconds")
        return execution_time
    
    def measure_power_usage(self, hardware_interface) -> float:
        """Measure power usage of hardware."""
        if hasattr(hardware_interface, "hardware") and hasattr(hardware_interface.hardware, "get_power_usage"):
            power_usage = hardware_interface.hardware.get_power_usage()
            self.record_metric("power_usage", power_usage, "watts")
            return power_usage
        return 0.0
    
    def calculate_throughput(self, operations: int, time_seconds: float) -> float:
        """Calculate throughput in operations per second."""
        throughput = operations / time_seconds if time_seconds > 0 else 0
        self.record_metric("throughput", throughput, "ops/s")
        return throughput
    
    def analyze_state_history(self, state_history: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Analyze simulation state history for performance metrics."""
        if not state_history:
            return {}
        
        # Calculate trajectory metrics
        positions = np.array([state["position"] for state in state_history])
        velocities = np.array([state["velocity"] for state in state_history])
        
        # Distance traveled
        path_length = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
        self.record_metric("path_length", path_length, "meters")
        
        # Average and max speed
        speeds = np.sqrt(np.sum(velocities**2, axis=1))
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        
        self.record_metric("average_speed", avg_speed, "m/s")
        self.record_metric("max_speed", max_speed, "m/s")
        
        return {
            "path_length": path_length,
            "average_speed": avg_speed,
            "max_speed": max_speed
        }
    
    def save_results(self, test_name: str) -> str:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        results = {
            "test_name": test_name,
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration": time.time() - self.start_time,
            "metrics": [metric.to_dict() for metric in self.metrics]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filepath

class HILBenchmark:
    """Benchmark tools for hardware-in-the-loop testing."""
    
    def __init__(self, hil_framework, config: Dict[str, Any] = {}):
        if config is None:
            config = {}
        
        self.hil_framework = hil_framework
        self.performance_benchmark = PerformanceBenchmark(config)
        self.latency_measurements = []
    
    def measure_hardware_latency(self) -> None:
        """Measure latency between sending data to hardware and receiving response."""
        for name, interface in self.hil_framework.hardware_interfaces.items():
            start_time = time.time()
            
            # Send test data
            test_data = {"sensor_data": {"test": np.array([1.0])}}
            interface.send_to_hardware(test_data)
            
            # Wait for response with timeout
            response = None
            timeout = time.time() + 1.0  # 1 second timeout
            
            while response is None and time.time() < timeout:
                response = interface.receive_from_hardware()
                if response is None:
                    time.sleep(0.001)
            
            if response is not None:
                latency = time.time() - start_time
                self.latency_measurements.append(latency)
                self.performance_benchmark.record_metric(f"{name}_latency", latency, "seconds")
    
    def run_benchmark_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a benchmark scenario and collect performance metrics."""
        # Start timing
        start_time = time.time()
        
        # Run the test scenario
        results = self.hil_framework.run_test_scenario(scenario)
        
        # Record execution time
        execution_time = time.time() - start_time
        self.performance_benchmark.record_metric("scenario_execution_time", execution_time, "seconds")
        
        # Measure hardware latency
        self.measure_hardware_latency()
        
        # Measure power usage for each hardware interface
        for name, interface in self.hil_framework.hardware_interfaces.items():
            self.performance_benchmark.measure_power_usage(interface)
        
        # Analyze state history
        trajectory_metrics = self.performance_benchmark.analyze_state_history(results["state_history"])
        
        # Calculate control stability metrics
        if "state_history" in results and len(results["state_history"]) > 1:
            orientations = np.array([state["orientation"] for state in results["state_history"]])
            orientation_stability = np.std(orientations, axis=0)
            
            self.performance_benchmark.record_metric("roll_stability", orientation_stability[0], "rad")
            self.performance_benchmark.record_metric("pitch_stability", orientation_stability[1], "rad")
            self.performance_benchmark.record_metric("yaw_stability", orientation_stability[2], "rad")
        
        # Save benchmark results
        benchmark_file = self.performance_benchmark.save_results(scenario.get("name", "unnamed_scenario"))
        
        # Add benchmark metrics to results
        results["benchmark"] = {
            "execution_time": execution_time,
            "avg_latency": np.mean(self.latency_measurements) if self.latency_measurements else 0,
            "trajectory_metrics": trajectory_metrics,
            "benchmark_file": benchmark_file
        }
        
        return results
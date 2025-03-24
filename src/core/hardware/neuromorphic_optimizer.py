"""
Neuromorphic Hardware Optimizer

Applies real-time adaptive optimization to neuromorphic hardware
based on performance metrics.
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np

from src.core.optimization.adaptive_realtime_optimizer import (
    AdaptiveRealtimeOptimizer, 
    AdaptiveOptimizationConfig,
    OptimizationTarget
)
from src.core.hardware.compatibility_validator import HardwareCompatibilityValidator
from src.core.utils.logging_framework import get_logger

logger = get_logger("neuromorphic_optimizer")

class NeuromorphicHardwareOptimizer:
    """
    Real-time optimizer for neuromorphic hardware configurations.
    Adapts hardware parameters based on performance metrics.
    """
    
    def __init__(self, hardware_type: str, config: Optional[AdaptiveOptimizationConfig] = None):
        """
        Initialize neuromorphic hardware optimizer.
        
        Args:
            hardware_type: Type of neuromorphic hardware (loihi, spinnaker, truenorth)
            config: Optimization configuration
        """
        self.hardware_type = hardware_type
        
        # Create default config if not provided
        if config is None:
            config = AdaptiveOptimizationConfig(
                learning_rate=0.05,
                exploration_rate=0.1,
                memory_size=20,
                adaptation_threshold=0.02,
                update_interval=5.0,  # 5 seconds between updates
                target=OptimizationTarget.BALANCED
            )
            
            # Adjust weights based on hardware type
            if hardware_type == "loihi":
                config.metrics_weights = {
                    "performance": 1.0,
                    "power_efficiency": 0.8,
                    "latency": 0.7,
                    "throughput": 1.0,
                    "reliability": 0.9,
                    "thermal": 0.6
                }
            elif hardware_type == "spinnaker":
                config.metrics_weights = {
                    "performance": 1.0,
                    "power_efficiency": 0.7,
                    "latency": 0.8,
                    "throughput": 1.0,
                    "reliability": 0.9,
                    "thermal": 0.7
                }
            elif hardware_type == "truenorth":
                config.metrics_weights = {
                    "performance": 0.9,
                    "power_efficiency": 1.0,
                    "latency": 0.8,
                    "throughput": 0.7,
                    "reliability": 0.9,
                    "thermal": 0.6
                }
        
        self.config = config
        self.optimizer = AdaptiveRealtimeOptimizer(config)
        self.compatibility_validator = HardwareCompatibilityValidator()
        self.hardware_instances: Dict[str, Any] = {}
        self.hardware_monitors: Dict[str, Any] = {}
        
        # Hardware-specific parameter ranges
        self.parameter_ranges = self._init_parameter_ranges()
        
    def _init_parameter_ranges(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Initialize parameter ranges for different hardware types."""
        ranges = {
            "loihi": {
                "neuron_bias": {"min": -1.0, "max": 1.0},
                "learning_rate": {"min": 0.001, "max": 0.1},
                "spike_threshold": {"min": 0.1, "max": 1.0},
                "refractory_period": {"min": 1, "max": 10},
                "weight_precision": {"min": 1, "max": 8},
                "core_allocation": {"min": 0.1, "max": 1.0},
                "power_mode": {"min": 0.1, "max": 1.0}
            },
            "spinnaker": {
                "neuron_bias": {"min": -1.0, "max": 1.0},
                "learning_rate": {"min": 0.001, "max": 0.1},
                "spike_threshold": {"min": 0.1, "max": 1.0},
                "refractory_period": {"min": 1, "max": 10},
                "routing_algorithm": {"min": 0.1, "max": 1.0},
                "placement_strategy": {"min": 0.1, "max": 1.0},
                "sdram_allocation": {"min": 0.1, "max": 1.0}
            },
            "truenorth": {
                "neuron_bias": {"min": -1.0, "max": 1.0},
                "spike_threshold": {"min": 0.1, "max": 1.0},
                "core_utilization": {"min": 0.1, "max": 1.0},
                "power_mode": {"min": 0.1, "max": 1.0}
            }
        }
        
        return ranges
    
    def register_hardware(self, 
                         hardware_id: str, 
                         hardware_instance: Any,
                         hardware_monitor: Any) -> bool:
        """
        Register hardware instance for optimization.
        
        Args:
            hardware_id: Unique identifier for hardware instance
            hardware_instance: Hardware instance object
            hardware_monitor: Monitor object that provides performance metrics
            
        Returns:
            bool: Success status
        """
        if hardware_id in self.hardware_instances:
            return False
            
        self.hardware_instances[hardware_id] = hardware_instance
        self.hardware_monitors[hardware_id] = hardware_monitor
        
        # Initialize parameters based on hardware type
        initial_parameters = self._get_initial_parameters()
        
        # Register with optimizer
        return self.optimizer.register_system(
            hardware_id,
            hardware_instance,
            initial_parameters,
            lambda: self._get_hardware_metrics(hardware_id),
            lambda params: self._apply_hardware_parameters(hardware_id, params)
        )
    
    def _get_initial_parameters(self) -> Dict[str, float]:
        """Get initial parameters based on hardware type."""
        if self.hardware_type == "loihi":
            return {
                "neuron_bias": 0.0,
                "learning_rate": 0.01,
                "spike_threshold": 0.5,
                "refractory_period": 5.0,
                "weight_precision": 8.0,
                "core_allocation": 0.7,
                "power_mode": 0.8
            }
        elif self.hardware_type == "spinnaker":
            return {
                "neuron_bias": 0.0,
                "learning_rate": 0.01,
                "spike_threshold": 0.5,
                "refractory_period": 5.0,
                "routing_algorithm": 0.5,  # 0=multicast, 1=point-to-point
                "placement_strategy": 0.5,  # 0=balanced, 1=clustered
                "sdram_allocation": 0.7
            }
        elif self.hardware_type == "truenorth":
            return {
                "neuron_bias": 0.0,
                "spike_threshold": 0.5,
                "core_utilization": 0.8,
                "power_mode": 0.7
            }
        else:
            # Generic parameters
            return {
                "neuron_bias": 0.0,
                "learning_rate": 0.01,
                "spike_threshold": 0.5
            }
    
    def _get_hardware_metrics(self, hardware_id: str) -> Dict[str, float]:
        """Get performance metrics from hardware monitor."""
        monitor = self.hardware_monitors.get(hardware_id)
        if not monitor:
            return {}
            
        try:
            # Get raw metrics from monitor
            raw_metrics = monitor.get_metrics()
            
            # Normalize metrics to 0-1 range
            normalized_metrics = {}
            
            # Performance (higher is better)
            if "performance" in raw_metrics:
                normalized_metrics["performance"] = min(1.0, raw_metrics["performance"] / 100.0)
                
            # Power efficiency (higher is better)
            if "power_consumption" in raw_metrics:
                # Invert power consumption (lower is better)
                max_power = 10.0  # Example maximum power in watts
                power_efficiency = 1.0 - min(1.0, raw_metrics["power_consumption"] / max_power)
                normalized_metrics["power_efficiency"] = power_efficiency
                
            # Latency (lower is better)
            if "latency" in raw_metrics:
                # Invert latency (lower is better)
                max_latency = 100.0  # Example maximum latency in ms
                latency_score = 1.0 - min(1.0, raw_metrics["latency"] / max_latency)
                normalized_metrics["latency"] = latency_score
                
            # Throughput (higher is better)
            if "throughput" in raw_metrics:
                max_throughput = 1000.0  # Example maximum throughput
                normalized_metrics["throughput"] = min(1.0, raw_metrics["throughput"] / max_throughput)
                
            # Reliability (higher is better)
            if "error_rate" in raw_metrics:
                # Invert error rate (lower is better)
                reliability = 1.0 - min(1.0, raw_metrics["error_rate"] / 0.1)  # 0.1 = 10% error rate
                normalized_metrics["reliability"] = reliability
                
            # Thermal (lower is better)
            if "temperature" in raw_metrics:
                # Invert temperature (lower is better)
                max_temp = 100.0  # Example maximum temperature in C
                thermal_score = 1.0 - min(1.0, raw_metrics["temperature"] / max_temp)
                normalized_metrics["thermal"] = thermal_score
                
            return normalized_metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics from hardware monitor: {str(e)}")
            return {}
    
    def _apply_hardware_parameters(self, hardware_id: str, parameters: Dict[str, float]) -> None:
        """Apply parameters to hardware instance."""
        hardware = self.hardware_instances.get(hardware_id)
        if not hardware:
            return
            
        # Convert normalized parameters to hardware-specific settings
        hardware_settings = {}
        
        # Apply hardware-specific parameter conversions
        if self.hardware_type == "loihi":
            if "neuron_bias" in parameters:
                hardware_settings["neuron_bias"] = parameters["neuron_bias"]
                
            if "learning_rate" in parameters:
                hardware_settings["learning_rate"] = parameters["learning_rate"]
                
            if "spike_threshold" in parameters:
                hardware_settings["spike_threshold"] = parameters["spike_threshold"]
                
            if "refractory_period" in parameters:
                # Convert to integer number of timesteps
                hardware_settings["refractory_period"] = int(parameters["refractory_period"])
                
            if "weight_precision" in parameters:
                # Convert to integer bits (1-8)
                hardware_settings["weight_precision"] = max(1, min(8, int(parameters["weight_precision"])))
                
            if "core_allocation" in parameters:
                hardware_settings["core_allocation"] = parameters["core_allocation"]
                
            if "power_mode" in parameters:
                # Convert to power mode (0.0-0.3: low, 0.3-0.7: balanced, 0.7-1.0: high)
                if parameters["power_mode"] < 0.3:
                    hardware_settings["power_mode"] = "low"
                elif parameters["power_mode"] < 0.7:
                    hardware_settings["power_mode"] = "balanced"
                else:
                    hardware_settings["power_mode"] = "high"
                    
        elif self.hardware_type == "spinnaker":
            if "neuron_bias" in parameters:
                hardware_settings["neuron_bias"] = parameters["neuron_bias"]
                
            if "learning_rate" in parameters:
                hardware_settings["learning_rate"] = parameters["learning_rate"]
                
            if "spike_threshold" in parameters:
                hardware_settings["spike_threshold"] = parameters["spike_threshold"]
                
            if "refractory_period" in parameters:
                hardware_settings["refractory_period"] = int(parameters["refractory_period"])
                
            if "routing_algorithm" in parameters:
                # Convert to routing algorithm
                if parameters["routing_algorithm"] < 0.5:
                    hardware_settings["routing_algorithm"] = "multicast"
                else:
                    hardware_settings["routing_algorithm"] = "point_to_point"
                    
            if "placement_strategy" in parameters:
                # Convert to placement strategy
                if parameters["placement_strategy"] < 0.5:
                    hardware_settings["placement_strategy"] = "balanced"
                else:
                    hardware_settings["placement_strategy"] = "clustered"
                    
            if "sdram_allocation" in parameters:
                hardware_settings["sdram_allocation"] = parameters["sdram_allocation"]
                
        elif self.hardware_type == "truenorth":
            if "neuron_bias" in parameters:
                hardware_settings["neuron_bias"] = parameters["neuron_bias"]
                
            if "spike_threshold" in parameters:
                hardware_settings["spike_threshold"] = parameters["spike_threshold"]
                
            if "core_utilization" in parameters:
                hardware_settings["core_utilization"] = parameters["core_utilization"]
                
            if "power_mode" in parameters:
                # Convert to power mode
                if parameters["power_mode"] < 0.3:
                    hardware_settings["power_mode"] = "ultra_low"
                elif parameters["power_mode"] < 0.7:
                    hardware_settings["power_mode"] = "standard"
                else:
                    hardware_settings["power_mode"] = "high_performance"
        
        # Apply settings to hardware
        try:
            hardware.apply_settings(hardware_settings)
        except Exception as e:
            logger.error(f"Failed to apply settings to hardware: {str(e)}")
    
    def update(self, hardware_id: str) -> Dict[str, Any]:
        """
        Update optimization for hardware instance.
        
        Args:
            hardware_id: Hardware instance identifier
            
        Returns:
            Dict[str, Any]: Optimization result
        """
        return self.optimizer.update(hardware_id)
    
    def get_optimization_stats(self, hardware_id: str) -> Dict[str, Any]:
        """Get optimization statistics for hardware instance."""
        return self.optimizer.get_optimization_stats(hardware_id)
    
    def reset(self, hardware_id: str, keep_learning: bool = False) -> bool:
        """Reset optimization for hardware instance."""
        return self.optimizer.reset(hardware_id, keep_learning)
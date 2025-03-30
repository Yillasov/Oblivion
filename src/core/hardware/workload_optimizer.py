"""
Workload-Specific Hardware Optimizer

Provides optimization strategies tailored to specific workload types
across different neuromorphic hardware platforms.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

from src.core.utils.logging_framework import get_logger
from src.core.hardware.compatibility_validator import HardwareCompatibilityValidator

logger = get_logger("workload_optimizer")

class WorkloadType(Enum):
    """Common UCAV workload types."""
    SENSOR_FUSION = "sensor_fusion"
    TARGET_TRACKING = "target_tracking"
    NAVIGATION = "navigation"
    THREAT_ASSESSMENT = "threat_assessment"
    STEALTH_OPERATION = "stealth_operation"
    SWARM_COORDINATION = "swarm_coordination"
    MISSION_PLANNING = "mission_planning"
    PAYLOAD_MANAGEMENT = "payload_management"


class WorkloadOptimizer:
    """Optimizes hardware configurations for specific workload types."""
    
    def __init__(self, hardware_type: str):
        """
        Initialize workload optimizer for specific hardware.
        
        Args:
            hardware_type: Target hardware type (loihi, spinnaker, truenorth, etc.)
        """
        self.hardware_type = hardware_type
        self.compatibility_validator = HardwareCompatibilityValidator()
        
        # Initialize optimization strategies
        self._init_optimization_strategies()
        
    def _init_optimization_strategies(self):
        """Initialize hardware-specific optimization strategies for different workloads."""
        # Common optimization strategies
        common_strategies = {
            WorkloadType.SENSOR_FUSION: {
                "priority": "throughput",
                "batch_processing": True,
                "memory_allocation": "dynamic"
            },
            WorkloadType.TARGET_TRACKING: {
                "priority": "latency",
                "batch_processing": False,
                "memory_allocation": "static"
            },
            WorkloadType.NAVIGATION: {
                "priority": "reliability",
                "batch_processing": True,
                "memory_allocation": "static"
            },
            WorkloadType.THREAT_ASSESSMENT: {
                "priority": "latency",
                "batch_processing": False,
                "memory_allocation": "dynamic"
            },
            WorkloadType.STEALTH_OPERATION: {
                "priority": "power_efficiency",
                "batch_processing": True,
                "memory_allocation": "static"
            },
            WorkloadType.SWARM_COORDINATION: {
                "priority": "throughput",
                "batch_processing": True,
                "memory_allocation": "dynamic"
            },
            WorkloadType.MISSION_PLANNING: {
                "priority": "reliability",
                "batch_processing": True,
                "memory_allocation": "static"
            },
            WorkloadType.PAYLOAD_MANAGEMENT: {
                "priority": "latency",
                "batch_processing": False,
                "memory_allocation": "dynamic"
            }
        }
        
        # Hardware-specific optimization strategies
        self.optimization_strategies = {
            "loihi": {
                WorkloadType.SENSOR_FUSION: {
                    **common_strategies[WorkloadType.SENSOR_FUSION],
                    "neuron_model": "LIF",
                    "weight_precision": 8,
                    "core_allocation": "distributed",
                    "learning_rule": "STDP",
                    "spike_encoding": "rate",
                    "hardware_specific": {
                        "phase_encoding": True,
                        "compartment_model": "simple",
                        "axon_delay": "minimal"
                    }
                },
                WorkloadType.TARGET_TRACKING: {
                    **common_strategies[WorkloadType.TARGET_TRACKING],
                    "neuron_model": "ALIF",
                    "weight_precision": 8,
                    "core_allocation": "concentrated",
                    "learning_rule": "Hebbian",
                    "spike_encoding": "temporal",
                    "hardware_specific": {
                        "phase_encoding": False,
                        "compartment_model": "advanced",
                        "axon_delay": "variable"
                    }
                },
                WorkloadType.NAVIGATION: {
                    **common_strategies[WorkloadType.NAVIGATION],
                    "neuron_model": "LIF",
                    "weight_precision": 8,
                    "core_allocation": "distributed",
                    "learning_rule": "STDP",
                    "spike_encoding": "rate",
                    "hardware_specific": {
                        "phase_encoding": True,
                        "compartment_model": "simple",
                        "axon_delay": "minimal"
                    }
                },
                WorkloadType.STEALTH_OPERATION: {
                    **common_strategies[WorkloadType.STEALTH_OPERATION],
                    "neuron_model": "LIF",
                    "weight_precision": 4,  # Lower precision for power efficiency
                    "core_allocation": "concentrated",
                    "learning_rule": "None",  # No learning during stealth
                    "spike_encoding": "sparse",
                    "hardware_specific": {
                        "phase_encoding": False,
                        "compartment_model": "simple",
                        "power_mode": "efficient"
                    }
                }
            },
            "spinnaker": {
                WorkloadType.SENSOR_FUSION: {
                    **common_strategies[WorkloadType.SENSOR_FUSION],
                    "neuron_model": "IF",
                    "weight_precision": 16,
                    "core_allocation": "distributed",
                    "learning_rule": "STDP",
                    "spike_encoding": "rate",
                    "hardware_specific": {
                        "routing_algorithm": "multicast",
                        "placement_strategy": "balanced",
                        "sdram_allocation": "generous"
                    }
                },
                WorkloadType.SWARM_COORDINATION: {
                    **common_strategies[WorkloadType.SWARM_COORDINATION],
                    "neuron_model": "IF",
                    "weight_precision": 16,
                    "core_allocation": "distributed",
                    "learning_rule": "BCM",
                    "spike_encoding": "rate",
                    "hardware_specific": {
                        "routing_algorithm": "multicast",
                        "placement_strategy": "balanced",
                        "sdram_allocation": "generous",
                        "communication_optimization": "high_priority"
                    }
                },
                WorkloadType.MISSION_PLANNING: {
                    **common_strategies[WorkloadType.MISSION_PLANNING],
                    "neuron_model": "IZH",
                    "weight_precision": 16,
                    "core_allocation": "concentrated",
                    "learning_rule": "Reinforcement",
                    "spike_encoding": "temporal",
                    "hardware_specific": {
                        "routing_algorithm": "point_to_point",
                        "placement_strategy": "clustered",
                        "sdram_allocation": "generous"
                    }
                }
            },
            "truenorth": {
                WorkloadType.TARGET_TRACKING: {
                    **common_strategies[WorkloadType.TARGET_TRACKING],
                    "neuron_model": "TrueNorthLIF",
                    "weight_precision": 1,  # Binary weights
                    "core_allocation": "concentrated",
                    "learning_rule": "Offline",
                    "spike_encoding": "binary",
                    "hardware_specific": {
                        "core_utilization": "maximized",
                        "crossbar_optimization": True,
                        "power_mode": "ultra_low"
                    }
                },
                WorkloadType.THREAT_ASSESSMENT: {
                    **common_strategies[WorkloadType.THREAT_ASSESSMENT],
                    "neuron_model": "TrueNorthLIF",
                    "weight_precision": 1,
                    "core_allocation": "distributed",
                    "learning_rule": "Offline",
                    "spike_encoding": "binary",
                    "hardware_specific": {
                        "core_utilization": "balanced",
                        "crossbar_optimization": True,
                        "power_mode": "standard"
                    }
                },
                WorkloadType.STEALTH_OPERATION: {
                    **common_strategies[WorkloadType.STEALTH_OPERATION],
                    "neuron_model": "TrueNorthLIF",
                    "weight_precision": 1,
                    "core_allocation": "minimal",
                    "learning_rule": "Offline",
                    "spike_encoding": "sparse_binary",
                    "hardware_specific": {
                        "core_utilization": "minimal",
                        "crossbar_optimization": True,
                        "power_mode": "ultra_low"
                    }
                }
            }
        }
    
    def get_optimization_strategy(self, workload_type: WorkloadType) -> Optional[Dict[str, Any]]:
        """
        Get optimization strategy for a specific workload type.
        
        Args:
            workload_type: Type of workload
            
        Returns:
            Optional[Dict[str, Any]]: Optimization strategy or None if not available
        """
        hardware_strategies = self.optimization_strategies.get(self.hardware_type, {})
        return hardware_strategies.get(workload_type)
    
    def optimize_config(self, config: Dict[str, Any], workload_type: WorkloadType) -> Dict[str, Any]:
        """
        Optimize hardware configuration for a specific workload.
        
        Args:
            config: Hardware configuration to optimize
            workload_type: Target workload type
            
        Returns:
            Dict[str, Any]: Optimized configuration
        """
        # Get optimization strategy
        strategy = self.get_optimization_strategy(workload_type)
        if not strategy:
            logger.warning(f"No optimization strategy for {workload_type.value} on {self.hardware_type}")
            return config
        
        # Create a copy of the configuration
        optimized_config = config.copy()
        
        # Apply general optimizations
        optimized_config["priority"] = strategy["priority"]
        
        # Apply neuron model optimization if applicable
        if "neuron_params" not in optimized_config:
            optimized_config["neuron_params"] = {}
        
        optimized_config["neuron_params"]["type"] = strategy["neuron_model"]
        optimized_config["neuron_params"]["weight_precision"] = strategy["weight_precision"]
        
        # Apply learning rule optimization
        if strategy["learning_rule"] != "None":
            optimized_config["learning"] = {
                "enabled": True,
                "rule": strategy["learning_rule"]
            }
        else:
            optimized_config["learning"] = {"enabled": False}
        
        # Apply spike encoding optimization
        optimized_config["encoding"] = {"scheme": strategy["spike_encoding"]}
        
        # Apply hardware-specific optimizations
        if "hardware_specific" in strategy:
            if "hardware_specific" not in optimized_config:
                optimized_config["hardware_specific"] = {}
            
            for key, value in strategy["hardware_specific"].items():
                optimized_config["hardware_specific"][key] = value
        
        # Add workload metadata
        if "_metadata" not in optimized_config:
            optimized_config["_metadata"] = {}
        
        optimized_config["_metadata"]["optimized_for"] = workload_type.value
        
        # Validate the optimized configuration
        is_valid, issues = self.compatibility_validator.validate_compatibility(
            self.hardware_type, optimized_config
        )
        
        if not is_valid:
            logger.warning(f"Optimized configuration has compatibility issues: {issues}")
            # Apply fixes for common issues
            optimized_config = self._fix_compatibility_issues(optimized_config, issues)
        
        return optimized_config
    
    def _fix_compatibility_issues(self, config: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
        """
        Fix common compatibility issues in the configuration.
        
        Args:
            config: Configuration with issues
            issues: List of compatibility issues
            
        Returns:
            Dict[str, Any]: Fixed configuration
        """
        if not isinstance(config, dict):
            logger.error("Invalid configuration format")
            return {}
            
        fixed_config = config.copy()
        
        # Ensure neuron_params exists to avoid KeyError
        if "neuron_params" not in fixed_config:
            fixed_config["neuron_params"] = {}
        
        for issue in issues:
            # Fix weight precision issues
            if "weight_precision" in issue:
                if self.hardware_type == "truenorth" and fixed_config.get("neuron_params", {}).get("weight_precision", 0) > 1:
                    logger.info("Fixing TrueNorth weight precision (setting to 1)")
                    fixed_config["neuron_params"]["weight_precision"] = 1
                elif self.hardware_type == "loihi" and fixed_config.get("neuron_params", {}).get("weight_precision", 0) > 8:
                    logger.info("Fixing Loihi weight precision (setting to 8)")
                    fixed_config["neuron_params"]["weight_precision"] = 8
            
            # Fix neuron model issues
            if "neuron type" in issue:
                if self.hardware_type == "loihi":
                    logger.info("Fixing neuron model for Loihi (setting to LIF)")
                    fixed_config["neuron_params"]["type"] = "LIF"
                elif self.hardware_type == "spinnaker":
                    logger.info("Fixing neuron model for SpiNNaker (setting to IF)")
                    fixed_config["neuron_params"]["type"] = "IF"
                elif self.hardware_type == "truenorth":
                    logger.info("Fixing neuron model for TrueNorth (setting to TrueNorthLIF)")
                    fixed_config["neuron_params"]["type"] = "TrueNorthLIF"
        
        return fixed_config
    
    def get_available_workload_types(self) -> List[WorkloadType]:
        """
        Get available workload types for the current hardware.
        
        Returns:
            List[WorkloadType]: Available workload types
        """
        hardware_strategies = self.optimization_strategies.get(self.hardware_type, {})
        return list(hardware_strategies.keys())
    
    def analyze_workload(self, metrics: Dict[str, Any]) -> Tuple[WorkloadType, float]:
        """
        Analyze metrics to determine the most likely workload type.
        
        Args:
            metrics: Performance and operational metrics
            
        Returns:
            Tuple[WorkloadType, float]: Most likely workload type and confidence score
        """
        # Simple heuristic-based workload detection
        scores = {}
        
        # Check sensor activity
        sensor_activity = metrics.get("sensor_activity", 0.0)
        if sensor_activity > 0.7:
            scores[WorkloadType.SENSOR_FUSION] = 0.6 + (sensor_activity - 0.7) * 0.5
            scores[WorkloadType.TARGET_TRACKING] = 0.5 + (sensor_activity - 0.7) * 0.5
        
        # Check navigation metrics
        nav_complexity = metrics.get("navigation_complexity", 0.0)
        if nav_complexity > 0.5:
            scores[WorkloadType.NAVIGATION] = 0.4 + nav_complexity * 0.6
        
        # Check threat metrics
        threat_level = metrics.get("threat_level", 0.0)
        if threat_level > 0.3:
            scores[WorkloadType.THREAT_ASSESSMENT] = 0.3 + threat_level * 0.7
        
        # Check stealth requirements
        stealth_requirement = metrics.get("stealth_requirement", 0.0)
        if stealth_requirement > 0.6:
            scores[WorkloadType.STEALTH_OPERATION] = 0.5 + (stealth_requirement - 0.6) * 0.8
        
        # Check swarm activity
        swarm_activity = metrics.get("swarm_activity", 0.0)
        if swarm_activity > 0.4:
            scores[WorkloadType.SWARM_COORDINATION] = 0.4 + swarm_activity * 0.6
        
        # Check mission complexity
        mission_complexity = metrics.get("mission_complexity", 0.0)
        if mission_complexity > 0.5:
            scores[WorkloadType.MISSION_PLANNING] = 0.3 + mission_complexity * 0.7
        
        # Check payload activity
        payload_activity = metrics.get("payload_activity", 0.0)
        if payload_activity > 0.3:
            scores[WorkloadType.PAYLOAD_MANAGEMENT] = 0.3 + payload_activity * 0.7
        
        # Find workload type with highest score
        if not scores:
            # Default to sensor fusion if no clear pattern
            return WorkloadType.SENSOR_FUSION, 0.5
        
        best_workload = max(scores.items(), key=lambda x: x[1])
        return best_workload[0], best_workload[1]
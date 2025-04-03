"""
Navigation System Testing Framework for UCAV platforms.

Provides tools for testing and validating navigation systems
under various simulated conditions.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time

from src.navigation.base import NavigationSystem, NavigationSpecs
from src.navigation.error_handling import safe_navigation_operation
from src.navigation.integration import NavigationIntegrator

# Configure logger
logger = logging.getLogger(__name__)


class TestScenario(Enum):
    """Test scenarios for navigation systems."""
    NORMAL_OPERATION = "normal_operation"
    GPS_DENIED = "gps_denied"
    HIGH_INTERFERENCE = "high_interference"
    RAPID_MANEUVER = "rapid_maneuver"
    LOW_VISIBILITY = "low_visibility"
    SYSTEM_FAILURE = "system_failure"


@dataclass
class TestParameters:
    """Parameters for navigation system tests."""
    duration: float = 60.0  # seconds
    update_rate: float = 10.0  # Hz
    position_tolerance: float = 5.0  # meters
    orientation_tolerance: float = 0.1  # radians
    record_metrics: bool = True
    inject_errors: bool = False
    error_probability: float = 0.05


class NavigationTestFramework:
    """
    Framework for testing navigation systems.
    
    Provides tools for simulating various conditions and
    validating navigation system performance.
    """
    
    def __init__(self):
        """Initialize the navigation testing framework."""
        self.integrator = NavigationIntegrator()
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.current_scenario: Optional[TestScenario] = None
        self.current_parameters: Optional[TestParameters] = None
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.sdk_bridge = None  # SDK bridge will be set by connect_sdk
        
        logger.info("Navigation testing framework initialized")
    
    def register_system(self, system_id: str, system: NavigationSystem) -> bool:
        """
        Register a navigation system for testing.
        
        Args:
            system_id: Identifier for the system
            system: Navigation system instance
            
        Returns:
            Success status
        """
        return self.integrator.register_navigation_system(system_id, system)
    
    def setup_test(self, scenario: TestScenario, parameters: TestParameters) -> None:
        """
        Set up a test scenario.
        
        Args:
            scenario: Test scenario to run
            parameters: Test parameters
        """
        self.current_scenario = scenario
        self.current_parameters = parameters
        
        # Initialize metrics history for this test
        for system_id in self.integrator.navigation_systems:
            if system_id not in self.metrics_history:
                self.metrics_history[system_id] = []
        
        logger.info(f"Set up test scenario: {scenario.value}")
    
    def connect_sdk(self, sdk_bridge) -> bool:
        """
        Connect to SDK bridge for cross-system integration.
        
        Args:
            sdk_bridge: SDK bridge instance
            
        Returns:
            Success status
        """
        self.sdk_bridge = sdk_bridge
        return True
    
    @safe_navigation_operation
    def run_test(self) -> Dict[str, Any]:
        """
        Run the current test scenario.
        
        Returns:
            Test results
        """
        if not self.current_scenario or not self.current_parameters:
            return {"error": "No test scenario configured"}
        
        # Initialize all systems
        try:
            self.integrator.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize navigation systems: {str(e)}")
            return {"error": f"Initialization failed: {str(e)}", "success": False}
        
        # Activate all systems
        active_systems = 0
        for system_id in self.integrator.navigation_systems:
            try:
                if self.integrator.activate_system(system_id):
                    active_systems += 1
            except Exception as e:
                logger.error(f"Failed to activate system {system_id}: {str(e)}")
        
        if active_systems == 0:
            return {"error": "No navigation systems could be activated", "success": False}
        
        # Generate test environment based on scenario
        environment = self._generate_test_environment()
        
        # Notify SDK about test start if connected
        if self.sdk_bridge:
            self.sdk_bridge.notify_test_start(
                self.current_scenario, 
                {
                    "duration": self.current_parameters.duration,
                    "update_rate": self.current_parameters.update_rate,
                    "position_tolerance": self.current_parameters.position_tolerance
                }
            )
        
        # Run test for specified duration
        start_time = time.time()
        end_time = start_time + self.current_parameters.duration
        
        update_interval = 1.0 / self.current_parameters.update_rate
        next_update = start_time
        
        results = {
            "scenario": self.current_scenario.value,
            "systems": {},
            "overall": {
                "success": True,
                "errors": 0,
                "max_position_error": 0.0,
                "avg_position_error": 0.0
            }
        }
        
        # Test execution loop
        while time.time() < end_time:
            current_time = time.time()
            
            if current_time >= next_update:
                # Update environment with timeout protection
                try:
                    environment = self._update_environment(environment, current_time - start_time)
                except Exception as e:
                    logger.error(f"Environment update failed: {str(e)}")
                    results["overall"]["errors"] += 1
                    # Use last known good environment
                
                # Inject errors if configured
                if self.current_parameters.inject_errors:
                    environment = self._inject_errors(environment)
                
                # Update each system
                for system_id, system in self.integrator.navigation_systems.items():
                    delta_time = update_interval
                    
                    try:
                        # Check if system has update method, otherwise use a generic approach
                        if hasattr(system, 'update') and callable(getattr(system, 'update')):
                            system_result = system.update(delta_time, environment)
                        else:
                            # Fallback for systems without update method
                            logger.warning(f"System {system_id} doesn't have update method, using generic update")
                            system_result = self._generic_system_update(system, delta_time, environment)
                        
                        # Record metrics
                        if self.current_parameters.record_metrics:
                            self._record_metrics(system_id, system, environment, system_result)
                            
                    except Exception as e:
                        logger.error(f"Error updating system {system_id}: {str(e)}")
                        results["overall"]["errors"] += 1
                        results["overall"]["success"] = False
                
                next_update = current_time + update_interval
        
        # Calculate final results
        for system_id, system in self.integrator.navigation_systems.items():
            system_metrics = self._calculate_system_metrics(system_id)
            results["systems"][system_id] = system_metrics
            
            # Update overall results
            results["overall"]["max_position_error"] = max(
                results["overall"]["max_position_error"],
                system_metrics.get("max_position_error", 0.0)
            )
            
            # Add to overall average
            results["overall"]["avg_position_error"] += system_metrics.get("avg_position_error", 0.0)
        
        # Finalize overall average
        if self.integrator.navigation_systems:
            results["overall"]["avg_position_error"] /= len(self.integrator.navigation_systems)
        
        # Store results
        self.test_results[self.current_scenario.value] = results
        
        logger.info(f"Completed test scenario: {self.current_scenario.value}")
        return results
    
    def _generate_test_environment(self) -> Dict[str, Any]:
        """Generate initial test environment based on scenario."""
        environment = {
            "timestamp": time.time(),
            "true_position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "true_orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "true_velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
            "atmospheric_conditions": {
                "pressure": 101325.0,  # Pa (sea level)
                "temperature": 288.15,  # K (15°C)
                "humidity": 0.5,
                "wind": {"speed": 0.0, "direction": 0.0}
            },
            "celestial_observations": [],
            "magnetic_field": {"x": 20000.0, "y": 0.0, "z": 40000.0},  # nT
            "gravity": {"x": 0.0, "y": 0.0, "z": -9.81}  # m/s²
        }
        
        # Customize environment based on scenario
        if self.current_scenario == TestScenario.GPS_DENIED:
            environment["gps_available"] = False
        elif self.current_scenario == TestScenario.HIGH_INTERFERENCE:
            environment["magnetic_interference"] = 10000.0  # nT
            environment["radio_interference"] = 0.8  # 0-1 scale
        elif self.current_scenario == TestScenario.LOW_VISIBILITY:
            environment["visibility"] = 0.2  # 0-1 scale
            environment["cloud_cover"] = 0.9  # 0-1 scale
        
        return environment
    
    def _update_environment(self, environment: Dict[str, Any], elapsed_time: float) -> Dict[str, Any]:
        """Update environment based on elapsed time and scenario."""
        # Update timestamp
        environment["timestamp"] = time.time()
        
        # Basic movement pattern (circular path)
        radius = 100.0  # meters
        angular_velocity = 0.1  # rad/s
        angle = angular_velocity * elapsed_time
        
        # Update true position
        environment["true_position"] = {
            "x": radius * np.cos(angle),
            "y": radius * np.sin(angle),
            "z": 100.0  # constant altitude
        }
        
        # Update true orientation
        environment["true_orientation"] = {
            "roll": 0.1 * np.sin(elapsed_time * 0.2),
            "pitch": 0.05 * np.sin(elapsed_time * 0.3),
            "yaw": angle + np.pi/2  # facing direction of travel
        }
        
        # Update true velocity
        environment["true_velocity"] = {
            "vx": -radius * angular_velocity * np.sin(angle),
            "vy": radius * angular_velocity * np.cos(angle),
            "vz": 0.0
        }
        
        # Scenario-specific updates
        if self.current_scenario == TestScenario.RAPID_MANEUVER and elapsed_time % 10 < 2:
            # Sudden maneuver every 10 seconds
            environment["true_orientation"]["roll"] += 0.5 * np.sin(elapsed_time * 2)
            environment["true_orientation"]["pitch"] += 0.3 * np.cos(elapsed_time * 2)
            
        elif self.current_scenario == TestScenario.SYSTEM_FAILURE and elapsed_time > 30:
            # Simulate system failure after 30 seconds
            if "system_failures" not in environment:
                environment["system_failures"] = []
            
            # Add random failures
            if np.random.random() < 0.1 and len(environment["system_failures"]) < 3:
                failure_types = ["sensor", "power", "computation", "memory"]
                environment["system_failures"].append({
                    "type": np.random.choice(failure_types),
                    "severity": np.random.random(),
                    "time": elapsed_time
                })
        
        return environment
    
    def _inject_errors(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Inject random errors into the environment data."""
        if self.current_parameters and np.random.random() < self.current_parameters.error_probability:
            error_type = np.random.choice(["position", "orientation", "sensor"])
            
            if error_type == "position":
                # Add position error
                for coord in ["x", "y", "z"]:
                    environment["true_position"][coord] += np.random.normal(0, 5.0)
                    
            elif error_type == "orientation":
                # Add orientation error
                for angle in ["roll", "pitch", "yaw"]:
                    environment["true_orientation"][angle] += np.random.normal(0, 0.1)
                    
            elif error_type == "sensor":
                # Add sensor error or dropout
                if "sensor_errors" not in environment:
                    environment["sensor_errors"] = []
                    
                environment["sensor_errors"].append({
                    "sensor": np.random.choice(["imu", "magnetometer", "barometer", "camera"]),
                    "magnitude": np.random.random(),
                    "duration": np.random.uniform(0.1, 2.0)
                })
        
        return environment
    
    def _record_metrics(self, system_id: str, system: NavigationSystem, 
                      environment: Dict[str, Any], system_result: Dict[str, Any]) -> None:
        """Record system metrics during test."""
        # Safely get system position
        if hasattr(system, 'get_position') and callable(getattr(system, 'get_position')):
            try:
                system_position = system.get_position()
            except Exception as e:
                logger.warning(f"Error getting position from system {system_id}: {str(e)}")
                system_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        else:
            # If system doesn't have get_position method, try to extract from result
            if isinstance(system_result, dict) and "position" in system_result:
                system_position = system_result["position"]
            else:
                logger.warning(f"System {system_id} doesn't provide position data")
                system_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # Calculate position error
        true_position = environment["true_position"]
        position_error = np.sqrt(
            (system_position.get("x", 0.0) - true_position["x"])**2 +
            (system_position.get("y", 0.0) - true_position["y"])**2 +
            (system_position.get("z", 0.0) - true_position["z"])**2
        )
        
        # Record metrics
        metrics = {
            "timestamp": environment["timestamp"],
            "position_error": position_error,
            "system_result": system_result,
            "environment": {
                "true_position": true_position,
                "true_orientation": environment["true_orientation"],
                "true_velocity": environment["true_velocity"]
            }
        }
        
        self.metrics_history[system_id].append(metrics)
    
    def _calculate_system_metrics(self, system_id: str) -> Dict[str, Any]:
        """Calculate overall metrics for a system based on recorded history."""
        if system_id not in self.metrics_history or not self.metrics_history[system_id]:
            return {"error": "No metrics recorded"}
            
        metrics = self.metrics_history[system_id]
        
        # Calculate position error statistics
        position_errors = [m["position_error"] for m in metrics]
        
        return {
            "max_position_error": max(position_errors),
            "min_position_error": min(position_errors),
            "avg_position_error": sum(position_errors) / len(position_errors),
            "std_position_error": np.std(position_errors),
            "within_tolerance": all(e <= (self.current_parameters.position_tolerance if self.current_parameters else float('inf')) for e in position_errors),
            "metrics_count": len(metrics)
        }
    
    def get_test_results(self, scenario: Optional[TestScenario] = None) -> Dict[str, Any]:
        """
        Get test results for a specific scenario or all scenarios.
        
        Args:
            scenario: Optional specific scenario to get results for
            
        Returns:
            Test results
        """
        if scenario:
            return self.test_results.get(scenario.value, {"error": "No results for scenario"})
        return self.test_results
    
    def compare_systems(self, metric: str = "avg_position_error") -> Dict[str, Any]:
        """
        Compare all systems based on a specific metric.
        
        Args:
            metric: Metric to compare systems by
            
        Returns:
            Comparison results
        """
        if not self.test_results:
            return {"error": "No test results available"}
            
        comparison = {}
        
        for scenario, results in self.test_results.items():
            if "systems" not in results:
                continue
                
            scenario_comparison = {}
            
            for system_id, system_results in results["systems"].items():
                if metric in system_results:
                    scenario_comparison[system_id] = system_results[metric]
            
            if scenario_comparison:
                # Find best system for this scenario
                best_system = min(scenario_comparison.items(), key=lambda x: x[1])
                
                comparison[scenario] = {
                    "values": scenario_comparison,
                    "best_system": best_system[0],
                    "best_value": best_system[1]
                }
        
        return comparison

    def _generic_system_update(self, system: NavigationSystem, delta_time: float, 
                             environment: Dict[str, Any]) -> Dict[str, Any]:
        """Generic update method for systems without an update method."""
        # Get current status
        status = system.get_status() if hasattr(system, 'get_status') and callable(getattr(system, 'get_status')) else {}
        
        # Try to update position if system is a position provider
        if hasattr(system, 'get_position') and callable(getattr(system, 'get_position')):
            position = system.get_position()
        else:
            position = {"x": 0.0, "y": 0.0, "z": 0.0}
            
        return {
            "status": status,
            "position": position,
            "delta_time": delta_time
        }

class CrossSystemIntegration:
    """Integration with other subsystems for comprehensive testing."""
    
    def __init__(self, test_framework: NavigationTestFramework):
        """Initialize cross-system integration."""
        self.test_framework = test_framework
        self.messaging = None
        self.hardware_optimizer = None
        self.power_system = None
        
    def connect_messaging(self, messaging_system: Any) -> bool:
        """Connect to messaging system."""
        self.messaging = messaging_system
        return True
        
    def connect_hardware_optimizer(self, optimizer: Any) -> bool:
        """Connect to hardware switching optimizer."""
        self.hardware_optimizer = optimizer
        return True
        
    def connect_power_system(self, power_system: Any) -> bool:
        """Connect to power management system."""
        self.power_system = power_system
        return True
        
    def notify_test_start(self, scenario: TestScenario) -> None:
        """Notify connected systems about test start."""
        if self.messaging:
            self.messaging.send_message("system_manager", {
                "event": "test_start",
                "scenario": scenario.value,
                "timestamp": time.time()
            })
            
    def notify_test_complete(self, results: Dict[str, Any]) -> None:
        """Notify connected systems about test completion."""
        if self.messaging:
            self.messaging.send_message("system_manager", {
                "event": "test_complete",
                "results": results,
                "timestamp": time.time()
            })
            
    def request_hardware_optimization(self, scenario: TestScenario) -> Optional[str]:
        """Request optimal hardware for scenario."""
        if self.hardware_optimizer:
            context = {
                "scenario": scenario.value,
                "workload_type": "navigation_testing",
                "priority": "high"
            }
            result = self.hardware_optimizer.optimize(context)
            return result.get("hardware_type")
        return None
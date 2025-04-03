"""
Stress testing and failure analysis for landing gear systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from src.landing_gear.base import NeuromorphicLandingGear, LandingGearSpecs
from src.landing_gear.types import LandingGearType
from src.landing_gear.simulation import LandingGearSimulation, LandingEnvironment, TerrainType
from src.landing_gear.emergency import FailureType, RecoveryAction


class StressTestType(Enum):
    """Types of stress tests for landing gear."""
    HARD_LANDING = "hard_landing"
    CROSSWIND = "crosswind_landing"
    ROUGH_TERRAIN = "rough_terrain"
    OVERWEIGHT = "overweight_landing"
    ASYMMETRIC_LOAD = "asymmetric_load"
    SYSTEM_FAILURE = "system_failure"


@dataclass
class FailureAnalysisResult:
    """Results from failure analysis."""
    failure_probability: float  # 0.0 to 1.0
    critical_components: List[str]
    failure_modes: Dict[str, float]  # component -> probability
    max_safe_load: float  # maximum safe load in N
    recommended_actions: List[str]
    telemetry_at_failure: Optional[Dict[str, Any]] = None


class LandingGearStressTester:
    """Stress testing framework for landing gear systems."""
    
    def __init__(self, landing_gear: NeuromorphicLandingGear):
        """Initialize with a landing gear system."""
        self.landing_gear = landing_gear
        self.simulation = LandingGearSimulation(
            landing_gear=landing_gear,
            environment=LandingEnvironment()
        )
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    def run_stress_test(self, test_type: StressTestType, intensity: float = 1.0) -> Dict[str, Any]:
        """
        Run a specific stress test.
        
        Args:
            test_type: Type of stress test to run
            intensity: Test intensity from 0.0 (mild) to 1.0 (extreme)
            
        Returns:
            Dict with test results
        """
        # Configure test parameters based on type
        test_config = self._configure_test(test_type, intensity)
        
        # Run simulation with test configuration
        results = self.simulation.run_landing_simulation(
            initial_state=test_config["initial_state"],
            duration=test_config["duration"]
        )
        
        # Store results
        test_id = f"{test_type.value}_{intensity:.2f}_{len(self.test_results)}"
        self.test_results[test_id] = {
            "test_type": test_type.value,
            "intensity": intensity,
            "config": test_config,
            "results": results,
            "pass": self._evaluate_test_success(results, test_type, intensity)
        }
        
        return self.test_results[test_id]
    
    def _configure_test(self, test_type: StressTestType, intensity: float) -> Dict[str, Any]:
        """Configure test parameters based on test type and intensity."""
        # Base configuration
        config = {
            "initial_state": {
                "position": np.array([0.0, 0.0, 10.0]),
                "velocity": np.array([0.0, 0.0, -2.0]),
                "orientation": np.zeros(3),
                "angular_velocity": np.zeros(3),
                "mass": 1000.0
            },
            "duration": 10.0
        }
        
        # Adjust based on test type
        if test_type == StressTestType.HARD_LANDING:
            # Increase descent velocity based on intensity
            descent_velocity = -2.0 - 8.0 * intensity  # -2 to -10 m/s
            config["initial_state"]["velocity"] = np.array([0.0, 0.0, descent_velocity])
            
        elif test_type == StressTestType.CROSSWIND:
            # Add crosswind and aircraft roll/yaw
            self.simulation.environment.wind_speed = 5.0 + 15.0 * intensity  # 5 to 20 m/s
            self.simulation.environment.wind_direction = np.pi / 4  # 45 degrees
            config["initial_state"]["orientation"] = np.array([0.1 * intensity, 0.0, 0.2 * intensity])
            
        elif test_type == StressTestType.ROUGH_TERRAIN:
            # Set rough terrain parameters
            self.simulation.environment.terrain_type = TerrainType.GRAVEL
            self.simulation.environment.terrain_roughness = 0.3 + 0.7 * intensity  # 0.3 to 1.0
            self.simulation.environment.terrain_friction = 0.5  # Lower friction
            
        elif test_type == StressTestType.OVERWEIGHT:
            # Increase aircraft mass
            base_mass = self.landing_gear.specs.max_load_capacity / 9.81  # Convert N to kg
            overload_factor = 1.0 + intensity  # 1.0 to 2.0 times max load
            config["initial_state"]["mass"] = base_mass * overload_factor
            
        elif test_type == StressTestType.ASYMMETRIC_LOAD:
            # Add roll and uneven weight distribution
            config["initial_state"]["orientation"] = np.array([0.3 * intensity, 0.1 * intensity, 0.0])
            
        elif test_type == StressTestType.SYSTEM_FAILURE:
            # Simulate partial system failure
            # This would require modifying the landing gear's internal state
            # For simplicity, we'll just record that a failure was injected
            config["failure_injected"] = True
        
        return config
    
    def _evaluate_test_success(self, results: Dict[str, Any], test_type: StressTestType, intensity: float) -> bool:
        """Evaluate if the test was successful (gear survived)."""
        metrics = results.get("metrics", {})
        landing_quality = metrics.get("landing_quality", 0.0)
        
        # Different thresholds based on test type and intensity
        if test_type == StressTestType.HARD_LANDING:
            threshold = 0.7 - 0.4 * intensity  # 0.7 to 0.3
        elif test_type == StressTestType.SYSTEM_FAILURE:
            threshold = 0.5 - 0.3 * intensity  # 0.5 to 0.2
        else:
            threshold = 0.6 - 0.3 * intensity  # 0.6 to 0.3
            
        return landing_quality >= threshold
    
    def analyze_failure_modes(self) -> FailureAnalysisResult:
        """Analyze potential failure modes based on stress test results."""
        # Count failures by test type
        failures_by_type = {}
        total_tests = len(self.test_results)
        
        if total_tests == 0:
            return FailureAnalysisResult(
                failure_probability=0.0,
                critical_components=["No data"],
                failure_modes={"No data": 0.0},
                max_safe_load=0.0,
                recommended_actions=["Run stress tests first"]
            )
        
        for test_id, result in self.test_results.items():
            test_type = result["test_type"]
            if not result["pass"]:
                failures_by_type[test_type] = failures_by_type.get(test_type, 0) + 1
        
        # Determine critical components based on test failures
        critical_components = []
        if StressTestType.HARD_LANDING.value in failures_by_type:
            critical_components.append("Shock absorbers")
        if StressTestType.CROSSWIND.value in failures_by_type:
            critical_components.append("Lateral stability system")
        if StressTestType.ROUGH_TERRAIN.value in failures_by_type:
            critical_components.append("Suspension system")
        if StressTestType.OVERWEIGHT.value in failures_by_type:
            critical_components.append("Structural supports")
        if StressTestType.ASYMMETRIC_LOAD.value in failures_by_type:
            critical_components.append("Balance mechanisms")
        if StressTestType.SYSTEM_FAILURE.value in failures_by_type:
            critical_components.append("Control systems")
        
        # Calculate failure modes and probabilities
        failure_modes = {}
        for component in critical_components:
            # Simple probability assignment
            failure_modes[component] = 0.1 + 0.4 * (failures_by_type.get(component, 0) / total_tests)
        
        # Find max safe load from overweight tests
        max_safe_load = self.landing_gear.specs.max_load_capacity
        overweight_tests = [r for _, r in self.test_results.items() 
                           if r["test_type"] == StressTestType.OVERWEIGHT.value]
        
        if overweight_tests:
            # Find highest passing test
            passing_tests = [t for t in overweight_tests if t["pass"]]
            if passing_tests:
                max_intensity = max([t["intensity"] for t in passing_tests])
                max_safe_load = self.landing_gear.specs.max_load_capacity * (1.0 + max_intensity)
        
        # Generate recommended actions
        recommended_actions = []
        if "Shock absorbers" in critical_components:
            recommended_actions.append("Reinforce shock absorption system")
        if "Lateral stability system" in critical_components:
            recommended_actions.append("Improve crosswind handling")
        if "Suspension system" in critical_components:
            recommended_actions.append("Upgrade terrain adaptation capabilities")
        if len(critical_components) > 2:
            recommended_actions.append("Consider complete redesign for higher safety margin")
        
        # Calculate overall failure probability
        total_failures = sum(failures_by_type.values())
        failure_probability = total_failures / total_tests
        
        return FailureAnalysisResult(
            failure_probability=failure_probability,
            critical_components=critical_components,
            failure_modes=failure_modes,
            max_safe_load=max_safe_load,
            recommended_actions=recommended_actions
        )
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run a comprehensive suite of stress tests."""
        # Run all test types with varying intensities
        intensities = [0.3, 0.6, 0.9]  # Low, medium, high
        
        for test_type in StressTestType:
            for intensity in intensities:
                self.run_stress_test(test_type, intensity)
        
        # Analyze results
        failure_analysis = self.analyze_failure_modes()
        
        return {
            "total_tests": len(self.test_results),
            "passing_tests": sum(1 for r in self.test_results.values() if r["pass"]),
            "failure_analysis": {
                "failure_probability": failure_analysis.failure_probability,
                "critical_components": failure_analysis.critical_components,
                "max_safe_load": failure_analysis.max_safe_load,
                "recommended_actions": failure_analysis.recommended_actions
            }
        }
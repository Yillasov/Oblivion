"""
UCAV Prototype Testing Framework

Provides testing capabilities for UCAV prototypes.
"""

import time
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import json

from src.core.utils.logging_framework import get_logger
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.manufacturing.quality.quality_inspector import QualityInspector
from src.simulation.models.ucav_geometry import UCAVGeometry
from src.manufacturing.testing.prototype_hil import PrototypeHILTestSuite, PrototypeHILTest
from src.core.utils.error_handling import handle_errors

logger = get_logger("prototype_testing")

class PrototypeTester:
    """UCAV prototype testing framework."""
    
    def __init__(self):
        """Initialize prototype tester."""
        self.system = NeuromorphicSystem()
        self.inspector = QualityInspector()
        self.test_results = {}
        self.hardware_interfaces = {}
        
    def add_hardware_interface(self, name: str, interface: Any) -> None:
        """Add hardware interface for HIL testing."""
        self.hardware_interfaces[name] = interface
        
    def run_prototype_tests(self, prototype: UCAVGeometry, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run standard prototype tests.
        
        Args:
            prototype: UCAV prototype geometry
            config: Test configuration
            
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Running prototype tests")
        
        results = {}
        
        # Run aerodynamic tests
        if "wind_tunnel_settings" in config:
            results["aero_testing"] = self._run_aero_tests(prototype, config["wind_tunnel_settings"])
            
        # Run structural tests
        if "structural_settings" in config:
            results["structural_testing"] = self._run_structural_tests(prototype, config["structural_settings"])
            
        # Run thermal tests
        if "thermal_settings" in config:
            results["thermal_testing"] = self._run_thermal_tests(prototype, config["thermal_settings"])
            
        # Run analysis
        results["analysis"] = self._analyze_test_results(results)
        
        self.test_results = results
        return results
    
    def run_hil_tests(self, prototype: UCAVGeometry, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run hardware-in-the-loop tests for prototype.
        
        Args:
            prototype: UCAV prototype geometry
            config: HIL test configuration
            
        Returns:
            Dict[str, Any]: HIL test results
        """
        logger.info("Running prototype HIL tests")
        
        # Create HIL test suite
        test_suite = PrototypeHILTestSuite(
            name=f"Prototype_HIL_{int(time.time())}",
            prototype=prototype
        )
        
        # Add hardware interfaces
        for name, interface in self.hardware_interfaces.items():
            test_suite.add_hardware_interface(name, interface)
        
        # Add aerodynamic HIL tests
        if "aero_hil" in config:
            self._add_aero_hil_tests(test_suite, config["aero_hil"])
            
        # Add structural HIL tests
        if "structural_hil" in config:
            self._add_structural_hil_tests(test_suite, config["structural_hil"])
            
        # Add thermal HIL tests
        if "thermal_hil" in config:
            self._add_thermal_hil_tests(test_suite, config["thermal_hil"])
            
        # Add control system HIL tests
        if "control_hil" in config:
            self._add_control_hil_tests(test_suite, config["control_hil"])
        
        # Run the test suite
        hil_results = test_suite.run(self.system)
        
        # Save results
        timestamp = int(time.time())
        results_dir = os.path.join(os.getcwd(), "results", "hil_tests")
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, f"hil_test_{timestamp}.json"), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(hil_results)
            json.dump(serializable_results, f, indent=2)
        
        return hil_results
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON serializable format."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        else:
            return data
    
    def _add_aero_hil_tests(self, test_suite: PrototypeHILTestSuite, config: Dict[str, Any]) -> None:
        """Add aerodynamic HIL tests to the test suite."""
        # Add basic lift test
        test_suite.add_prototype_test(
            test_type="aero",
            name="LiftCoefficient",
            inputs={
                "computation": "aero_testing",
                "angle_of_attack": 5.0,
                "airspeed": config.get("airspeed", 100.0),
                "altitude": config.get("altitude", 1000.0)
            },
            expected_outputs={
                "lift_coefficient": lambda x: 0.1 <= x <= 1.5
            },
            validation_func=lambda outputs: (
                "lift_coefficient" in outputs and 
                0.1 <= outputs["lift_coefficient"] <= 1.5
            )
        )
        
        # Add drag test
        test_suite.add_prototype_test(
            test_type="aero",
            name="DragCoefficient",
            inputs={
                "computation": "aero_testing",
                "angle_of_attack": 5.0,
                "airspeed": config.get("airspeed", 100.0),
                "altitude": config.get("altitude", 1000.0)
            },
            expected_outputs={
                "drag_coefficient": lambda x: 0.01 <= x <= 0.2
            },
            validation_func=lambda outputs: (
                "drag_coefficient" in outputs and 
                0.01 <= outputs["drag_coefficient"] <= 0.2
            )
        )
    
    def _add_structural_hil_tests(self, test_suite: PrototypeHILTestSuite, config: Dict[str, Any]) -> None:
        """Add structural HIL tests to the test suite."""
        # Add wing load test
        test_suite.add_prototype_test(
            test_type="structural",
            name="WingLoadTest",
            inputs={
                "computation": "structural_testing",
                "load_factor": config.get("load_factor", 3.0),
                "test_points": config.get("test_points", 10)
            },
            expected_outputs={
                "max_stress": lambda x: x < 500.0,  # MPa
                "safety_factor": lambda x: x > 1.5
            }
        )
    
    def _add_thermal_hil_tests(self, test_suite: PrototypeHILTestSuite, config: Dict[str, Any]) -> None:
        """Add thermal HIL tests to the test suite."""
        # Add thermal cycle test
        test_suite.add_prototype_test(
            test_type="thermal",
            name="ThermalCycleTest",
            inputs={
                "computation": "thermal_testing",
                "min_temp": config.get("min_temp", -40),
                "max_temp": config.get("max_temp", 85),
                "cycles": config.get("cycles", 5)
            },
            expected_outputs={
                "max_temp_deformation": lambda x: x < 1.0,  # mm
                "thermal_cycles_completed": lambda x: x >= 5
            }
        )
    
    def _add_control_hil_tests(self, test_suite: PrototypeHILTestSuite, config: Dict[str, Any]) -> None:
        """Add control system HIL tests to the test suite."""
        # Add control response test
        test_suite.add_prototype_test(
            test_type="control",
            name="ControlResponseTest",
            inputs={
                "computation": "control_testing",
                "command": "roll",
                "magnitude": 15.0,  # degrees
                "duration": 2.0     # seconds
            },
            expected_outputs={
                "response_time": lambda x: x < 0.5,  # seconds
                "steady_state_error": lambda x: x < 2.0  # degrees
            }
        )
    
    def _run_aero_tests(self, prototype: UCAVGeometry, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run aerodynamic tests.
        
        Args:
            prototype: UCAV prototype geometry
            config: Test configuration
            
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Running aerodynamic tests")
        
        # Initialize system if needed
        self.system.initialize()
        
        # Prepare test data
        test_data = {
            "computation": "aero_testing",
            "prototype": prototype.export_for_cfd(),  # Use export_for_cfd instead of to_dict
            "max_speed": config.get("max_speed", 1.2),
            "test_points": config.get("test_points", 5),
            "angle_of_attack_range": config.get("angle_of_attack_range", [0, 15])
        }
        
        # Process data through neuromorphic system
        results = self.system.process_data(test_data)
        
        # Cleanup
        self.system.cleanup()
        
        return results
    
    def _run_structural_tests(self, prototype: UCAVGeometry, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run structural tests.
        
        Args:
            prototype: UCAV prototype geometry
            config: Test configuration
            
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Running structural tests")
        
        # Initialize system if needed
        self.system.initialize()
        
        # Prepare test data
        test_data = {
            "computation": "structural_testing",
            "prototype": prototype.export_for_cfd(),  # Use export_for_cfd instead of to_dict
            "max_load": config.get("max_load", 4.0),
            "test_points": config.get("test_points", 24)
        }
        
        # Process data through neuromorphic system
        results = self.system.process_data(test_data)
        
        # Cleanup
        self.system.cleanup()
        
        return results
    
    def _run_thermal_tests(self, prototype: UCAVGeometry, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run thermal tests.
        
        Args:
            prototype: UCAV prototype geometry
            config: Test configuration
            
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Running thermal tests")
        
        # Initialize system if needed
        self.system.initialize()
        
        # Prepare test data
        test_data = {
            "computation": "thermal_testing",
            "prototype": prototype.export_for_cfd(),  # Use export_for_cfd instead of to_dict
            "min_temp": config.get("min_temp", -40),
            "max_temp": config.get("max_temp", 85),
            "cycles": config.get("cycles", 10)
        }
        
        # Process data through neuromorphic system
        results = self.system.process_data(test_data)
        
        # Cleanup
        self.system.cleanup()
        
        return results
    
    def _analyze_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze test results.
        
        Args:
            results: Test results
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info("Analyzing test results")
        
        # Initialize system if needed
        self.system.initialize()
        
        # Extract relevant metrics from each test type
        analysis_metrics = {}
        
        # Process aerodynamic test results
        if "aero_testing" in results:
            aero_results = results["aero_testing"]
            analysis_metrics["aerodynamic"] = {
                "lift_to_drag_ratio": aero_results.get("lift_to_drag_ratio", 0),
                "stall_angle": aero_results.get("stall_angle", 0),
                "max_speed": aero_results.get("max_speed", 0),
                "passed": aero_results.get("passed", False)
            }
        
        # Process structural test results
        if "structural_testing" in results:
            struct_results = results["structural_testing"]
            analysis_metrics["structural"] = {
                "max_stress": struct_results.get("max_stress", 0),
                "safety_factor": struct_results.get("safety_factor", 0),
                "critical_points": struct_results.get("critical_points", []),
                "passed": struct_results.get("passed", False)
            }
        
        # Process thermal test results
        if "thermal_testing" in results:
            thermal_results = results["thermal_testing"]
            analysis_metrics["thermal"] = {
                "max_temp_deformation": thermal_results.get("max_temp_deformation", 0),
                "thermal_cycles_completed": thermal_results.get("thermal_cycles_completed", 0),
                "critical_points": thermal_results.get("critical_points", []),
                "passed": thermal_results.get("passed", False)
            }
        
        # Prepare analysis data for neuromorphic processing
        analysis_data = {
            "computation": "test_analysis",
            "test_metrics": analysis_metrics
        }
        
        # Process data through neuromorphic system
        analysis_results = self.system.process_data(analysis_data)
        
        # Add overall status
        all_passed = all(
            metrics.get("passed", False) 
            for test_type, metrics in analysis_metrics.items()
        )
        
        analysis_results["status"] = "completed" if all_passed else "partial"
        
        # Cleanup
        self.system.cleanup()
        
        return analysis_results

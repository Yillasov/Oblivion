"""
UCAV Prototype Testing Framework

Provides testing capabilities for UCAV prototypes.
"""

import time
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json

from src.core.utils.logging_framework import get_logger
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.manufacturing.quality.quality_inspector import QualityInspector
from src.simulation.models.ucav_geometry import UCAVGeometry
from src.manufacturing.testing.prototype_hil import PrototypeHILTestSuite

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
        
        results = {}
        
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
        
        results = {}
        
        # Run thermal tests
        if "thermal_settings" in config:
            results["thermal_testing"] = self._run_thermal_tests(prototype, config["thermal_settings"])
            
        # Run analysis
        results["analysis"] = self._analyze_test_results(results)
        
        self.test_results = results
        return results
    
    def _analyze_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze test results.
        
        Args:
            results: Test results
            
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Analyze test results")
        
        results = {}
        
        # Run analysis
        results["analysis"] = self._analyze_test_results(results)
        
        self.test_results = results
        return results

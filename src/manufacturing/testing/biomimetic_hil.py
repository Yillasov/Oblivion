#!/usr/bin/env python3
"""
Biomimetic Hardware-in-the-Loop Testing Framework

Simplified version of the prototype HIL testing framework with biomimetic capabilities.
"""

import sys
import os

# Fix the import path issue
# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
# Add project root to Python path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
from typing import Dict, List, Any, Optional, Callable
import numpy as np

# Now imports should work correctly
from src.core.utils.logging_framework import get_logger
from src.manufacturing.testing.prototype_hil import PrototypeHILTest, PrototypeHILTestSuite
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.simulation.models.ucav_geometry import UCAVGeometry
from src.biomimetic.sensors.integration_framework import BiomimeticSensorInterface, SensorDataType

logger = get_logger("biomimetic_hil")


class BiomimeticHILTest(PrototypeHILTest):
    """Hardware-in-the-loop test case for biomimetic systems."""
    
    def __init__(self, 
                name: str,
                test_type: str,
                inputs: Dict[str, Any],
                expected_outputs: Optional[Dict[str, Any]] = None,
                validation_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
                timeout: float = 30.0,
                biomimetic_config: Optional[Dict[str, Any]] = None):
        """Initialize a biomimetic HIL test case."""
        super().__init__(name, test_type, inputs, expected_outputs, validation_func, timeout)
        self.biomimetic_config = biomimetic_config or {}
        self.sensor_data = {}
        self.muscle_data = {}
        self.cpg_data = {}
        self.biomechanical_data = {}  # Added for biomechanical test data
    
    def record_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """Record data from biomimetic components during test."""
        if data_type == "sensor":
            self.sensor_data = data
        elif data_type == "muscle":
            self.muscle_data = data
        elif data_type == "cpg":
            self.cpg_data = data
        elif data_type == "biomechanical":  # Added biomechanical data type
            self.biomechanical_data = data
    
    def get_test_report(self) -> Dict[str, Any]:
        """Get detailed test report including biomimetic data."""
        report = super().get_test_report()
        report.update({
            "biomimetic_config": self.biomimetic_config,
            "sensor_data": self.sensor_data,
            "muscle_data": self.muscle_data,
            "cpg_data": self.cpg_data,
            "biomechanical_data": self.biomechanical_data  # Added to report
        })
        return report


class BiomimeticHILTestSuite(PrototypeHILTestSuite):
    """Test suite for biomimetic hardware-in-the-loop testing."""
    
    def __init__(self, name: str, prototype: UCAVGeometry, sensor_interface: Optional[BiomimeticSensorInterface] = None):
        """Initialize biomimetic HIL test suite."""
        super().__init__(name, prototype)
        self.sensor_interface = sensor_interface
        self.biomimetic_interfaces = {}
    
    def add_interface(self, name: str, interface: Any, interface_type: str = "biomimetic") -> None:
        """Add interface to the test suite."""
        if interface_type == "biomimetic":
            self.biomimetic_interfaces[name] = interface
        else:
            self.hardware_interfaces[name] = interface
    
    def add_test(self, 
                test_type: str,
                name: str, 
                inputs: Dict[str, Any],
                expected_outputs: Optional[Dict[str, Any]] = None,
                validation_func: Optional[Callable] = None,
                timeout: float = 30.0,
                biomimetic_config: Optional[Dict[str, Any]] = None) -> None:
        """Add biomimetic-specific HIL test case."""
        # Add prototype geometry to inputs
        inputs["prototype"] = self.prototype.get_geometry_data()
        
        # Add sensor interface if available
        if self.sensor_interface:
            inputs["sensor_interface"] = self.sensor_interface
        
        test = BiomimeticHILTest(
            name=name,
            test_type=test_type,
            inputs=inputs,
            expected_outputs=expected_outputs,
            validation_func=validation_func,
            timeout=timeout,
            biomimetic_config=biomimetic_config
        )
        self.add_test_case(test)
    
    def run(self, system: NeuromorphicSystem) -> Dict[str, Any]:
        """Run all test cases in the suite with biomimetic interfaces."""
        logger.info(f"Running biomimetic HIL test suite: {self.name}")
        
        # Initialize all interfaces
        for interfaces, interface_type in [
            (self.hardware_interfaces.items(), "hardware"),
            (self.biomimetic_interfaces.items(), "biomimetic")
        ]:
            for name, interface in interfaces:
                logger.info(f"Initializing {interface_type} interface: {name}")
                if hasattr(interface, "initialize"):
                    interface.initialize()
                if interface_type == "hardware":
                    system.add_hardware_interface(name, interface)
        
        # Run tests
        results = super().run(system)
        
        # Cleanup biomimetic interfaces
        for name, interface in self.biomimetic_interfaces.items():
            if hasattr(interface, "cleanup"):
                interface.cleanup()
        
        return results


def create_test_suite(test_type: str, prototype: UCAVGeometry, 
                     sensor_interface: Optional[BiomimeticSensorInterface] = None) -> BiomimeticHILTestSuite:
    """Create a test suite for the specified biomimetic component type."""
    test_suite = BiomimeticHILTestSuite(
        name=f"Biomimetic_{test_type.capitalize()}_HIL_{int(time.time())}",
        prototype=prototype,
        sensor_interface=sensor_interface
    )
    
    # Common test configurations
    test_configs = {
        "sensor": [
            {
                "name": "ProprioceptiveSensorTest",
                "inputs": {
                    "computation": "biomimetic_testing",
                    "sensor_type": SensorDataType.PROPRIOCEPTIVE.value,
                    "test_duration": 5.0,
                    "sample_rate": 100.0
                },
                "expected_outputs": {
                    "data_quality": lambda x: x > 0.8,
                    "response_time": lambda x: x < 0.05
                },
                "biomimetic_config": {
                    "noise_tolerance": 0.1,
                    "drift_tolerance": 0.05
                }
            },
            {
                "name": "FlowSensorTest",
                "inputs": {
                    "computation": "biomimetic_testing",
                    "sensor_type": SensorDataType.FLOW.value,
                    "test_duration": 5.0,
                    "flow_velocity": [10.0, 0.0, 0.0],
                    "sample_rate": 50.0
                },
                "expected_outputs": {
                    "data_quality": lambda x: x > 0.75,
                    "direction_accuracy": lambda x: x > 0.9
                }
            }
        ],
        "muscle": [
            {
                "name": "MuscleContractionTest",
                "inputs": {
                    "computation": "biomimetic_testing",
                    "contraction_cycles": 10,
                    "contraction_amplitude": 0.8,
                    "cycle_frequency": 1.0
                },
                "expected_outputs": {
                    "force_output": lambda x: 0.7 <= x <= 1.2,
                    "energy_efficiency": lambda x: x > 0.6,
                    "response_time": lambda x: x < 0.1
                }
            },
            {
                "name": "MuscleEnduranceTest",
                "inputs": {
                    "computation": "biomimetic_testing",
                    "contraction_cycles": 100,
                    "contraction_amplitude": 0.5,
                    "cycle_frequency": 2.0
                },
                "expected_outputs": {
                    "performance_degradation": lambda x: x < 0.2,
                    "temperature_rise": lambda x: x < 15.0
                }
            }
        ],
        "cpg": [
            {
                "name": "CPGRhythmTest",
                "inputs": {
                    "computation": "biomimetic_testing",
                    "target_frequency": 2.0,
                    "test_duration": 10.0,
                    "perturbation_time": 5.0
                },
                "expected_outputs": {
                    "frequency_accuracy": lambda x: x > 0.9,
                    "phase_stability": lambda x: x > 0.85,
                    "perturbation_recovery": lambda x: x < 1.0
                }
            },
            {
                "name": "CPGCoordinationTest",
                "inputs": {
                    "computation": "biomimetic_testing",
                    "oscillator_count": 4,
                    "coupling_strength": 0.3,
                    "test_duration": 15.0
                },
                "expected_outputs": {
                    "phase_coherence": lambda x: x > 0.8,
                    "synchronization_time": lambda x: x < 3.0
                }
            }
        ]
    }
    
    # Add tests for the specified type
    if test_type in test_configs:
        for test_config in test_configs[test_type]:
            test_suite.add_test(
                test_type=test_type,
                name=test_config["name"],
                inputs=test_config["inputs"],
                expected_outputs=test_config["expected_outputs"],
                biomimetic_config=test_config.get("biomimetic_config")
            )
    
    return test_suite


if __name__ == "__main__":
    # Example usage
    from src.simulation.models.ucav_geometry import UCAVGeometry
    from src.biomimetic.sensors.integration_framework import create_wing_sensor_system
    
    # Create prototype geometry
    prototype = UCAVGeometry(
        wingspan=10.0,
        length=15.0,
        mean_chord=2.5,
        sweep_angle=30.0,
        taper_ratio=0.4
    )
    
    # Create sensor interface
    sensor_interface = create_wing_sensor_system("wing_muscle_controller")
    
    # Create test suites
    sensor_suite = create_test_suite("sensor", prototype, sensor_interface)
    muscle_suite = create_test_suite("muscle", prototype)
    cpg_suite = create_test_suite("cpg", prototype)
    biomechanical_suite = create_test_suite("biomechanical", prototype)  # Added biomechanical test suite
    
    print(f"Created {len(sensor_suite.test_cases)} sensor tests")
    print(f"Created {len(muscle_suite.test_cases)} muscle tests")
    print(f"Created {len(cpg_suite.test_cases)} CPG tests")
    print(f"Created {len(biomechanical_suite.test_cases)} biomechanical tests")  # Added output
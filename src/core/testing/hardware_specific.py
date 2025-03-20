"""
Hardware-specific testing framework for neuromorphic systems.

Provides specialized test cases and validation methods for different
hardware platforms (Loihi, SpiNNaker, TrueNorth, etc.).
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Type
import numpy as np

from src.core.testing.hil_framework import HILTestCase, HILTestSuite
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.core.hardware.hardware_registry import hardware_registry
from src.core.learning.integration import LearningIntegration

logger = logging.getLogger(__name__)


class HardwareSpecificTest(HILTestCase):
    """Test case with hardware-specific validation logic."""
    
    def __init__(self, 
                 name: str,
                 hardware_type: str,
                 inputs: Dict[str, Any],
                 expected_outputs: Optional[Dict[str, Any]] = None,
                 validation_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
                 timeout: float = 10.0):
        """
        Initialize hardware-specific test case.
        
        Args:
            name: Test name
            hardware_type: Target hardware type
            inputs: Test inputs
            expected_outputs: Expected outputs
            validation_func: Custom validation function
            timeout: Test timeout in seconds
        """
        super().__init__(name, inputs, expected_outputs, validation_func, timeout)
        self.hardware_type = hardware_type
        
    def validate(self, outputs: Dict[str, Any]) -> bool:
        """
        Validate test outputs with hardware-specific logic.
        
        Args:
            outputs: Test outputs
            
        Returns:
            bool: Test result
        """
        # Apply hardware-specific validation adjustments
        if self.hardware_type == "loihi":
            # Loihi has higher timing variance
            return self._validate_loihi(outputs)
        elif self.hardware_type == "spinnaker":
            # SpiNNaker has different precision characteristics
            return self._validate_spinnaker(outputs)
        elif self.hardware_type == "truenorth":
            # TrueNorth has binary outputs
            return self._validate_truenorth(outputs)
        else:
            # Fall back to standard validation
            return super().validate(outputs)
    
    def _validate_loihi(self, outputs: Dict[str, Any]) -> bool:
        """Loihi-specific validation."""
        if self.validation_func:
            return self.validation_func(outputs)
            
        if not self.expected_outputs:
            return True
            
        # Loihi has timing variance, so we use a more relaxed comparison
        self.passed = True
        for key, expected in self.expected_outputs.items():
            if key not in outputs:
                self.passed = False
                continue
                
            actual = outputs[key]
            
            # For numeric values, use a more relaxed tolerance
            if isinstance(expected, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
                if abs(expected - actual) > 0.2:  # More relaxed tolerance
                    self.passed = False
            # For arrays, use relaxed tolerance
            elif isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
                if not np.allclose(expected, actual, rtol=0.2, atol=0.2):
                    self.passed = False
            # Otherwise use exact comparison
            elif expected != actual:
                self.passed = False
                
        return self.passed
    
    def _validate_spinnaker(self, outputs: Dict[str, Any]) -> bool:
        """SpiNNaker-specific validation."""
        # Similar to Loihi but with different tolerances
        # Simplified for brevity
        return self._validate_with_tolerance(outputs, rtol=0.15, atol=0.15)
    
    def _validate_truenorth(self, outputs: Dict[str, Any]) -> bool:
        """TrueNorth-specific validation."""
        # TrueNorth often has binary outputs
        # Simplified for brevity
        return self._validate_with_tolerance(outputs, rtol=0.1, atol=0.1)
    
    def _validate_with_tolerance(self, outputs: Dict[str, Any], rtol: float, atol: float) -> bool:
        """Validate with specific tolerance levels."""
        if self.validation_func:
            return self.validation_func(outputs)
            
        if not self.expected_outputs:
            return True
            
        self.passed = True
        for key, expected in self.expected_outputs.items():
            if key not in outputs:
                self.passed = False
                continue
                
            actual = outputs[key]
            
            # For numeric values
            if isinstance(expected, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
                if abs(expected - actual) > atol + rtol * abs(expected):
                    self.passed = False
            # For arrays
            elif isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
                if not np.allclose(expected, actual, rtol=rtol, atol=atol):
                    self.passed = False
            # Otherwise use exact comparison
            elif expected != actual:
                self.passed = False
                
        return self.passed


class HardwareSpecificTestSuite(HILTestSuite):
    """Test suite for hardware-specific tests."""
    
    def __init__(self, name: str, hardware_type: str):
        """
        Initialize hardware-specific test suite.
        
        Args:
            name: Suite name
            hardware_type: Target hardware type
        """
        super().__init__(name)
        self.hardware_type = hardware_type
        
    def add_hardware_test(self, 
                        name: str, 
                        inputs: Dict[str, Any],
                        expected_outputs: Optional[Dict[str, Any]] = None,
                        validation_func: Optional[Callable] = None) -> None:
        """
        Add hardware-specific test case.
        
        Args:
            name: Test name
            inputs: Test inputs
            expected_outputs: Expected outputs
            validation_func: Custom validation function
        """
        test = HardwareSpecificTest(
            name=name,
            hardware_type=self.hardware_type,
            inputs=inputs,
            expected_outputs=expected_outputs,
            validation_func=validation_func
        )
        self.add_test_case(test)
        
    def run(self, system: NeuromorphicSystem) -> Dict[str, Any]:
        """
        Run test suite on specified hardware.
        
        Args:
            system: Neuromorphic system
            
        Returns:
            Dict[str, Any]: Test results
        """
        # Check if system is using the right hardware
        hw_info = system.get_hardware_info()
        if hw_info.get("hardware_type") != self.hardware_type:
            logger.warning(f"Test suite for {self.hardware_type} running on {hw_info.get('hardware_type')}")
            
        return super().run(system)


class HardwareTestFactory:
    """Factory for creating hardware-specific test suites."""
    
    @staticmethod
    def create_test_suite(hardware_type: str, name: Optional[str] = None) -> HardwareSpecificTestSuite:
        """
        Create hardware-specific test suite.
        
        Args:
            hardware_type: Target hardware type
            name: Optional suite name
            
        Returns:
            HardwareSpecificTestSuite: Test suite
        """
        suite_name = name or f"{hardware_type.capitalize()}TestSuite"
        return HardwareSpecificTestSuite(suite_name, hardware_type)
    
    @staticmethod
    def create_standard_test_suite(hardware_type: str) -> HardwareSpecificTestSuite:
        """
        Create standard test suite for hardware type.
        
        Args:
            hardware_type: Target hardware type
            
        Returns:
            HardwareSpecificTestSuite: Standard test suite
        """
        suite = HardwareTestFactory.create_test_suite(hardware_type)
        
        # Add standard tests based on hardware type
        if hardware_type == "loihi":
            HardwareTestFactory._add_loihi_tests(suite)
        elif hardware_type == "spinnaker":
            HardwareTestFactory._add_spinnaker_tests(suite)
        elif hardware_type == "truenorth":
            HardwareTestFactory._add_truenorth_tests(suite)
        else:
            HardwareTestFactory._add_generic_tests(suite)
            
        return suite
    
    @staticmethod
    def _add_loihi_tests(suite: HardwareSpecificTestSuite) -> None:
        """Add Loihi-specific tests."""
        # Basic connectivity test
        suite.add_hardware_test(
            "LoihiConnectivity",
            inputs={"command": "ping"},
            expected_outputs={"status": "connected"}
        )
        
        # Neuron allocation test
        suite.add_hardware_test(
            "LoihiNeuronAllocation",
            inputs={"allocate_neurons": 100},
            validation_func=lambda outputs: "neuron_ids" in outputs and len(outputs["neuron_ids"]) == 100
        )
        
        # Learning test
        suite.add_hardware_test(
            "LoihiLearning",
            inputs={
                "learning_rule": "stdp",
                "pre_spikes": np.random.random((10, 5)) > 0.7,
                "post_spikes": np.random.random((10, 5)) > 0.7
            },
            validation_func=lambda outputs: "weight_changes" in outputs
        )
    
    @staticmethod
    def _add_spinnaker_tests(suite: HardwareSpecificTestSuite) -> None:
        """Add SpiNNaker-specific tests."""
        # Basic connectivity test
        suite.add_hardware_test(
            "SpiNNakerConnectivity",
            inputs={"command": "ping"},
            expected_outputs={"status": "connected"}
        )
        
        # Simplified for brevity
    
    @staticmethod
    def _add_truenorth_tests(suite: HardwareSpecificTestSuite) -> None:
        """Add TrueNorth-specific tests."""
        # Basic connectivity test
        suite.add_hardware_test(
            "TrueNorthConnectivity",
            inputs={"command": "ping"},
            expected_outputs={"status": "connected"}
        )
        
        # Simplified for brevity
    
    @staticmethod
    def _add_generic_tests(suite: HardwareSpecificTestSuite) -> None:
        """Add generic hardware tests."""
        # Basic connectivity test
        suite.add_hardware_test(
            "GenericConnectivity",
            inputs={"command": "ping"},
            expected_outputs={"status": "connected"}
        )
        
        # Simplified for brevity

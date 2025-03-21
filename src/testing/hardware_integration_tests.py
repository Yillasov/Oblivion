from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
import logging
import time  # Import the time module

logger = logging.getLogger("hardware_integration_tests")

@dataclass
class HardwareIntegrationTestCase:
    """Defines a test case for hardware integration with power supply systems."""
    name: str
    hardware_type: str
    inputs: Dict[str, Any]
    expected_outputs: Optional[Dict[str, Any]] = None
    validation_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    timeout: float = 10.0

    def validate(self, outputs: Dict[str, Any]) -> bool:
        """Validate test outputs with hardware-specific logic."""
        # Apply hardware-specific validation adjustments
        if self.validation_func is not None:
            return self.validation_func(outputs)
        
        if self.expected_outputs is not None:
            for key, expected_value in self.expected_outputs.items():
                if key not in outputs or outputs[key] != expected_value:
                    logger.warning(f"Output '{key}' values don't match: expected {expected_value}, got {outputs.get(key)}")
                    return False
        
        return True

class HardwareIntegrationTestSuite:
    """Test suite for hardware integration tests."""
    
    def __init__(self, name: str, hardware_type: str):
        """Initialize hardware integration test suite."""
        self.name = name
        self.hardware_type = hardware_type
        self.test_cases: Dict[str, HardwareIntegrationTestCase] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    def add_test_case(self, test_case: HardwareIntegrationTestCase) -> bool:
        """Add a hardware integration test case to the suite."""
        if test_case.name in self.test_cases:
            logger.warning(f"Test case '{test_case.name}' already exists")
            return False
        
        self.test_cases[test_case.name] = test_case
        return True
    
    def run_test(self, test_name: str, system: Any) -> Dict[str, Any]:
        """Run a specific hardware integration test case."""
        if test_name not in self.test_cases:
            logger.error(f"Test case '{test_name}' not found")
            return {"success": False, "error": "Test case not found"}
        
        test_case = self.test_cases[test_name]
        
        try:
            # Execute test
            outputs = system.run_hardware_test(test_case.inputs)
            success = test_case.validate(outputs)
        except Exception as e:
            logger.error(f"Error running test '{test_name}': {str(e)}")
            return {"success": False, "error": str(e)}
        
        result = {
            "success": success,
            "outputs": outputs
        }
        
        self.test_results[test_name] = result
        return result
    
    def run_all_tests(self, system: Any) -> Dict[str, Dict[str, Any]]:
        """Run all registered hardware integration test cases."""
        results = {}
        for test_name in self.test_cases:
            results[test_name] = self.run_test(test_name, system)
        return results
    
    def export_results(self, output_dir: str) -> str:
        """Export test results to a file."""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"hardware_test_results_{int(time.time())}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        return output_file
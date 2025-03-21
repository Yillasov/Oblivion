from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger("power_testing_framework")

@dataclass
class PowerTestCase:
    """Defines a test case for power supply systems."""
    name: str
    inputs: Dict[str, Any]
    expected_outputs: Optional[Dict[str, Any]] = None
    validation_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    timeout: float = 10.0

    def validate(self, outputs: Dict[str, Any]) -> bool:
        """Validate test outputs."""
        # If validation function is provided, use it
        if self.validation_func is not None:
            return self.validation_func(outputs)
        
        # If expected outputs are provided, compare directly
        if self.expected_outputs is not None:
            for key, expected_value in self.expected_outputs.items():
                if key not in outputs or outputs[key] != expected_value:
                    logger.warning(f"Output '{key}' values don't match: expected {expected_value}, got {outputs.get(key)}")
                    return False
        
        return True

class PowerTestingFramework:
    """Framework for automated testing of power supply systems."""
    
    def __init__(self):
        """Initialize the testing framework."""
        self.test_cases: Dict[str, PowerTestCase] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    def add_test_case(self, test_case: PowerTestCase) -> bool:
        """Add a test case to the framework."""
        if test_case.name in self.test_cases:
            logger.warning(f"Test case '{test_case.name}' already exists")
            return False
        
        self.test_cases[test_case.name] = test_case
        return True
    
    def run_test(self, test_name: str, system: Any) -> Dict[str, Any]:
        """Run a specific test case."""
        if test_name not in self.test_cases:
            logger.error(f"Test case '{test_name}' not found")
            return {"success": False, "error": "Test case not found"}
        
        test_case = self.test_cases[test_name]
        start_time = time.time()
        
        try:
            # Execute test
            outputs = system.run_test(test_case.inputs)
            success = test_case.validate(outputs)
        except Exception as e:
            logger.error(f"Error running test '{test_name}': {str(e)}")
            return {"success": False, "error": str(e)}
        
        execution_time = time.time() - start_time
        result = {
            "success": success,
            "outputs": outputs,
            "execution_time": execution_time
        }
        
        self.test_results[test_name] = result
        return result
    
    def run_all_tests(self, system: Any) -> Dict[str, Dict[str, Any]]:
        """Run all registered test cases."""
        results = {}
        for test_name in self.test_cases:
            results[test_name] = self.run_test(test_name, system)
        return results
    
    def export_results(self, output_dir: str) -> str:
        """Export test results to a file."""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"test_results_{int(time.time())}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        return output_file
"""
Testing framework for payload systems.

Provides tools for automated testing of payload functionality and performance.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import time
import json
import os

from src.payload.base import PayloadInterface


@dataclass
class TestCase:
    """Test case for a payload system."""
    test_id: str
    payload_id: str
    test_type: str  # e.g., "functional", "performance", "integration"
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    timeout: float = 10.0  # seconds


@dataclass
class TestResult:
    """Result of a test case execution."""
    test_id: str
    payload_id: str
    success: bool
    actual_output: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


class TestingFramework:
    """
    Framework for automated testing of payload systems.
    """
    
    def __init__(self):
        """Initialize the testing framework."""
        self.test_cases: Dict[str, TestCase] = {}
        self.test_results: Dict[str, TestResult] = {}
        self.payloads: Dict[str, PayloadInterface] = {}
        self.custom_validators: Dict[str, Callable] = {}
    
    def register_payload(self, payload_id: str, payload: PayloadInterface) -> bool:
        """Register a payload for testing."""
        if payload_id in self.payloads:
            return False
        
        self.payloads[payload_id] = payload
        return True
    
    def add_test_case(self, test_case: TestCase) -> bool:
        """Add a test case to the framework."""
        if test_case.test_id in self.test_cases:
            return False
        
        self.test_cases[test_case.test_id] = test_case
        return True
    
    def register_validator(self, validator_id: str, validator_func: Callable) -> bool:
        """Register a custom validation function."""
        if validator_id in self.custom_validators:
            return False
        
        self.custom_validators[validator_id] = validator_func
        return True
    
    def run_test(self, test_id: str) -> TestResult:
        """Run a specific test case."""
        if test_id not in self.test_cases:
            return TestResult(
                test_id=test_id,
                payload_id="unknown",
                success=False,
                actual_output={},
                execution_time=0.0,
                error_message="Test case not found"
            )
        
        test_case = self.test_cases[test_id]
        
        if test_case.payload_id not in self.payloads:
            return TestResult(
                test_id=test_id,
                payload_id=test_case.payload_id,
                success=False,
                actual_output={},
                execution_time=0.0,
                error_message="Payload not registered"
            )
        
        payload = self.payloads[test_case.payload_id]
        
        # Execute test
        start_time = time.time()
        error_message = None
        
        try:
            if test_case.test_type == "functional":
                actual_output = self._run_functional_test(payload, test_case.input_data)
            elif test_case.test_type == "performance":
                actual_output = self._run_performance_test(payload, test_case.input_data)
            else:
                actual_output = self._run_generic_test(payload, test_case.input_data)
        except Exception as e:
            actual_output = {"error": str(e)}
            error_message = str(e)
        
        execution_time = time.time() - start_time
        
        # Validate result
        success = self._validate_result(test_case, actual_output)
        
        # Create test result
        result = TestResult(
            test_id=test_id,
            payload_id=test_case.payload_id,
            success=success,
            actual_output=actual_output,
            execution_time=execution_time,
            error_message=error_message
        )
        
        # Store result
        self.test_results[test_id] = result
        
        return result
    
    def run_all_tests(self) -> Dict[str, TestResult]:
        """Run all registered test cases."""
        results = {}
        for test_id in self.test_cases:
            results[test_id] = self.run_test(test_id)
        return results
    
    def export_results(self, output_dir: str) -> str:
        """Export test results to a file."""
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"test_results_{int(time.time())}.json")
        
        # Convert results to serializable format
        serializable_results = {}
        for test_id, result in self.test_results.items():
            serializable_results[test_id] = {
                "test_id": result.test_id,
                "payload_id": result.payload_id,
                "success": result.success,
                "actual_output": result.actual_output,
                "execution_time": result.execution_time,
                "error_message": result.error_message
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return output_file
    
    def _run_functional_test(self, payload: PayloadInterface, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a functional test on a payload."""
        # Initialize payload if needed
        if hasattr(payload, 'initialize') and callable(getattr(payload, 'initialize')):
            payload.initialize()
        
        # Test specific functionality based on input
        if "test_function" in input_data:
            func_name = input_data["test_function"]
            if hasattr(payload, func_name) and callable(getattr(payload, func_name)):
                func = getattr(payload, func_name)
                args = input_data.get("args", {})
                result = func(**args)
                return {"result": result}
        
        # Default to deploy test
        if hasattr(payload, 'deploy') and callable(getattr(payload, 'deploy')):
            target_data = input_data.get("target_data", {})
            deploy_result = payload.deploy(target_data)
            status = payload.get_status() if hasattr(payload, 'get_status') else {}
            return {"deploy_result": deploy_result, "status": status}
        
        return {"error": "No testable function found"}
    
    def _run_performance_test(self, payload: PayloadInterface, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a performance test on a payload."""
        iterations = input_data.get("iterations", 1)
        func_name = input_data.get("test_function", "deploy")
        
        if not hasattr(payload, func_name) or not callable(getattr(payload, func_name)):
            return {"error": f"Function {func_name} not found"}
        
        # Initialize payload if needed
        if hasattr(payload, 'initialize') and callable(getattr(payload, 'initialize')):
            payload.initialize()
        
        func = getattr(payload, func_name)
        args = input_data.get("args", {})
        
        # Measure execution time over multiple iterations
        times = []
        results = []
        
        for _ in range(iterations):
            start_time = time.time()
            result = func(**args)
            execution_time = time.time() - start_time
            
            times.append(execution_time)
            results.append(result)
        
        return {
            "avg_execution_time": sum(times) / len(times),
            "min_execution_time": min(times),
            "max_execution_time": max(times),
            "results": results
        }
    
    def _run_generic_test(self, payload: PayloadInterface, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a generic test on a payload."""
        # Initialize payload if needed
        if hasattr(payload, 'initialize') and callable(getattr(payload, 'initialize')):
            payload.initialize()
        
        # Get specifications
        if hasattr(payload, 'get_specifications') and callable(getattr(payload, 'get_specifications')):
            specs = payload.get_specifications()
        else:
            specs = None
        
        # Get status
        if hasattr(payload, 'get_status') and callable(getattr(payload, 'get_status')):
            status = payload.get_status()
        else:
            status = None
        
        return {
            "specifications": specs,
            "status": status,
            "initialized": getattr(payload, 'initialized', False)
        }
    
    def _validate_result(self, test_case: TestCase, actual_output: Dict[str, Any]) -> bool:
        """Validate test results against expected output."""
        # Check for custom validator
        if "validator" in test_case.expected_output:
            validator_id = test_case.expected_output["validator"]
            if validator_id in self.custom_validators:
                return self.custom_validators[validator_id](actual_output, test_case.expected_output)
        
        # Simple validation - check if expected keys exist with expected values
        for key, expected_value in test_case.expected_output.items():
            if key == "validator":
                continue
                
            if key not in actual_output:
                return False
                
            if actual_output[key] != expected_value:
                return False
                
        return True
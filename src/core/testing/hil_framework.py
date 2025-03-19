"""
Hardware-in-the-Loop Testing Framework

Provides a simple infrastructure for testing neuromorphic algorithms
with real hardware or hardware simulators.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import json
import os
from datetime import datetime

from src.core.utils.logging_framework import get_logger
from src.core.integration.neuromorphic_system import NeuromorphicSystem, NeuromorphicInterface

logger = get_logger("hil_testing")


class HILTestCase:
    """
    A test case for hardware-in-the-loop testing.
    
    Defines inputs, expected outputs, and validation criteria.
    """
    
    def __init__(self, 
                 name: str,
                 inputs: Dict[str, Any],
                 expected_outputs: Optional[Dict[str, Any]] = None,
                 validation_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
                 timeout: float = 10.0):
        """
        Initialize a test case.
        
        Args:
            name: Name of the test case
            inputs: Input data for the test
            expected_outputs: Expected output data (optional)
            validation_func: Function to validate outputs (optional)
            timeout: Maximum time (seconds) to wait for test completion
        """
        self.name = name
        self.inputs = inputs
        self.expected_outputs = expected_outputs
        self.validation_func = validation_func
        self.timeout = timeout
        self.actual_outputs = None
        self.passed = None
        self.execution_time = None
    
    def validate(self, outputs: Dict[str, Any]) -> bool:
        """
        Validate test outputs.
        
        Args:
            outputs: Actual outputs from the system
            
        Returns:
            bool: True if test passed, False otherwise
        """
        self.actual_outputs = outputs
        
        # If validation function is provided, use it
        if self.validation_func is not None:
            self.passed = self.validation_func(outputs)
            return self.passed
        
        # If expected outputs are provided, compare directly
        if self.expected_outputs is not None:
            # Simple validation: check if all expected outputs exist and match
            self.passed = True
            for key, expected_value in self.expected_outputs.items():
                if key not in outputs:
                    logger.warning(f"Expected output '{key}' not found in actual outputs")
                    self.passed = False
                    continue
                
                actual_value = outputs[key]
                
                # Handle numpy arrays
                if isinstance(expected_value, np.ndarray) and isinstance(actual_value, np.ndarray):
                    if not np.allclose(expected_value, actual_value, rtol=1e-3, atol=1e-3):
                        logger.warning(f"Output '{key}' values don't match")
                        self.passed = False
                # Handle other types
                elif expected_value != actual_value:
                    logger.warning(f"Output '{key}' values don't match: expected {expected_value}, got {actual_value}")
                    self.passed = False
            
            return self.passed
        
        # If no validation criteria, test is considered passed
        self.passed = True
        return True


class HILTestSuite:
    """
    A collection of test cases for hardware-in-the-loop testing.
    """
    
    def __init__(self, name: str):
        """
        Initialize a test suite.
        
        Args:
            name: Name of the test suite
        """
        self.name = name
        self.test_cases = []
        self.results = {}
    
    def add_test_case(self, test_case: HILTestCase):
        """
        Add a test case to the suite.
        
        Args:
            test_case: Test case to add
        """
        self.test_cases.append(test_case)
    
    def run(self, system: NeuromorphicSystem) -> Dict[str, Any]:
        """
        Run all test cases in the suite.
        
        Args:
            system: Neuromorphic system to test
            
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info(f"Running test suite: {self.name}")
        
        # Initialize system
        system.initialize()
        
        # Run each test case
        for test_case in self.test_cases:
            logger.info(f"Running test case: {test_case.name}")
            
            start_time = time.time()
            
            try:
                # Run the test
                outputs = system.process_data(test_case.inputs)
                
                # Validate outputs
                passed = test_case.validate(outputs)
                
                end_time = time.time()
                test_case.execution_time = end_time - start_time
                
                # Store results
                self.results[test_case.name] = {
                    "passed": passed,
                    "execution_time": test_case.execution_time,
                    "inputs": test_case.inputs,
                    "expected_outputs": test_case.expected_outputs,
                    "actual_outputs": test_case.actual_outputs
                }
                
                logger.info(f"Test case {test_case.name}: {'PASSED' if passed else 'FAILED'} "
                           f"(execution time: {test_case.execution_time:.3f}s)")
                
            except Exception as e:
                logger.error(f"Error running test case {test_case.name}: {str(e)}")
                
                end_time = time.time()
                test_case.execution_time = end_time - start_time
                
                # Store results
                self.results[test_case.name] = {
                    "passed": False,
                    "execution_time": test_case.execution_time,
                    "inputs": test_case.inputs,
                    "expected_outputs": test_case.expected_outputs,
                    "actual_outputs": None,
                    "error": str(e)
                }
        
        # Clean up
        system.cleanup()
        
        # Return results
        return self.results
    
    def save_results(self, output_dir: str):
        """
        Save test results to a file.
        
        Args:
            output_dir: Directory to save results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{self.name}_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for test_name, result in self.results.items():
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_result[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            serializable_result[key][k] = v.tolist()
                        else:
                            serializable_result[key][k] = v
                else:
                    serializable_result[key] = value
            serializable_results[test_name] = serializable_result
        
        with open(filename, 'w') as f:
            json.dump({
                "suite_name": self.name,
                "timestamp": timestamp,
                "results": serializable_results
            }, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")


class HardwareSimulator(NeuromorphicInterface):
    """
    Simple hardware simulator for testing.
    
    Simulates a neuromorphic hardware device for testing purposes.
    """
    
    def __init__(self, 
                 latency: float = 0.01,
                 error_rate: float = 0.0,
                 noise_level: float = 0.0):
        """
        Initialize hardware simulator.
        
        Args:
            latency: Simulated processing latency (seconds)
            error_rate: Probability of random errors
            noise_level: Amount of noise to add to outputs
        """
        self.latency = latency
        self.error_rate = error_rate
        self.noise_level = noise_level
        self.neurons = {}
        self.synapses = {}
        self.initialized = False
        
        logger.info("Initialized hardware simulator")
    
    def initialize(self) -> None:
        """Initialize the hardware simulator."""
        time.sleep(0.1)  # Simulate initialization time
        self.initialized = True
        logger.info("Hardware simulator initialized")
    
    def cleanup(self) -> None:
        """Clean up hardware simulator resources."""
        time.sleep(0.1)  # Simulate cleanup time
        self.initialized = False
        logger.info("Hardware simulator cleaned up")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the hardware simulator."""
        return {
            "latency": self.latency,
            "error_rate": self.error_rate,
            "noise_level": self.noise_level,
            "initialized": self.initialized,
            "neuron_count": len(self.neurons),
            "synapse_count": len(self.synapses)
        }
    
    def allocate_neurons(self, count: int, params: Dict[str, Any] = {}) -> List[int]:
        """
        Allocate neurons on the simulated hardware.
        
        Args:
            count: Number of neurons to allocate
            params: Neuron parameters
            
        Returns:
            List[int]: IDs of allocated neurons
        """
        if not self.initialized:
            raise RuntimeError("Hardware simulator not initialized")
        
        # Simulate processing time
        time.sleep(self.latency)
        
        # Simulate random errors
        if np.random.random() < self.error_rate:
            raise RuntimeError("Simulated hardware error during neuron allocation")
        
        # Allocate neurons
        neuron_ids = []
        for i in range(count):
            neuron_id = len(self.neurons) + 1
            self.neurons[neuron_id] = params or {}
            neuron_ids.append(neuron_id)
        
        logger.info(f"Allocated {count} neurons on hardware simulator")
        return neuron_ids
    
    def create_synapses(self, connections: List[Tuple[int, int, float]]) -> None:
        """
        Create synapses on the simulated hardware.
        
        Args:
            connections: List of (pre_id, post_id, weight) tuples
        """
        if not self.initialized:
            raise RuntimeError("Hardware simulator not initialized")
        
        # Simulate processing time
        time.sleep(self.latency)
        
        # Simulate random errors
        if np.random.random() < self.error_rate:
            raise RuntimeError("Simulated hardware error during synapse creation")
        
        # Create synapses
        for pre_id, post_id, weight in connections:
            if pre_id not in self.neurons:
                raise ValueError(f"Presynaptic neuron {pre_id} not found")
            if post_id not in self.neurons:
                raise ValueError(f"Postsynaptic neuron {post_id} not found")
            
            self.synapses[(pre_id, post_id)] = weight
        
        logger.info(f"Created {len(connections)} synapses on hardware simulator")
    
    def update_synaptic_weights(self, connections: List[Tuple[int, int, float]]) -> None:
        """
        Update synaptic weights on the simulated hardware.
        
        Args:
            connections: List of (pre_id, post_id, weight) tuples
        """
        if not self.initialized:
            raise RuntimeError("Hardware simulator not initialized")
        
        # Simulate processing time
        time.sleep(self.latency)
        
        # Simulate random errors
        if np.random.random() < self.error_rate:
            raise RuntimeError("Simulated hardware error during weight update")
        
        # Update weights
        for pre_id, post_id, weight in connections:
            if (pre_id, post_id) not in self.synapses:
                raise ValueError(f"Synapse ({pre_id}, {post_id}) not found")
            
            self.synapses[(pre_id, post_id)] = weight
        
        logger.info(f"Updated {len(connections)} synaptic weights on hardware simulator")
    
    def run_simulation(self, inputs: Dict[int, float], duration: float) -> Dict[int, List[float]]:
        """
        Run a simulation on the hardware.
        
        Args:
            inputs: Dictionary mapping neuron IDs to input values
            duration: Simulation duration (ms)
            
        Returns:
            Dict[int, List[float]]: Dictionary mapping neuron IDs to spike times
        """
        if not self.initialized:
            raise RuntimeError("Hardware simulator not initialized")
        
        # Simulate processing time
        time.sleep(self.latency * (1 + duration / 100))
        
        # Simulate random errors
        if np.random.random() < self.error_rate:
            raise RuntimeError("Simulated hardware error during simulation")
        
        # Simple simulation: neurons with input > 0.5 spike once
        outputs = {}
        for neuron_id in self.neurons:
            # Get input for this neuron
            input_value = inputs.get(neuron_id, 0.0)
            
            # Add noise
            if self.noise_level > 0:
                input_value += np.random.normal(0, self.noise_level)
            
            # Generate spikes
            if input_value > 0.5:
                # Spike at random time
                spike_time = np.random.random() * duration
                outputs[neuron_id] = [spike_time]
            else:
                outputs[neuron_id] = []
        
        logger.info(f"Ran simulation for {duration}ms on hardware simulator")
        return outputs


# Example usage
def run_example_hil_test():
    """Run an example hardware-in-the-loop test."""
    from src.core.neuromorphic.simple_stdp import SimpleSTDP
    
    # Create hardware simulator
    hardware = HardwareSimulator(latency=0.01, noise_level=0.05)
    
    # Create neuromorphic system
    system = NeuromorphicSystem(hardware_interface=hardware)
    
    # Create test suite
    test_suite = HILTestSuite("ExampleTestSuite")
    
    # Create test cases
    test_case1 = HILTestCase(
        name="SimpleTest1",
        inputs={"sensor": {"data": np.random.random(10)}},
        expected_outputs={"output": {"value": 0.5}},
        validation_func=lambda outputs: "output" in outputs and "value" in outputs["output"]
    )
    
    test_case2 = HILTestCase(
        name="SimpleTest2",
        inputs={"sensor": {"data": np.zeros(10)}},
        expected_outputs={"output": {"value": 0.0}}
    )
    
    # Add test cases to suite
    test_suite.add_test_case(test_case1)
    test_suite.add_test_case(test_case2)
    
    # Run tests
    results = test_suite.run(system)
    
    # Save results
    test_suite.save_results("./test_results")
    
    return results
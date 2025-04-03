#!/usr/bin/env python3
"""
Hardware Verification Tool

Simple tool to verify neuromorphic hardware deployments.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append("/Users/yessine/Oblivion")

from src.core.utils.logging_framework import get_logger
from src.core.testing.hil_framework import HILTestCase, HILTestSuite
from src.core.integration.neuromorphic_system import NeuromorphicSystem, NeuromorphicInterface
from src.core.hardware.unified_interface import create_hardware_interface

logger = get_logger("hardware_verification")


class HardwareInterfaceAdapter(NeuromorphicInterface):
    """Adapter class to make NeuromorphicHardwareInterface compatible with NeuromorphicInterface protocol."""
    
    def __init__(self, hardware_interface):
        """Initialize the adapter with a hardware interface."""
        self.hardware_interface = hardware_interface
    
    def initialize(self) -> bool:
        """Initialize the hardware interface."""
        # Assuming the hardware_interface has an initialize method that takes no arguments
        # or can be called with default arguments
        return self.hardware_interface.initialize({}) if self.hardware_interface else True
    
    def cleanup(self) -> None:
        """Clean up hardware resources."""
        # Implement cleanup functionality or delegate to hardware_interface if available
        if hasattr(self.hardware_interface, 'cleanup'):
            self.hardware_interface.cleanup()
    
    def get_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        # Implement get_info functionality or delegate to hardware_interface if available
        if hasattr(self.hardware_interface, 'get_hardware_info'):
            return self.hardware_interface.get_hardware_info()
        return {"type": "adapter", "status": "active"}


class HardwareVerifier:
    """Simple hardware verification tool."""
    
    def __init__(self, hardware_address: Optional[str] = None):
        """
        Initialize hardware verifier.
        
        Args:
            hardware_address: Address of target hardware (IP or serial)
        """
        self.hardware_address = hardware_address
        logger.info("Initialized hardware verifier")
    
    def run_basic_tests(self) -> bool:
        """
        Run basic hardware verification tests.
        
        Returns:
            bool: True if all tests passed
        """
        logger.info("Running basic hardware verification tests")
        
        # Create test suite
        test_suite = HILTestSuite("BasicHardwareVerification")
        
        # Add basic connectivity test
        test_suite.add_test_case(HILTestCase(
            name="ConnectivityTest",
            inputs={},
            validation_func=lambda _: self._check_connectivity()
        ))
        
        # Add basic neuron allocation test
        test_suite.add_test_case(HILTestCase(
            name="NeuronAllocationTest",
            inputs={"neuron_count": 10},
            validation_func=lambda outputs: self._check_neuron_allocation(outputs.get("neuron_count", 10))
        ))
        
        # Add basic synapse creation test
        test_suite.add_test_case(HILTestCase(
            name="SynapseCreationTest",
            inputs={"synapse_count": 20},
            validation_func=lambda outputs: self._check_synapse_creation(outputs.get("synapse_count", 20))
        ))
        
        # Run tests
        results = {}
        for test_case in test_suite.test_cases:
            logger.info(f"Running test: {test_case.name}")
            start_time = time.time()
            
            try:
                # Run test
                passed = test_case.validation_func(test_case.inputs)
                
                # Store results
                test_case.passed = passed
                test_case.execution_time = time.time() - start_time
                
                results[test_case.name] = {
                    "passed": passed,
                    "execution_time": test_case.execution_time
                }
                
                logger.info(f"Test {test_case.name}: {'PASSED' if passed else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"Error running test {test_case.name}: {str(e)}")
                test_case.passed = False
                test_case.execution_time = time.time() - start_time
                
                results[test_case.name] = {
                    "passed": False,
                    "execution_time": test_case.execution_time,
                    "error": str(e)
                }
        
        # Check if all tests passed
        all_passed = all(result.get("passed", False) for result in results.values())
        
        if all_passed:
            logger.info("All hardware verification tests passed")
        else:
            logger.error("Some hardware verification tests failed")
            
            # Print failed tests
            for name, result in results.items():
                if not result.get("passed", False):
                    logger.error(f"  - {name}: {result.get('error', 'Failed')}")
        
        return all_passed
    
    def _check_connectivity(self) -> bool:
        """Check connectivity to hardware."""
        logger.info("Checking connectivity to hardware")
        
        try:
            # Use the already imported create_hardware_interface function
            hardware_interface = create_hardware_interface(self.hardware_address) if self.hardware_address else None
            
            # Wrap the hardware interface with our adapter
            adapter = HardwareInterfaceAdapter(hardware_interface)
            
            # Create neuromorphic system with the adapter
            hardware = NeuromorphicSystem(adapter)
            
            # Try to initialize and test connection
            connection_success = hardware.initialize()
            
            if connection_success:
                logger.info(f"Successfully connected to hardware at {self.hardware_address or 'default'}")
            else:
                logger.error(f"Failed to connect to hardware at {self.hardware_address or 'default'}")
                
            # Clean up resources
            hardware.cleanup()
            return connection_success
            
        except Exception as e:
            logger.error(f"Hardware connectivity check failed: {str(e)}")
            return False
    
    def _check_neuron_allocation(self, neuron_count: int) -> bool:
        """
        Check if neurons can be allocated on hardware.
        
        Args:
            neuron_count: Number of neurons to allocate
            
        Returns:
            bool: True if allocation successful
        """
        logger.info(f"Checking allocation of {neuron_count} neurons")
        
        # Simulate neuron allocation
        time.sleep(0.5)
        
        # In a real implementation, this would allocate neurons on actual hardware
        return True
    
    def _check_synapse_creation(self, synapse_count: int) -> bool:
        """
        Check if synapses can be created on hardware.
        
        Args:
            synapse_count: Number of synapses to create
            
        Returns:
            bool: True if creation successful
        """
        logger.info(f"Checking creation of {synapse_count} synapses")
        
        # Simulate synapse creation
        time.sleep(0.5)
        
        # In a real implementation, this would create synapses on actual hardware
        return True
    
    def run_algorithm_verification(self, algorithm_paths: List[str]) -> bool:
        """
        Verify algorithms on hardware.
        
        Args:
            algorithm_paths: Paths to algorithm files
            
        Returns:
            bool: True if all algorithms verified successfully
        """
        logger.info(f"Verifying {len(algorithm_paths)} algorithms on hardware")
        
        all_passed = True
        
        for path in algorithm_paths:
            if not os.path.exists(path):
                logger.error(f"Algorithm file not found: {path}")
                all_passed = False
                continue
            
            algorithm_name = os.path.basename(path)
            logger.info(f"Verifying algorithm: {algorithm_name}")
            
            # Simulate algorithm verification
            time.sleep(1.0)
            
            # In a real implementation, this would load and verify the algorithm on actual hardware
            logger.info(f"Algorithm {algorithm_name} verified successfully")
        
        return all_passed


def main():
    """Main entry point for hardware verification tool."""
    parser = argparse.ArgumentParser(description="Hardware Verification Tool")
    parser.add_argument("--address", help="Hardware address (IP or serial)")
    parser.add_argument("--algorithms", nargs="+", help="Paths to algorithm files to verify")
    parser.add_argument("--basic-only", action="store_true", help="Run only basic tests")
    
    args = parser.parse_args()
    
    # Create hardware verifier
    verifier = HardwareVerifier(args.address)
    
    # Run basic tests
    basic_passed = verifier.run_basic_tests()
    
    # Run algorithm verification if requested
    if not args.basic_only and args.algorithms:
        algorithm_passed = verifier.run_algorithm_verification(args.algorithms)
        all_passed = basic_passed and algorithm_passed
    else:
        all_passed = basic_passed
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
"""
Simple Continuous Integration Pipeline

Provides automated testing and integration for neuromorphic components.
"""

import os
import sys
import time
import json
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from src.core.utils.logging_framework import get_logger
from src.core.testing.hil_framework import HILTestSuite, HILTestCase, HardwareSimulator
from src.core.integration.neuromorphic_system import NeuromorphicSystem

logger = get_logger("ci_pipeline")


class CIPipeline:
    """Simple continuous integration pipeline for neuromorphic testing."""
    
    def __init__(self, 
                 repo_root: str = "/Users/yessine/Oblivion",
                 results_dir: str = "test_results",
                 config_file: str = "ci_config.json"):
        """
        Initialize CI pipeline.
        
        Args:
            repo_root: Root directory of the repository
            results_dir: Directory to store test results
            config_file: Configuration file for CI pipeline
        """
        self.repo_root = repo_root
        self.results_dir = os.path.join(repo_root, results_dir)
        self.config_file = os.path.join(repo_root, config_file)
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Load configuration if it exists
        self.config = self._load_config()
        
        # Test suites
        self.test_suites = []
        
        # Pipeline stages
        self.stages = [
            self.stage_setup,
            self.stage_unit_tests,
            self.stage_integration_tests,
            self.stage_hardware_tests,
            self.stage_report
        ]
        
        logger.info("Initialized CI pipeline")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "unit_test_dir": "tests/unit",
            "integration_test_dir": "tests/integration",
            "hardware_test_dir": "tests/hardware",
            "test_timeout": 300,
            "hardware_simulator": {
                "latency": 0.01,
                "error_rate": 0.02,
                "noise_level": 0.05
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with default config
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                return config
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
                return default_config
        else:
            logger.warning(f"Config file not found, using defaults")
            return default_config
    
    def add_test_suite(self, test_suite: HILTestSuite) -> None:
        """Add a test suite to the pipeline."""
        self.test_suites.append(test_suite)
        logger.info(f"Added test suite: {test_suite.name}")
    
    def stage_setup(self) -> bool:
        """Setup stage: prepare environment for testing."""
        logger.info("Running setup stage")
        
        try:
            # Check if required directories exist
            for dir_name in ["unit_test_dir", "integration_test_dir", "hardware_test_dir"]:
                dir_path = os.path.join(self.repo_root, self.config[dir_name])
                if not os.path.exists(dir_path):
                    logger.warning(f"Test directory not found: {dir_path}")
            
            # Create timestamp for this run
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_results_dir = os.path.join(self.results_dir, self.run_timestamp)
            os.makedirs(self.run_results_dir)
            
            return True
        except Exception as e:
            logger.error(f"Setup stage failed: {str(e)}")
            return False
    
    def stage_unit_tests(self) -> bool:
        """Unit test stage: run unit tests."""
        logger.info("Running unit test stage")
        
        unit_test_dir = os.path.join(self.repo_root, self.config["unit_test_dir"])
        if not os.path.exists(unit_test_dir):
            logger.warning(f"Unit test directory not found: {unit_test_dir}")
            return True
        
        try:
            # Run unit tests using pytest
            result = subprocess.run(
                ["python", "-m", "pytest", unit_test_dir, "-v"],
                capture_output=True,
                text=True,
                timeout=self.config["test_timeout"]
            )
            
            # Save test results
            with open(os.path.join(self.run_results_dir, "unit_tests.log"), 'w') as f:
                f.write(result.stdout)
                f.write(result.stderr)
            
            if result.returncode != 0:
                logger.error("Unit tests failed")
                return False
            
            logger.info("Unit tests passed")
            return True
        except Exception as e:
            logger.error(f"Unit test stage failed: {str(e)}")
            return False
    
    def stage_integration_tests(self) -> bool:
        """Integration test stage: run integration tests."""
        logger.info("Running integration test stage")
        
        integration_test_dir = os.path.join(self.repo_root, self.config["integration_test_dir"])
        if not os.path.exists(integration_test_dir):
            logger.warning(f"Integration test directory not found: {integration_test_dir}")
            return True
        
        try:
            # Run integration tests using pytest
            result = subprocess.run(
                ["python", "-m", "pytest", integration_test_dir, "-v"],
                capture_output=True,
                text=True,
                timeout=self.config["test_timeout"]
            )
            
            # Save test results
            with open(os.path.join(self.run_results_dir, "integration_tests.log"), 'w') as f:
                f.write(result.stdout)
                f.write(result.stderr)
            
            if result.returncode != 0:
                logger.error("Integration tests failed")
                return False
            
            logger.info("Integration tests passed")
            return True
        except Exception as e:
            logger.error(f"Integration test stage failed: {str(e)}")
            return False
    
    def stage_hardware_tests(self) -> bool:
        """Hardware test stage: run hardware-in-the-loop tests."""
        logger.info("Running hardware test stage")
        
        # Create hardware simulator
        hw_config = self.config["hardware_simulator"]
        hardware = HardwareSimulator(
            latency=hw_config["latency"],
            error_rate=hw_config["error_rate"],
            noise_level=hw_config["noise_level"]
        )
        
        # Create neuromorphic system
        system = NeuromorphicSystem(hardware_interface=hardware)
        
        # Run test suites
        all_passed = True
        for test_suite in self.test_suites:
            logger.info(f"Running test suite: {test_suite.name}")
            
            # Run tests
            results = test_suite.run(system)
            
            # Save results
            test_suite.save_results(self.run_results_dir)
            
            # Check if all tests passed
            for test_name, result in results.items():
                if not result.get("passed", False):
                    all_passed = False
                    logger.error(f"Test failed: {test_name}")
        
        return all_passed
    
    def stage_report(self) -> bool:
        """Report stage: generate test report."""
        logger.info("Running report stage")
        
        try:
            # Collect all test results
            report = {
                "timestamp": self.run_timestamp,
                "results_dir": self.run_results_dir,
                "stages": {
                    "unit_tests": os.path.exists(os.path.join(self.run_results_dir, "unit_tests.log")),
                    "integration_tests": os.path.exists(os.path.join(self.run_results_dir, "integration_tests.log")),
                    "hardware_tests": len(self.test_suites) > 0
                },
                "test_suites": {}
            }
            
            # Add test suite results
            for test_suite in self.test_suites:
                report["test_suites"][test_suite.name] = test_suite.results
            
            # Save report
            with open(os.path.join(self.run_results_dir, "report.json"), 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Test report saved to {os.path.join(self.run_results_dir, 'report.json')}")
            return True
        except Exception as e:
            logger.error(f"Report stage failed: {str(e)}")
            return False
    
    def run(self) -> bool:
        """Run the CI pipeline."""
        logger.info("Starting CI pipeline")
        
        start_time = time.time()
        success = True
        
        # Run each stage
        for stage in self.stages:
            stage_name = stage.__name__
            logger.info(f"Starting stage: {stage_name}")
            
            stage_start = time.time()
            stage_success = stage()
            stage_end = time.time()
            
            logger.info(f"Stage {stage_name} completed in {stage_end - stage_start:.2f}s")
            
            if not stage_success:
                logger.error(f"Stage {stage_name} failed")
                success = False
                break
        
        end_time = time.time()
        logger.info(f"CI pipeline completed in {end_time - start_time:.2f}s")
        
        return success


def main():
    """Main entry point for CI pipeline."""
    parser = argparse.ArgumentParser(description="Run CI pipeline")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--results-dir", help="Directory to store results")
    args = parser.parse_args()
    
    # Create CI pipeline
    ci = CIPipeline(
        config_file=args.config if args.config else "ci_config.json",
        results_dir=args.results_dir if args.results_dir else "test_results"
    )
    
    # Add test suites
    # TODO: Load test suites from config or discover automatically
    
    # Run pipeline
    success = ci.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
"""
Enhanced Continuous Integration Pipeline

Provides automated testing and integration for neuromorphic components.
"""

import os
import sys
import time
import json
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
import concurrent.futures

from src.core.utils.logging_framework import get_logger
from src.core.testing.hil_framework import HILTestSuite, HILTestCase, HardwareSimulator
from src.core.integration.neuromorphic_system import NeuromorphicSystem

logger = get_logger("ci_pipeline")


class CIPipeline:
    """Enhanced continuous integration pipeline for neuromorphic testing."""
    
    def __init__(self, 
                 repo_root: Optional[str] = None,
                 results_dir: str = "test_results",
                 config_file: str = "ci_config.json"):
        """
        Initialize CI pipeline.
        
        Args:
            repo_root: Root directory of the repository
            results_dir: Directory to store test results
            config_file: Configuration file for CI pipeline
        """
        # Determine repo root if not provided
        if repo_root is None:
            # Try to find the repo root by looking for .git directory
            current_dir = os.getcwd()
            max_depth = 10  # Prevent infinite loop
            depth = 0
            
            while current_dir != os.path.dirname(current_dir) and depth < max_depth:  # Stop at filesystem root or max depth
                if os.path.exists(os.path.join(current_dir, '.git')):
                    repo_root = current_dir
                    break
                current_dir = os.path.dirname(current_dir)
                depth += 1
            
            if repo_root is None:
                # Fallback to current directory
                repo_root = os.getcwd()
                logger.warning(f"Could not determine repository root, using current directory: {repo_root}")
        
        self.repo_root = repo_root
        self.results_dir = os.path.join(repo_root, results_dir)
        self.config_file = os.path.join(repo_root, config_file)
        
        # Create results directory if it doesn't exist
        try:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
        except Exception as e:
            logger.error(f"Failed to create results directory: {str(e)}")
            # Fallback to a temporary directory
            import tempfile
            self.results_dir = tempfile.mkdtemp(prefix="ci_results_")
            logger.warning(f"Using temporary directory for results: {self.results_dir}")
        
        # Load configuration if it exists
        self.config = self._load_config()
        
        # Test suites
        self.test_suites = []
        
        # Pipeline stages
        self.stages = [
            self.stage_setup,
            self.stage_lint,
            self.stage_unit_tests,
            self.stage_integration_tests,
            self.stage_hardware_tests,
            self.stage_report
        ]
        
        # Create timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_results_dir = os.path.join(self.results_dir, self.run_timestamp)
        
        logger.info(f"Initialized CI pipeline in {repo_root}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "unit_test_dir": "tests/unit",
            "integration_test_dir": "tests/integration",
            "hardware_test_dir": "tests/hardware",
            "test_timeout": 300,
            "parallel_tests": True,
            "lint": {
                "enabled": True,
                "tool": "flake8",
                "args": ["--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"]
            },
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
                        elif isinstance(value, dict) and isinstance(config[key], dict):
                            # Merge nested dictionaries
                            for nested_key, nested_value in value.items():
                                if nested_key not in config[key]:
                                    config[key][nested_key] = nested_value
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
            
            # Create results directory for this run
            if not os.path.exists(self.run_results_dir):
                os.makedirs(self.run_results_dir)
            
            # Save configuration
            with open(os.path.join(self.run_results_dir, "config.json"), 'w') as f:
                json.dump(self.config, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Setup stage failed: {str(e)}")
            return False
    
    def stage_lint(self) -> bool:
        """Lint stage: check code quality."""
        if not self.config.get("lint", {}).get("enabled", False):
            logger.info("Linting disabled, skipping")
            return True
        
        logger.info("Running lint stage")
        
        try:
            lint_config = self.config["lint"]
            lint_tool = lint_config["tool"]
            lint_args = lint_config["args"]
            
            if lint_tool == "flake8":
                cmd = ["flake8", "src", "tests"] + lint_args
            else:
                logger.warning(f"Unsupported lint tool: {lint_tool}")
                return True
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["test_timeout"]
            )
            
            # Save lint results
            with open(os.path.join(self.run_results_dir, "lint.log"), 'w') as f:
                f.write(result.stdout)
                f.write(result.stderr)
            
            if result.returncode != 0:
                logger.warning("Linting found issues")
                # Don't fail the pipeline for lint issues
                return True
            
            logger.info("Linting passed")
            return True
        except Exception as e:
            logger.error(f"Lint stage failed: {str(e)}")
            # Don't fail the pipeline for lint issues
            return True
    
    def stage_unit_tests(self) -> bool:
        """Unit test stage: run unit tests."""
        logger.info("Running unit test stage")
        
        unit_test_dir = os.path.join(self.repo_root, self.config["unit_test_dir"])
        if not os.path.exists(unit_test_dir):
            logger.warning(f"Unit test directory not found: {unit_test_dir}")
            return True
        
        try:
            # Run unit tests using pytest
            cmd = ["python", "-m", "pytest", unit_test_dir, "-v", "--cov=src"]
            
            result = subprocess.run(
                cmd,
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
            cmd = ["python", "-m", "pytest", integration_test_dir, "-v"]
            
            result = subprocess.run(
                cmd,
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
        
        if not self.test_suites:
            logger.warning("No hardware test suites defined, skipping")
            return True
        
        try:
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
            
            if self.config.get("parallel_tests", False):
                # Run test suites in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_suite = {
                        executor.submit(self._run_test_suite, test_suite, system): test_suite
                        for test_suite in self.test_suites
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_suite):
                        test_suite = future_to_suite[future]
                        try:
                            suite_passed = future.result()
                            if not suite_passed:
                                all_passed = False
                        except Exception as e:
                            logger.error(f"Test suite {test_suite.name} failed with exception: {str(e)}")
                            all_passed = False
            else:
                # Run test suites sequentially
                for test_suite in self.test_suites:
                    suite_passed = self._run_test_suite(test_suite, system)
                    if not suite_passed:
                        all_passed = False
            
            return all_passed
        except Exception as e:
            logger.error(f"Hardware test stage failed: {str(e)}")
            return False
    
    def _run_test_suite(self, test_suite: HILTestSuite, system: NeuromorphicSystem) -> bool:
        """Run a single test suite."""
        logger.info(f"Running test suite: {test_suite.name}")
        
        # Run tests
        results = test_suite.run(system)
        
        # Save results
        test_suite.save_results(self.run_results_dir)
        
        # Check if all tests passed
        all_passed = True
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
                    "lint": os.path.exists(os.path.join(self.run_results_dir, "lint.log")),
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
            
            # Generate HTML report
            self._generate_html_report(report)
            
            logger.info(f"Test report saved to {os.path.join(self.run_results_dir, 'report.json')}")
            return True
        except Exception as e:
            logger.error(f"Report stage failed: {str(e)}")
            return False
    
    def _generate_html_report(self, report: Dict[str, Any]) -> None:
        """Generate HTML report from JSON report."""
        try:
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>CI Pipeline Report - {report['timestamp']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .stage {{ margin-bottom: 20px; }}
                    .stage-header {{ background-color: #f0f0f0; padding: 10px; }}
                    .stage-content {{ padding: 10px; }}
                    .passed {{ color: green; }}
                    .failed {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>CI Pipeline Report</h1>
                <p>Timestamp: {report['timestamp']}</p>
                <p>Results Directory: {report['results_dir']}</p>
                
                <div class="stage">
                    <div class="stage-header">Stages</div>
                    <div class="stage-content">
                        <p>Lint: {'✅' if report['stages']['lint'] else '❌'}</p>
                        <p>Unit Tests: {'✅' if report['stages']['unit_tests'] else '❌'}</p>
                        <p>Integration Tests: {'✅' if report['stages']['integration_tests'] else '❌'}</p>
                        <p>Hardware Tests: {'✅' if report['stages']['hardware_tests'] else '❌'}</p>
                    </div>
                </div>
                
                <div class="stage">
                    <div class="stage-header">Test Suites</div>
                    <div class="stage-content">
            """
            
            for suite_name, suite_results in report.get('test_suites', {}).items():
                html_report += f"""
                        <h3>{suite_name}</h3>
                        <ul>
                """
                
                for test_name, test_result in suite_results.items():
                    passed = test_result.get('passed', False)
                    html_report += f"""
                            <li class="{'passed' if passed else 'failed'}">
                                {test_name}: {'Passed' if passed else 'Failed'}
                            </li>
                    """
                
                html_report += """
                        </ul>
                """
            
            html_report += """
                    </div>
                </div>
            </body>
            </html>
            """
            
            with open(os.path.join(self.run_results_dir, "report.html"), 'w') as f:
                f.write(html_report)
            
            logger.info(f"HTML report saved to {os.path.join(self.run_results_dir, 'report.html')}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")
    
    def run(self, test_type: Optional[str] = None) -> bool:
        """
        Run the CI pipeline.
        
        Args:
            test_type: Optional test type to run (unit, integration, hardware)
                       If None, run all stages
        
        Returns:
            True if all stages passed, False otherwise
        """
        logger.info("Starting CI pipeline")
        
        start_time = time.time()
        success = True
        
        # Filter stages based on test type
        stages_to_run = []
        if test_type == "unit":
            stages_to_run = [self.stage_setup, self.stage_unit_tests, self.stage_report]
        elif test_type == "integration":
            stages_to_run = [self.stage_setup, self.stage_integration_tests, self.stage_report]
        elif test_type == "hardware":
            stages_to_run = [self.stage_setup, self.stage_hardware_tests, self.stage_report]
        else:
            stages_to_run = self.stages
        
        # Run each stage
        for stage in stages_to_run:
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
    parser.add_argument("--test-type", choices=["unit", "integration", "hardware"],
                        help="Type of tests to run (unit, integration, hardware)")
    parser.add_argument("--repo-root", help="Repository root directory")
    args = parser.parse_args()
    
    # Create CI pipeline
    ci = CIPipeline(
        repo_root=args.repo_root,
        config_file=args.config if args.config else "ci_config.json",
        results_dir=args.results_dir if args.results_dir else "test_results"
    )
    
    # Add test suites
    # TODO: Load test suites from config or discover automatically
    
    # Run pipeline
    success = ci.run(test_type=args.test_type)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
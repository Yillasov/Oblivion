from typing import Dict, Any, List
import numpy as np
import json
import os

from src.simulation.physics.environment import EnvironmentModel
from .framework import HILTestFramework

class TestScenarioRunner:
    """Runner for HIL test scenarios."""
    
    def __init__(self, hil_framework: HILTestFramework, scenarios_dir: str):
        self.hil_framework = hil_framework
        self.scenarios_dir = scenarios_dir
        self.results = {}
    
    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Load a test scenario from file."""
        scenario_path = os.path.join(self.scenarios_dir, f"{scenario_name}.json")
        
        with open(scenario_path, 'r') as f:
            scenario = json.load(f)
        
        return scenario
    
    def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Run a named test scenario."""
        # Load scenario
        scenario = self.load_scenario(scenario_name)
        
        # Run test
        results = self.hil_framework.run_test_scenario(scenario)
        
        # Store results
        self.results[scenario_name] = results
        
        return results
    
    def run_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Run all test scenarios in the scenarios directory."""
        # Get all scenario files
        scenario_files = [f for f in os.listdir(self.scenarios_dir) 
                         if f.endswith('.json')]
        
        # Run each scenario
        for scenario_file in scenario_files:
            scenario_name = scenario_file.replace('.json', '')
            self.run_scenario(scenario_name)
        
        return self.results
    
    def analyze_results(self, scenario_name: str) -> Dict[str, Any]:
        """Analyze test results for a specific scenario."""
        if scenario_name not in self.results:
            return {"error": "Scenario results not found"}
        
        results = self.results[scenario_name]
        state_history = results["state_history"]
        
        # Calculate basic metrics
        metrics = {
            "max_altitude": max([state["position"][2] for state in state_history]),
            "max_speed": max([np.linalg.norm(state["velocity"]) for state in state_history]),
            "avg_acceleration": np.mean([np.linalg.norm(state["acceleration"]) for state in state_history]),
            "duration": results["duration"]
        }
        
        # Add scenario-specific metrics
        if "metrics" in results["scenario"]:
            for metric_name, metric_func in results["scenario"]["metrics"].items():
                # This would require evaluating functions from the scenario file
                # For simplicity, we'll skip this in the example
                pass
        
        return metrics
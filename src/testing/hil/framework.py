#!/usr/bin/env python3
"""
Base interface for hardware components in HIL testing.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
import threading
import time
import queue

from src.simulation.physics.simulation_runner import SimulationRunner
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.hardware.neuromorphic.integration import HardwareSNNIntegration

class HardwareInterface:
    
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = False
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.thread = None
    
    def initialize(self) -> bool:
        """Initialize hardware connection."""
        self.initialized = True
        return True
    
    def start(self) -> None:
        """Start hardware interface thread."""
        if not self.initialized:
            self.initialize()
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self) -> None:
        """Stop hardware interface thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _run_loop(self) -> None:
        """Main hardware interface loop."""
        while self.running:
            try:
                # Get input data if available
                try:
                    input_data = self.input_queue.get(block=False)
                    self._process_input(input_data)
                except queue.Empty:
                    pass
                
                # Generate output data
                output_data = self._generate_output()
                if output_data:
                    self.output_queue.put(output_data)
                
                time.sleep(0.001)  # Prevent CPU hogging
            except Exception as e:
                print(f"Hardware interface error: {e}")
    
    def _process_input(self, input_data: Dict[str, Any]) -> None:
        """Process input data from simulation."""
        pass
    
    def _generate_output(self) -> Optional[Dict[str, Any]]:
        """Generate output data for simulation."""
        return None
    
    def send_to_hardware(self, data: Dict[str, Any]) -> None:
        """Send data to hardware."""
        self.input_queue.put(data)
    
    def receive_from_hardware(self) -> Optional[Dict[str, Any]]:
        """Receive data from hardware."""
        try:
            return self.output_queue.get(block=False)
        except queue.Empty:
            return None

class NeuromorphicHardwareInterface(HardwareInterface):
    """Interface for neuromorphic hardware in HIL testing."""
    
    def __init__(self, hardware_integration: HardwareSNNIntegration, config: Dict[str, Any]):
        super().__init__(config)
        self.hardware = hardware_integration
        self.last_sensor_data = {}
    
    def _process_input(self, input_data: Dict[str, Any]) -> None:
        """Process sensor data from simulation."""
        if "sensor_data" in input_data:
            self.last_sensor_data = input_data["sensor_data"]
    
    def _generate_output(self) -> Optional[Dict[str, Any]]:
        """Generate control outputs using neuromorphic hardware."""
        if not self.last_sensor_data:
            return None
        
        # Process sensor data through hardware
        dt = self.config.get("time_step", 0.01)
        control_outputs = self.hardware.update(self.last_sensor_data, dt)
        
        return {"control_outputs": control_outputs}

class HILTestFramework:
    """Hardware-in-the-loop testing framework."""
    
    def __init__(self, 
                simulation: SimulationRunner,
                hardware_interfaces: Dict[str, HardwareInterface],
                config: Dict[str, Any]):
        self.simulation = simulation
        self.hardware_interfaces = hardware_interfaces
        self.config = config
        self.running = False
        self.thread = None
        self.time_step = config.get("time_step", 0.01)
        self.real_time = config.get("real_time", True)
        self.test_duration = config.get("test_duration", 60.0)  # seconds
        self.current_time = 0.0
    
    def start_test(self) -> None:
        """Start HIL test."""
        # Initialize and start hardware interfaces
        for name, interface in self.hardware_interfaces.items():
            if not interface.initialized:
                interface.initialize()
            interface.start()
        
        # Start test thread
        self.running = True
        self.thread = threading.Thread(target=self._run_test)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_test(self) -> None:
        """Stop HIL test."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        # Stop hardware interfaces
        for name, interface in self.hardware_interfaces.items():
            interface.stop()
    
    def _run_test(self) -> None:
        """Main test loop."""
        self.current_time = 0.0
        
        while self.running and self.current_time < self.test_duration:
            # Run simulation step
            sim_state = self.simulation.run_step()
            
            # Convert simulation state to sensor data
            sensor_data = self.simulation._state_to_sensor_data()
            
            # Send sensor data to hardware interfaces
            for name, interface in self.hardware_interfaces.items():
                interface.send_to_hardware({"sensor_data": sensor_data})
            
            # Collect control outputs from hardware interfaces
            control_inputs = {}
            for name, interface in self.hardware_interfaces.items():
                hw_data = interface.receive_from_hardware()
                if hw_data and "control_outputs" in hw_data:
                    # Convert control outputs to simulation inputs
                    hw_control_inputs = self.simulation._outputs_to_inputs(hw_data["control_outputs"])
                    control_inputs.update(hw_control_inputs)
            
            # Update simulation with hardware control inputs
            if control_inputs:
                self.simulation.run_step(control_inputs)
            
            # Update test time
            self.current_time += self.time_step
            
            # Sleep if real-time testing is enabled
            if self.real_time:
                time.sleep(self.time_step)
    
    def run_test_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test scenario."""
        # Configure simulation for scenario
        if "environment" in scenario:
            self.simulation.environment = scenario["environment"]
        
        if "mission_params" in scenario:
            self.simulation.mission_params = scenario["mission_params"]
        
        # Set test duration
        self.test_duration = scenario.get("duration", self.test_duration)
        
        # Start test
        self.start_test()
        
        # Wait for test completion
        while self.running and self.thread and self.thread.is_alive():
            time.sleep(0.1)
        
        # Collect and return results
        results = {
            "state_history": self.simulation.state_history,
            "duration": self.current_time,
            "scenario": scenario
        }
        
        return results
    
    def run_mission_profile(self, profile_path: str) -> Dict[str, Any]:
        """Run a test based on a mission profile file."""
        from src.testing.scenarios.mission_profiles import ScenarioGenerator
        
        # Create scenario generator
        generator = ScenarioGenerator(os.path.dirname(profile_path))
        
        # Load the profile
        profile_name = os.path.basename(profile_path).replace('.yaml', '')
        profile = generator.load_profile(profile_name)
        
        # Convert to scenario and run
        scenario = profile.to_scenario()
        return self.run_benchmark(scenario)
    
    def run_benchmark(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a benchmark test scenario with performance measurements."""
        from src.testing.benchmark.performance import HILBenchmark
        
        # Create benchmark instance
        benchmark = HILBenchmark(self, self.config.get("benchmark_config", {}))
        
        # Run benchmark scenario
        results = benchmark.run_benchmark_scenario(scenario)
        
        return results
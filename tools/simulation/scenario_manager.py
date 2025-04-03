#!/usr/bin/env python3
"""
Simulation Scenario Management System

Provides functionality to define, load, and run simulation scenarios.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import json
import time
from typing import Dict, List, Any, Optional
import importlib.util
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.utils.logging_framework import get_logger
from src.simulation.core.scheduler import SimulationScheduler, create_default_scheduler
from src.simulation.core.recorder import DataRecorder

logger = get_logger("scenario_manager")

class Scenario:
    """Represents a simulation scenario with configuration and setup logic."""
    
    def __init__(self, name: str, description: str, duration: float = 60.0, time_scale: float = 1.0):
        """
        Initialize a scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            duration: Scenario duration in seconds
            time_scale: Simulation time scale
        """
        self.name = name
        self.description = description
        self.duration = duration
        self.time_scale = time_scale
        self.config = {}
        self.recorder = DataRecorder(save_interval=0.1)
        
    def setup(self, scheduler: SimulationScheduler) -> bool:
        """
        Set up the scenario with the given scheduler.
        
        Args:
            scheduler: Simulation scheduler
            
        Returns:
            bool: True if setup was successful
        """
        # This method should be overridden by subclasses
        return True
        
    def save_to_file(self, filename: str) -> None:
        """
        Save scenario configuration to file.
        
        Args:
            filename: Path to save the scenario
        """
        data = {
            "name": self.name,
            "description": self.description,
            "duration": self.duration,
            "time_scale": self.time_scale,
            "config": self.config
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved scenario '{self.name}' to {filename}")
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'Scenario':
        """
        Load scenario from file.
        
        Args:
            filename: Path to the scenario file
            
        Returns:
            Scenario: Loaded scenario
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        scenario = cls(
            name=data["name"],
            description=data["description"],
            duration=data["duration"],
            time_scale=data["time_scale"]
        )
        scenario.config = data["config"]
        
        return scenario


class ScenarioManager:
    """Manages simulation scenarios."""
    
    def __init__(self, scenarios_dir: str):
        """
        Initialize the scenario manager.
        
        Args:
            scenarios_dir: Directory containing scenario files
        """
        self.scenarios_dir = scenarios_dir
        self.scenarios = {}
        self.current_scenario = None
        self.scheduler = None
        
        # Create scenarios directory if it doesn't exist
        os.makedirs(scenarios_dir, exist_ok=True)
        
        logger.info(f"Scenario manager initialized with directory: {scenarios_dir}")
    
    def load_scenario(self, name: str) -> bool:
        """
        Load a scenario by name.
        
        Args:
            name: Scenario name
            
        Returns:
            bool: True if scenario was loaded successfully
        """
        # Check if scenario is already loaded
        if name in self.scenarios:
            return True
        
        # Try to load from Python module
        module_path = os.path.join(self.scenarios_dir, f"{name}.py")
        if os.path.exists(module_path):
            try:
                # Load module
                try:
                    spec = importlib.util.spec_from_file_location(name, module_path)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Could not load module spec for {module_path}")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    logger.error(f"Failed to load module {name}: {e}")
                    raise
                
                # Get scenario class
                if hasattr(module, 'create_scenario'):
                    scenario = module.create_scenario()
                    self.scenarios[name] = scenario
                    logger.info(f"Loaded scenario '{name}' from Python module")
                    return True
            except Exception as e:
                logger.error(f"Error loading scenario '{name}' from Python module: {e}")
        
        # Try to load from JSON file
        json_path = os.path.join(self.scenarios_dir, f"{name}.json")
        if os.path.exists(json_path):
            try:
                scenario = Scenario.load_from_file(json_path)
                self.scenarios[name] = scenario
                logger.info(f"Loaded scenario '{name}' from JSON file")
                return True
            except Exception as e:
                logger.error(f"Error loading scenario '{name}' from JSON file: {e}")
        
        logger.error(f"Scenario '{name}' not found")
        return False
    
    def load_all_scenarios(self) -> None:
        """Load all scenarios from the scenarios directory."""
        # Get all Python and JSON files
        for filename in os.listdir(self.scenarios_dir):
            if filename.endswith('.py') or filename.endswith('.json'):
                name = os.path.splitext(filename)[0]
                self.load_scenario(name)
    
    def run_scenario(self, name: str) -> bool:
        """
        Run a scenario by name.
        
        Args:
            name: Scenario name
            
        Returns:
            bool: True if scenario was started successfully
        """
        # Load scenario if not already loaded
        if name not in self.scenarios and not self.load_scenario(name):
            return False
        
        # Get scenario
        scenario = self.scenarios[name]
        
        # Create scheduler
        self.scheduler = create_default_scheduler()
        self.scheduler.set_time_scale(scenario.time_scale)
        
        # Set up scenario
        if not scenario.setup(self.scheduler):
            logger.error(f"Failed to set up scenario '{name}'")
            return False
        
        # Set current scenario
        self.current_scenario = scenario
        
        # Start scheduler
        self.scheduler.start(threaded=True)
        
        logger.info(f"Started scenario '{name}'")
        return True
    
    def stop_current_scenario(self) -> None:
        """Stop the current scenario."""
        if self.scheduler:
            self.scheduler.stop()
            
        if self.current_scenario:
            logger.info(f"Stopped scenario '{self.current_scenario.name}'")
            self.current_scenario = None
    
    def get_scenario_list(self) -> List[str]:
        """
        Get a list of available scenarios.
        
        Returns:
            List[str]: List of scenario names
        """
        return list(self.scenarios.keys())
    
    def create_scenario(self, name: str, description: str, duration: float = 60.0, 
                       time_scale: float = 1.0) -> Scenario:
        """
        Create a new scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            duration: Scenario duration in seconds
            time_scale: Simulation time scale
            
        Returns:
            Scenario: Created scenario
        """
        scenario = Scenario(name, description, duration, time_scale)
        self.scenarios[name] = scenario
        return scenario


def create_example_scenarios(manager: ScenarioManager) -> None:
    """
    Create example scenarios.
    
    Args:
        manager: Scenario manager
    """
    # Basic flight scenario
    basic_flight = manager.create_scenario(
        name="basic_flight",
        description="Basic flight simulation with altitude changes",
        duration=30.0,
        time_scale=1.0
    )
    basic_flight.config = {
        "initial_altitude": 1000.0,
        "initial_velocity": 100.0,
        "waypoints": [
            {"time": 0.0, "altitude": 1000.0},
            {"time": 10.0, "altitude": 1500.0},
            {"time": 20.0, "altitude": 1200.0},
            {"time": 30.0, "altitude": 1000.0}
        ]
    }
    basic_flight.save_to_file(os.path.join(manager.scenarios_dir, "basic_flight.json"))
    
    # Create a Python module for a more complex scenario
    complex_scenario_path = os.path.join(manager.scenarios_dir, "complex_flight.py")
    with open(complex_scenario_path, 'w') as f:
        f.write('''"""
Complex Flight Scenario

A more complex flight scenario with sensor simulation and aerodynamics.
"""

from src.core.utils.logging_framework import get_logger
from src.simulation.core.scheduler import SimulationScheduler, TaskConfig
from src.simulation.sensors.sensor_framework import create_default_sensors
from src.simulation.aerodynamics.ucav_model import create_default_ucav_model
from tools.simulation.scenario_manager import Scenario

logger = get_logger("complex_flight")

class ComplexFlightScenario(Scenario):
    """A complex flight scenario with sensor simulation and aerodynamics."""
    
    def __init__(self):
        super().__init__(
            name="complex_flight",
            description="Complex flight with sensors and aerodynamics",
            duration=60.0,
            time_scale=1.0
        )
        
        # Additional configuration
        self.config = {
            "initial_altitude": 2000.0,
            "initial_velocity": 150.0,
            "wind_speed": 5.0,
            "wind_direction": 45.0,
            "turbulence": 0.2
        }
        
    def setup(self, scheduler: SimulationScheduler) -> bool:
        """Set up the scenario with the given scheduler."""
        try:
            # Create sensor manager
            self.sensors = create_default_sensors()
            
            # Create UCAV model
            self.ucav = create_default_ucav_model()
            
            # Initialize state
            self.state = {
                "platform": {
                    "position": [0.0, 0.0, -self.config["initial_altitude"]],
                    "velocity": [self.config["initial_velocity"], 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0]
                },
                "environment": {
                    "wind_speed": self.config["wind_speed"],
                    "wind_direction": self.config["wind_direction"],
                    "turbulence": self.config["turbulence"]
                },
                "sensors": {}
            }
            
            # Add tasks to scheduler
            
            # Update sensors
            scheduler.add_task(
                TaskConfig(
                    name="update_sensors",
                    update_rate=20.0,
                    group="sensors"
                ),
                self._update_sensors
            )
            
            # Update aerodynamics
            scheduler.add_task(
                TaskConfig(
                    name="update_aerodynamics",
                    update_rate=50.0,
                    group="physics"
                ),
                self._update_aerodynamics
            )
            
            # Update recorder
            scheduler.add_task(
                TaskConfig(
                    name="update_recorder",
                    update_rate=10.0,
                    group="system"
                ),
                self._update_recorder
            )
            
            logger.info("Complex flight scenario set up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up complex flight scenario: {e}")
            return False
    
    def _update_sensors(self, sim_time: float, *args, **kwargs) -> None:
        """Update sensor readings."""
        # Update sensor data
        sensor_data = self.sensors.update_all(
            sim_time, 
            self.state["platform"], 
            self.state["environment"]
        )
        
        # Store sensor data in state
        self.state["sensors"] = sensor_data
    
    def _update_aerodynamics(self, sim_time: float, *args, **kwargs) -> None:
        """Update aerodynamic forces and platform state."""
        # Get current state
        position = self.state["platform"]["position"]
        velocity = self.state["platform"]["velocity"]
        
        # Calculate forces (simplified)
        altitude = -position[2]
        airspeed = velocity[0]
        
        # Calculate aerodynamic forces
        forces = self.ucav.calculate_forces(
            velocity=airspeed,
            altitude=altitude,
            alpha=2.0,  # Fixed angle of attack for simplicity
            beta=0.0    # No sideslip
        )
        
        # Update position (very simplified)
        dt = 0.02  # Assuming 50Hz update rate
        position[0] += velocity[0] * dt
        position[1] += velocity[1] * dt
        position[2] += velocity[2] * dt
        
        # Update velocity (simplified)
        # In a real simulation, this would use proper flight dynamics
        if sim_time < 20.0:
            # Climb
            velocity[2] = -2.0
        elif sim_time < 40.0:
            # Level flight
            velocity[2] = 0.0
        else:
            # Descent
            velocity[2] = 2.0
    
    def _update_recorder(self, sim_time: float, *args, **kwargs) -> None:
        """Update data recorder."""
        self.recorder.update(sim_time, self.state)


def create_scenario():
    """Create and return the scenario."""
    return ComplexFlightScenario()
''')
    
    logger.info("Created example scenarios")

# Add this at the end of the file
if __name__ == "__main__":
    print("Simulation Scenario Management System")
    print("-------------------------------------")
    
    # Create a scenario manager
    scenarios_dir = os.path.join(os.path.dirname(__file__), "../../scenarios")
    manager = ScenarioManager(scenarios_dir)
    
    # Create example scenarios if they don't exist
    if not os.path.exists(os.path.join(scenarios_dir, "basic_flight.json")):
        print("Creating example scenarios...")
        create_example_scenarios(manager)
    
    # Load all scenarios
    manager.load_all_scenarios()
    
    # Display available scenarios
    scenarios = manager.get_scenario_list()
    if scenarios:
        print("\nAvailable scenarios:")
        for name in scenarios:
            scenario = manager.scenarios[name]
            print(f"  - {name}: {scenario.description}")
    else:
        print("\nNo scenarios available")
    
    print("\nUse the ScenarioManager class in your code to manage and run scenarios.")
#!/usr/bin/env python3
"""
Complex Flight Scenario

A more complex flight scenario with sensor simulation and aerodynamics.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

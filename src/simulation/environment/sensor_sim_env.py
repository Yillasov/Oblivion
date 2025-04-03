"""
Sensor Simulation Environment

Provides a controlled environment for testing and validating sensor algorithms
with simulated data.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import logging
import json

from src.simulation.sensors.sensor_framework import SensorManager, SensorType, SensorConfig, Sensor
from src.core.signal.neuromorphic_signal import NeuromorphicSignalProcessor, create_signal_processor
from src.core.fusion.sensor_fusion import SensorFusion, FusionConfig

logger = logging.getLogger(__name__)


@dataclass
class SimulationScenario:
    """Configuration for a simulation scenario."""
    name: str
    duration: float  # seconds
    time_step: float  # seconds
    platform_trajectory: List[Dict[str, Any]]  # List of platform states at different times
    targets: List[Dict[str, Any]]  # List of targets in the environment
    obstacles: List[Dict[str, Any]]  # List of obstacles in the environment
    environmental_conditions: Dict[str, Any]  # Weather, time of day, etc.


@dataclass
class SensorSimConfig:
    """Configuration for the sensor simulation environment."""
    scenarios_path: str = "configs/scenarios"
    output_path: str = "output/sensor_sim"
    real_time: bool = False  # Run in real-time or as fast as possible
    record_data: bool = True  # Record sensor data to files
    visualize: bool = False  # Enable visualization


class SensorSimEnvironment:
    """Main class for sensor simulation environment."""
    
    def __init__(self, config: SensorSimConfig = SensorSimConfig()):
        """Initialize the sensor simulation environment."""
        self.config = config
        self.sensor_manager = SensorManager()
        self.signal_processors = {}
        self.fusion_system = SensorFusion()
        
        # Simulation state
        self.current_scenario = None
        self.simulation_time = 0.0
        self.is_running = False
        self.is_paused = False
        self.current_step = 0
        
        # Data recording
        self.recorded_data = {
            "raw_sensor_data": [],
            "processed_data": [],
            "fusion_data": []
        }
        
        # Create output directory if it doesn't exist
        if self.config.record_data and not os.path.exists(self.config.output_path):
            os.makedirs(self.config.output_path)
        
        logger.info("Sensor simulation environment initialized")
    
    def load_scenario(self, scenario_name: str) -> bool:
        """Load a simulation scenario from file."""
        try:
            scenario_path = os.path.join(self.config.scenarios_path, f"{scenario_name}.json")
            
            if not os.path.exists(scenario_path):
                logger.error(f"Scenario file not found: {scenario_path}")
                return False
            
            with open(scenario_path, 'r') as f:
                scenario_data = json.load(f)
            
            # Create scenario object
            self.current_scenario = SimulationScenario(
                name=scenario_data.get("name", scenario_name),
                duration=scenario_data.get("duration", 60.0),
                time_step=scenario_data.get("time_step", 0.1),
                platform_trajectory=scenario_data.get("platform_trajectory", []),
                targets=scenario_data.get("targets", []),
                obstacles=scenario_data.get("obstacles", []),
                environmental_conditions=scenario_data.get("environmental_conditions", {})
            )
            
            logger.info(f"Loaded scenario: {self.current_scenario.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading scenario {scenario_name}: {e}")
            return False
    
    def add_sensor(self, sensor: Sensor) -> None:
        """Add a sensor to the simulation."""
        self.sensor_manager.add_sensor(sensor)
        
        # Create a signal processor for this sensor
        processor_type = "filter"  # Default processor type
        
        # Choose processor type based on sensor type
        if sensor.config.type == SensorType.NEUROMORPHIC_VISION:
            processor_type = "edge_detector"
        elif sensor.config.type == SensorType.SYNTHETIC_APERTURE_RADAR:
            processor_type = "filter"
        elif sensor.config.type == SensorType.LIDAR:
            processor_type = "integrator"
        elif sensor.config.type == SensorType.TERAHERTZ:
            processor_type = "frequency"
        
        # Create and store the processor
        self.signal_processors[sensor.config.name] = create_signal_processor(processor_type)
    
    def run_simulation(self) -> bool:
        """Run the full simulation scenario."""
        if self.current_scenario is None:
            logger.error("No scenario loaded")
            return False
        
        if not self.sensor_manager.sensors:
            logger.error("No sensors added to simulation")
            return False
        
        # Reset simulation state
        self.simulation_time = 0.0
        self.current_step = 0
        self.is_running = True
        self.is_paused = False
        self.recorded_data = {
            "raw_sensor_data": [],
            "processed_data": [],
            "fusion_data": []
        }
        
        logger.info(f"Starting simulation: {self.current_scenario.name}")
        
        # Main simulation loop
        while self.is_running and self.simulation_time < self.current_scenario.duration:
            if not self.is_paused:
                self._step_simulation()
                
                # Sleep if real-time simulation
                if self.config.real_time:
                    # Use a default time step if current_scenario is None (which shouldn't happen here)
                    sleep_time = getattr(self.current_scenario, 'time_step', 0.1)
                    time.sleep(sleep_time)
        
        # Save recorded data if enabled
        if self.config.record_data:
            self._save_recorded_data()
        
        logger.info(f"Simulation completed: {self.current_scenario.name}")
        return True
    
    def _step_simulation(self) -> None:
        """Execute a single simulation step."""
        # Get platform state at current time
        platform_state = self._get_platform_state_at_time(self.simulation_time)
        
        # Get environment state at current time
        environment_state = self._get_environment_state_at_time(self.simulation_time)
        
        # Update all sensors
        raw_sensor_data = self.sensor_manager.update_all(
            self.simulation_time, platform_state, environment_state
        )
        
        # Process sensor data with signal processors
        processed_data = {}
        for sensor_name, sensor_data in raw_sensor_data.items():
            if sensor_name in self.signal_processors:
                processor = self.signal_processors[sensor_name]
                processed_data[sensor_name] = {
                    "processed": processor.process(np.array(sensor_data.get("data", []))),
                    "original": sensor_data
                }
        
        # Perform sensor fusion
        fusion_data = self.fusion_system.process(processed_data, self.simulation_time)
        
        # Record data if enabled
        if self.config.record_data:
            self.recorded_data["raw_sensor_data"].append({
                "time": self.simulation_time,
                "data": raw_sensor_data
            })
            
            self.recorded_data["processed_data"].append({
                "time": self.simulation_time,
                "data": processed_data
            })
            
            self.recorded_data["fusion_data"].append({
                "time": self.simulation_time,
                "data": fusion_data
            })
        
        # Increment simulation time and step
        self.simulation_time += getattr(self.current_scenario, 'time_step', 0.1)
        self.current_step += 1
    
    def _get_platform_state_at_time(self, time: float) -> Dict[str, Any]:
        """Get the platform state at the specified time."""
        if not self.current_scenario or not self.current_scenario.platform_trajectory:
            return {"position": np.zeros(3), "orientation": np.zeros(3), "velocity": np.zeros(3)}
        
        # Find the closest trajectory points
        trajectory = self.current_scenario.platform_trajectory
        
        # If time is before first point, return first point
        if time <= trajectory[0].get("time", 0.0):
            return trajectory[0]
        
        # If time is after last point, return last point
        if time >= trajectory[-1].get("time", 0.0):
            return trajectory[-1]
        
        # Find two points to interpolate between
        for i in range(len(trajectory) - 1):
            t1 = trajectory[i].get("time", 0.0)
            t2 = trajectory[i + 1].get("time", 0.0)
            
            if t1 <= time <= t2:
                # Linear interpolation
                alpha = (time - t1) / max(0.001, t2 - t1)
                
                # Interpolate position, orientation, velocity
                state = {}
                for key in ["position", "orientation", "velocity"]:
                    if key in trajectory[i] and key in trajectory[i + 1]:
                        v1 = np.array(trajectory[i][key])
                        v2 = np.array(trajectory[i + 1][key])
                        state[key] = v1 + alpha * (v2 - v1)
                    else:
                        state[key] = np.zeros(3)
                
                return state
        
        # Fallback
        return {"position": np.zeros(3), "orientation": np.zeros(3), "velocity": np.zeros(3)}
    
    def _get_environment_state_at_time(self, time: float) -> Dict[str, Any]:
        """Get the environment state at the specified time."""
        if not self.current_scenario:
            return {}
        
        # Start with base environmental conditions
        env_state = self.current_scenario.environmental_conditions.copy()
        
        # Add time
        env_state["time"] = time
        
        # Add targets (with simple linear motion)
        targets = []
        for target in self.current_scenario.targets:
            # Copy target data
            target_state = target.copy()
            
            # Update position based on velocity if available
            if "position" in target and "velocity" in target:
                pos = np.array(target["position"])
                vel = np.array(target["velocity"])
                target_state["position"] = (pos + vel * time).tolist()
            
            targets.append(target_state)
        
        env_state["targets"] = targets
        
        # Add obstacles (static for now)
        env_state["obstacles"] = self.current_scenario.obstacles
        
        return env_state
    
    def _save_recorded_data(self) -> None:
        """Save recorded simulation data to files."""
        if not self.config.record_data or not self.current_scenario:
            return
        
        # Create scenario output directory
        scenario_dir = os.path.join(
            self.config.output_path, 
            f"{self.current_scenario.name}_{int(time.time())}"
        )
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Save raw sensor data
        with open(os.path.join(scenario_dir, "raw_sensor_data.json"), 'w') as f:
            json.dump(self.recorded_data["raw_sensor_data"], f, indent=2)
        
        # Save processed data
        with open(os.path.join(scenario_dir, "processed_data.json"), 'w') as f:
            json.dump(self.recorded_data["processed_data"], f, indent=2)
        
        # Save fusion data
        with open(os.path.join(scenario_dir, "fusion_data.json"), 'w') as f:
            json.dump(self.recorded_data["fusion_data"], f, indent=2)
        
        # Save scenario configuration
        with open(os.path.join(scenario_dir, "scenario_config.json"), 'w') as f:
            json.dump({
                "name": self.current_scenario.name,
                "duration": self.current_scenario.duration,
                "time_step": self.current_scenario.time_step,
                "platform_trajectory_count": len(self.current_scenario.platform_trajectory),
                "targets_count": len(self.current_scenario.targets),
                "obstacles_count": len(self.current_scenario.obstacles)
            }, f, indent=2)
        
        logger.info(f"Saved simulation data to {scenario_dir}")


# Create a simple example scenario
def create_example_scenario() -> SimulationScenario:
    """Create a simple example scenario for testing."""
    # Create a circular flight path
    trajectory = []
    duration = 60.0
    time_step = 0.5
    radius = 1000.0  # meters
    
    for t in np.arange(0, duration, time_step):
        angle = t * 0.1  # radians
        position = [radius * np.cos(angle), radius * np.sin(angle), -500.0]
        velocity = [-radius * 0.1 * np.sin(angle), radius * 0.1 * np.cos(angle), 0.0]
        orientation = [0.0, 0.0, angle]
        
        trajectory.append({
            "time": float(t),
            "position": position,
            "velocity": velocity,
            "orientation": orientation
        })
    
    # Create some targets
    targets = [
        {
            "id": 1,
            "position": [500.0, 0.0, -500.0],
            "velocity": [0.0, 0.0, 0.0],
            "type": "static",
            "signature": {
                "radar_cross_section": 5.0,
                "infrared_signature": 2.0
            }
        },
        {
            "id": 2,
            "position": [0.0, 800.0, -450.0],
            "velocity": [-5.0, 0.0, 0.0],
            "type": "moving",
            "signature": {
                "radar_cross_section": 10.0,
                "infrared_signature": 3.5
            }
        }
    ]
    
    # Create some obstacles
    obstacles = [
        {
            "id": 1,
            "position": [300.0, 300.0, -500.0],
            "size": [50.0, 50.0, 100.0],
            "material": "concrete"
        }
    ]
    
    # Environmental conditions
    env_conditions = {
        "temperature": 288.15,  # K (15°C)
        "pressure": 101325.0,   # Pa (sea level)
        "humidity": 0.6,        # 60%
        "wind": [2.0, 1.0, 0.0],  # m/s
        "precipitation": 0.0,   # mm/h
        "cloud_cover": 0.3,     # 30%
        "time_of_day": "day"
    }
    
    return SimulationScenario(
        name="example_scenario",
        duration=duration,
        time_step=time_step,
        platform_trajectory=trajectory,
        targets=targets,
        obstacles=obstacles,
        environmental_conditions=env_conditions
    )


# Helper function to create and run a simple simulation
def run_simple_simulation() -> None:
    """Create and run a simple sensor simulation."""
    # Create simulation environment
    sim_config = SensorSimConfig(
        scenarios_path="configs/scenarios",
        output_path="output/sensor_sim",
        record_data=True,
        real_time=False
    )
    sim_env = SensorSimEnvironment(sim_config)
    
    # Create and add example sensors
    from src.simulation.sensors.sensor_framework import SensorConfig, Radar, Altimeter
    
    # Add radar sensor
    radar_config = SensorConfig(
        type=SensorType.SYNTHETIC_APERTURE_RADAR,
        name="sar_sensor",
        update_rate=10.0,
        fov_horizontal=120.0,
        fov_vertical=60.0,
        max_range=50000.0,
        accuracy=0.9,
        noise_factor=0.02
    )
    sim_env.add_sensor(Radar(radar_config))
    
    # Add neuromorphic vision sensor
    vision_config = SensorConfig(
        type=SensorType.NEUROMORPHIC_VISION,
        name="neuro_vision",
        update_rate=30.0,
        fov_horizontal=90.0,
        fov_vertical=60.0,
        max_range=5000.0,
        accuracy=0.95,
        noise_factor=0.01
    )
    sim_env.add_sensor(Sensor(vision_config))
    
    # Create example scenario
    scenario = create_example_scenario()
    
    # Save scenario to file
    import os
    import json
    
    os.makedirs(sim_config.scenarios_path, exist_ok=True)
    scenario_path = os.path.join(sim_config.scenarios_path, f"{scenario.name}.json")
    
    # Convert scenario to dict for JSON serialization
    scenario_dict = {
        "name": scenario.name,
        "duration": scenario.duration,
        "time_step": scenario.time_step,
        "platform_trajectory": scenario.platform_trajectory,
        "targets": scenario.targets,
        "obstacles": scenario.obstacles,
        "environmental_conditions": scenario.environmental_conditions
    }
    
    with open(scenario_path, 'w') as f:
        json.dump(scenario_dict, f, indent=2)
    
    # Load and run the scenario
    sim_env.load_scenario(scenario.name)
    sim_env.run_simulation()
    
    logger.info("Simple simulation completed")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_simple_simulation()
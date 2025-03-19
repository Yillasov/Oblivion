#!/usr/bin/env python3
"""
Run Sensor Simulation Environment

A simple script to run the sensor simulation environment with various configurations.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.simulation.environment.sensor_sim_env import (
    SensorSimEnvironment, 
    SensorSimConfig,
    create_example_scenario,
    run_simple_simulation
)
from src.simulation.sensors.sensor_framework import (
    SensorType, 
    SensorConfig, 
    Sensor,
    Radar,
    Altimeter,
    SensorManager
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run sensor simulation environment")
    
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="example_scenario",
        help="Name of scenario to run"
    )
    
    parser.add_argument(
        "--create-example", 
        action="store_true",
        help="Create example scenario"
    )
    
    parser.add_argument(
        "--real-time", 
        action="store_true",
        help="Run simulation in real-time"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output/sensor_sim",
        help="Directory to save output data"
    )
    
    parser.add_argument(
        "--scenarios-dir", 
        type=str, 
        default="configs/scenarios",
        help="Directory containing scenario files"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("sensor_sim")
    
    # Create example scenario if requested
    if args.create_example:
        logger.info("Creating example scenario")
        scenario = create_example_scenario()
        
        # Save scenario to file
        os.makedirs(args.scenarios_dir, exist_ok=True)
        scenario_path = os.path.join(args.scenarios_dir, f"{scenario.name}.json")
        
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
            
        logger.info(f"Saved example scenario to {scenario_path}")
        
        # Use this scenario
        args.scenario = scenario.name
    
    # Create simulation environment
    sim_config = SensorSimConfig(
        scenarios_path=args.scenarios_dir,
        output_path=args.output_dir,
        record_data=True,
        real_time=args.real_time
    )
    
    sim_env = SensorSimEnvironment(sim_config)
    
    # Add default sensors
    logger.info("Adding default sensors")
    
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
    
    # Add lidar sensor
    lidar_config = SensorConfig(
        type=SensorType.LIDAR,
        name="lidar_sensor",
        update_rate=20.0,
        fov_horizontal=360.0,
        fov_vertical=30.0,
        max_range=200.0,
        accuracy=0.98,
        noise_factor=0.01
    )
    sim_env.add_sensor(Sensor(lidar_config))
    
    # Load and run the scenario
    logger.info(f"Loading scenario: {args.scenario}")
    if sim_env.load_scenario(args.scenario):
        logger.info(f"Running simulation: {args.scenario}")
        sim_env.run_simulation()
        logger.info(f"Simulation completed: {args.scenario}")
    else:
        logger.error(f"Failed to load scenario: {args.scenario}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
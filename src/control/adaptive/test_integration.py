#!/usr/bin/env python3
"""
Test script for the AdaptiveNeuromorphicControl integration.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#!/usr/bin/env python3


import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.control.adaptive.integration import AdaptiveNeuromorphicControl

def main():
    # Create a neuromorphic system instance
    neuromorphic_system = NeuromorphicSystem()
    
    # Initialize the adaptive neuromorphic control
    adaptive_control = AdaptiveNeuromorphicControl(
        hardware_integration=neuromorphic_system,
        airframe_type="fixed_wing",
        adaptive_config={
            "adaptation_rate": 0.1,
            "stability_weight": 0.5,
            "tracking_weight": 0.3,
            "energy_weight": 0.2
        }
    )
    
    # Create sample sensor data
    sensor_data = {
        "roll_rate": np.array([0.05, -0.02, 0.01]),
        "pitch_rate": np.array([0.01, 0.03, -0.01]),
        "yaw_rate": np.array([-0.01, 0.02, 0.01]),
        "acceleration": np.array([0.1, 0.2, 9.8])
    }
    
    # Create sample reference commands
    reference_commands = {
        "roll_rate": np.array([0.0, 0.0, 0.0]),
        "pitch_rate": np.array([0.02, 0.02, 0.02]),
        "yaw_rate": np.array([0.0, 0.0, 0.0])
    }
    
    # Run the adaptive control update
    dt = 0.01  # 10ms time step
    control_outputs = adaptive_control.update(sensor_data, reference_commands, dt)
    
    # Print the results
    print("Adaptive Control Test Results:")
    print("------------------------------")
    print(f"Performance Metrics: {adaptive_control.performance_metrics}")
    print("\nControl Outputs:")
    for key, value in control_outputs.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
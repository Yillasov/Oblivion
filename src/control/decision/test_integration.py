#!/usr/bin/env python3
"""
Test script for the DecisionControlIntegration.
"""

import numpy as np
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.control.adaptive.integration import AdaptiveNeuromorphicControl
from src.control.decision.integration import DecisionControlIntegration

def main():
    # Create a neuromorphic system instance
    neuromorphic_system = NeuromorphicSystem()
    
    # Initialize the adaptive neuromorphic control
    adaptive_control = AdaptiveNeuromorphicControl(
        hardware_integration=neuromorphic_system,
        airframe_type="fixed_wing",
        adaptive_config={
            "adaptation_rate": 0.1,
            "stability_weight": 0.5
        }
    )
    
    # Initialize the decision control integration
    decision_control = DecisionControlIntegration(
        adaptive_control=adaptive_control,
        decision_config={
            "threat_threshold": 0.7,
            "obstacle_threshold": 0.5,
            "mission_priority": "high"
        }
    )
    
    # Create sample sensor data
    sensor_data = {
        "position": np.array([100.0, 200.0, 500.0]),
        "velocity": np.array([50.0, 0.0, 0.0]),
        "roll_rate": np.array([0.05, -0.02, 0.01]),
        "pitch_rate": np.array([0.01, 0.03, -0.01]),
        "yaw_rate": np.array([-0.01, 0.02, 0.01]),
        "acceleration": np.array([0.1, 0.2, 9.8]),
        "altitude": np.array([500.0]),
        "threat_detection": np.array([0.8])  # High threat level
    }
    
    # Create sample mission parameters
    mission_params = {
        "target_position": np.array([500.0, 600.0, 450.0]),
        "base_position": np.array([0.0, 0.0, 100.0]),
        "mission_type": "reconnaissance",
        "priority_level": "high",
        "threat_response": "evasive"
    }
    
    # Run the decision control update
    dt = 0.01  # 10ms time step
    control_outputs = decision_control.update(sensor_data, mission_params, dt)
    
    # Print the results
    print("Decision Control Integration Test Results:")
    print("------------------------------------------")
    print("Highest Priority Decision:", 
          decision_control.decision_system.get_highest_priority_decision(
              decision_control.decision_system.update(sensor_data, mission_params)
          ) if decision_control.decision_system.get_highest_priority_decision(
              decision_control.decision_system.update(sensor_data, mission_params)
          ) is not None else "No decision")
    
    print("\nControl Outputs:")
    for key, value in control_outputs.items():
        print(f"  {key}: {value}")
    
    print("\nAdaptive Control Performance Metrics:")
    for key, value in adaptive_control.performance_metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
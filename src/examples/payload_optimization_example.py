"""
Example demonstrating the use of the Payload Optimization System.
"""

from typing import Dict, Any
import time

from src.payload.optimization import (
    PayloadOptimizer, OptimizationConstraints, MissionOptimizer, MissionProfile
)
from src.payload.conventional.weapons import AirToAirMissile, GuidedBomb
from src.payload.conventional.sensors import RadarSystem, ElectroOpticalSystem
from src.payload.conventional.electronic_warfare import JammingSystem
from src.payload.non_conventional.directed_energy import HighEnergyLaser
from src.payload.non_conventional.drone_systems import MicroDroneSwarm


def run_payload_optimization_example():
    """Run an example of the payload optimization system."""
    
    # Create a payload optimizer
    optimizer = PayloadOptimizer()
    optimizer.initialize()
    
    # Create some payload instances
    radar = RadarSystem("AESA-X")
    eo_system = ElectroOpticalSystem("EO-IR-500")
    missile = AirToAirMissile("AIM-120")
    bomb = GuidedBomb("GBU-39")
    jammer = JammingSystem("ALQ-250")
    laser = HighEnergyLaser("HEL-50")
    drone_swarm = MicroDroneSwarm("SWARM-100")
    
    # Register payloads with the optimizer
    optimizer.register_payload("radar", radar)
    optimizer.register_payload("eo_system", eo_system)
    optimizer.register_payload("missile", missile)
    optimizer.register_payload("bomb", bomb)
    optimizer.register_payload("jammer", jammer)
    optimizer.register_payload("laser", laser)
    optimizer.register_payload("drone_swarm", drone_swarm)
    
    # Create optimization constraints
    constraints = OptimizationConstraints(
        max_weight=800.0,  # kg
        max_power=15000.0,  # watts
        priority_targets=["fighter", "sam_site"],
        mission_type="air_superiority",
        environmental_factors={
            "weather": "clear",
            "time_of_day": "day",
            "terrain": "mountainous"
        }
    )
    
    # Run payload optimization
    print("Running payload optimization...")
    result = optimizer.optimize_configuration(constraints)
    
    print(f"\nRecommended payloads: {result.recommended_payloads}")
    print(f"Estimated effectiveness: {result.estimated_effectiveness:.2f}")
    print(f"Power usage: {result.power_usage:.2f} watts")
    print(f"Weight total: {result.weight_total:.2f} kg")
    print(f"Optimization score: {result.optimization_score:.2f}")
    print("\nPayload settings:")
    for pid, settings in result.payload_settings.items():
        print(f"  {pid}: {settings}")
    
    # Create a mission optimizer
    mission_optimizer = MissionOptimizer(optimizer)
    
    # Create mission profiles
    air_superiority = MissionProfile(
        name="Air Superiority",
        description="Establish air dominance by engaging enemy aircraft",
        target_types=["fighter", "bomber", "awacs"],
        priority_level=9,
        environmental_requirements={
            "weather": "all",
            "time_of_day": "all"
        },
        payload_preferences={
            "radar": 0.9,
            "missile": 0.9,
            "jammer": 0.7,
            "laser": 0.5
        }
    )
    
    ground_attack = MissionProfile(
        name="Ground Attack",
        description="Precision strikes against ground targets",
        target_types=["sam_site", "command_post", "vehicle_convoy"],
        priority_level=8,
        environmental_requirements={
            "weather": "clear",
            "time_of_day": "day"
        },
        payload_preferences={
            "eo_system": 0.9,
            "bomb": 0.9,
            "drone_swarm": 0.7
        }
    )
    
    # Register mission profiles
    mission_optimizer.register_mission_profile("air_superiority", air_superiority)
    mission_optimizer.register_mission_profile("ground_attack", ground_attack)
    
    # Platform constraints
    platform_constraints = {
        "max_weight": 1000.0,
        "max_power": 20000.0,
        "max_volume": 5.0
    }
    
    # Run mission-specific optimization
    print("\n\nRunning mission-specific optimization...")
    
    # Air superiority mission
    print("\nAir Superiority Mission:")
    air_result = mission_optimizer.optimize_for_mission("air_superiority", platform_constraints)
    print(f"Recommended payloads: {air_result.recommended_payloads}")
    print(f"Estimated effectiveness: {air_result.estimated_effectiveness:.2f}")
    print(f"Optimization score: {air_result.optimization_score:.2f}")
    
    # Ground attack mission
    print("\nGround Attack Mission:")
    ground_result = mission_optimizer.optimize_for_mission("ground_attack", platform_constraints)
    print(f"Recommended payloads: {ground_result.recommended_payloads}")
    print(f"Estimated effectiveness: {ground_result.estimated_effectiveness:.2f}")
    print(f"Optimization score: {ground_result.optimization_score:.2f}")
    
    # Target-specific optimization
    print("\n\nRunning target-specific optimization...")
    
    # Define a specific target
    target_data = {
        "type": "fighter",
        "distance": 40.0,
        "speed": 500.0,
        "heading": 270.0,
        "altitude": 9000.0,
        "ecm_capability": "advanced",
        "threat_level": "high"
    }
    
    # Available payloads for this scenario
    available_payloads = ["radar", "missile", "jammer", "laser"]
    
    # Run target-specific optimization
    target_result = optimizer.optimize_for_target(target_data, available_payloads)
    
    print("\nTarget-specific optimization results:")
    print(f"Recommended payloads: {target_result.get('recommended_payloads', [])}")
    print(f"Engagement sequence: {target_result.get('engagement_sequence', [])}")
    print(f"Estimated effectiveness: {target_result.get('effectiveness', 0.0):.2f}")
    
    print("\nPayload Optimization complete.")


if __name__ == "__main__":
    run_payload_optimization_example()
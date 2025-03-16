"""
Example demonstrating the integration of Payload Control and Optimization systems.
"""

from typing import Dict, Any
import time

from src.payload.control import NeuromorphicPayloadController, PayloadCoordinator, CoordinationStrategy
from src.payload.optimization import (
    PayloadOptimizer, OptimizationConstraints, MissionOptimizer, MissionProfile
)
from src.payload.conventional.weapons import AirToAirMissile, GuidedBomb
from src.payload.conventional.sensors import RadarSystem, ElectroOpticalSystem
from src.payload.conventional.electronic_warfare import JammingSystem
from src.payload.non_conventional.directed_energy import HighEnergyLaser
from src.payload.non_conventional.drone_systems import MicroDroneSwarm
from src.payload.non_conventional.countermeasures import AdaptiveDecoy


def run_integrated_example():
    """Run an example of the integrated payload systems."""
    
    print("Initializing Neuromorphic Payload Systems...")
    
    # Create controller and optimizer
    controller = NeuromorphicPayloadController()
    optimizer = PayloadOptimizer()
    
    # Initialize systems
    controller.initialize()
    optimizer.initialize()
    
    # Create mission optimizer
    mission_optimizer = MissionOptimizer(optimizer)
    
    # Create payload instances
    radar = RadarSystem("AESA-X")
    eo_system = ElectroOpticalSystem("EO-IR-500")
    missile = AirToAirMissile("AIM-120")
    bomb = GuidedBomb("GBU-39")
    jammer = JammingSystem("ALQ-250")
    laser = HighEnergyLaser("HEL-50")
    drone_swarm = MicroDroneSwarm("SWARM-100")
    decoy = AdaptiveDecoy("DECOY-X")
    
    # Register payloads with both systems
    payloads = {
        "radar": radar,
        "eo_system": eo_system,
        "missile": missile,
        "bomb": bomb,
        "jammer": jammer,
        "laser": laser,
        "drone_swarm": drone_swarm,
        "decoy": decoy
    }
    
    for pid, payload in payloads.items():
        controller.register_payload(pid, payload)
        optimizer.register_payload(pid, payload)
    
    # Create mission profiles
    air_defense = MissionProfile(
        name="Air Defense",
        description="Defend against incoming aerial threats",
        target_types=["fighter", "bomber", "cruise_missile"],
        priority_level=10,
        environmental_requirements={
            "weather": "all",
            "time_of_day": "all"
        },
        payload_preferences={
            "radar": 1.0,
            "missile": 0.9,
            "jammer": 0.8,
            "laser": 0.7,
            "decoy": 0.6
        }
    )
    
    # Register mission profile
    mission_optimizer.register_mission_profile("air_defense", air_defense)
    
    # Platform constraints
    platform_constraints = {
        "max_weight": 1200.0,
        "max_power": 25000.0,
        "max_volume": 6.0
    }
    
    print("\nOptimizing payload configuration for Air Defense mission...")
    
    # Run mission optimization
    optimization_result = mission_optimizer.optimize_for_mission("air_defense", platform_constraints)
    
    print(f"Recommended payloads: {optimization_result.recommended_payloads}")
    print(f"Estimated effectiveness: {optimization_result.estimated_effectiveness:.2f}")
    
    # Activate optimized payloads
    print("\nActivating optimized payloads...")
    for pid in optimization_result.recommended_payloads:
        success = controller.activate_payload(pid)
        print(f"  Activated {pid}: {success}")
    
    # Create payload groups based on optimization results
    sensor_group = [pid for pid in optimization_result.recommended_payloads 
                   if pid in ["radar", "eo_system"]]
    weapon_group = [pid for pid in optimization_result.recommended_payloads 
                   if pid in ["missile", "laser"]]
    support_group = [pid for pid in optimization_result.recommended_payloads 
                    if pid in ["jammer", "decoy"]]
    
    controller.create_payload_group("sensors", sensor_group)
    controller.create_payload_group("weapons", weapon_group)
    controller.create_payload_group("support", support_group)
    
    # Set mission parameters
    controller.set_mission_parameters({
        "mission_type": "air_defense",
        "priority_targets": ["fighter", "bomber"],
        "engagement_range": 60.0,
        "rules_of_engagement": "defensive",
        "environmental_conditions": {
            "weather": "clear",
            "time_of_day": "day"
        }
    })
    
    # Create a coordinator
    coordinator = PayloadCoordinator(controller)
    
    # Define a coordination strategy based on optimization results
    strategy = CoordinationStrategy(
        name="Optimized Air Defense",
        priority={pid: idx + 1 for idx, pid in enumerate(reversed(optimization_result.recommended_payloads))},
        timing={pid: idx * 0.5 for idx, pid in enumerate(optimization_result.recommended_payloads)},
        dependencies={
            "missile": ["radar"],
            "laser": ["radar", "eo_system"] if "eo_system" in optimization_result.recommended_payloads else ["radar"]
        },
        constraints={"max_simultaneous": 3}
    )
    
    # Register and activate the strategy
    coordinator.register_strategy("optimized_defense", strategy)
    coordinator.set_active_strategy("optimized_defense")
    
    # Simulate an incoming threat
    threat_data = {
        "type": "fighter",
        "distance": 55.0,
        "speed": 550.0,
        "heading": 90.0,
        "altitude": 7500.0,
        "ecm_capability": "advanced",
        "threat_level": "high"
    }
    
    print("\nIncoming threat detected!")
    print(f"Threat data: {threat_data}")
    
    # Optimize for this specific threat
    print("\nOptimizing response for specific threat...")
    threat_optimization = optimizer.optimize_for_target(threat_data, optimization_result.recommended_payloads)
    
    print(f"Threat-specific recommendations: {threat_optimization.get('recommended_payloads', [])}")
    print(f"Estimated effectiveness: {threat_optimization.get('effectiveness', 0.0):.2f}")
    
    # Execute coordinated operation
    print("\nExecuting coordinated defense operation...")
    operation_result = coordinator.execute_coordinated_operation(threat_data)
    
    print(f"Operation result: {operation_result}")
    
    # Get controller status
    print("\nController status:")
    print(controller.get_controller_status())
    
    # Shutdown systems
    controller.shutdown()
    print("\nSystems shutdown complete.")


if __name__ == "__main__":
    run_integrated_example()
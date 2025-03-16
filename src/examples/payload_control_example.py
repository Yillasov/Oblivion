"""
Example demonstrating the use of the Neuromorphic Payload Control system.
"""

from typing import Dict, Any
import time

from src.payload.control import NeuromorphicPayloadController, PayloadCoordinator, CoordinationStrategy
from src.payload.conventional.weapons import AirToAirMissile, GuidedBomb
from src.payload.conventional.sensors import RadarSystem
from src.payload.non_conventional.directed_energy import HighEnergyLaser
from src.payload.non_conventional.drone_systems import MicroDroneSwarm


def run_payload_control_example():
    """Run an example of the payload control system."""
    
    # Create a neuromorphic controller
    controller = NeuromorphicPayloadController()
    controller.initialize()
    
    # Create some payload instances
    radar = RadarSystem("AESA-X")
    missile = AirToAirMissile("AIM-120")
    bomb = GuidedBomb("GBU-39")
    laser = HighEnergyLaser("HEL-50")
    drone_swarm = MicroDroneSwarm("SWARM-100")
    
    # Register payloads with the controller
    controller.register_payload("radar", radar)
    controller.register_payload("missile", missile)
    controller.register_payload("bomb", bomb)
    controller.register_payload("laser", laser)
    controller.register_payload("drone_swarm", drone_swarm)
    
    # Activate payloads
    controller.activate_payload("radar")
    controller.activate_payload("missile")
    controller.activate_payload("laser")
    controller.activate_payload("drone_swarm")
    
    # Create payload groups
    controller.create_payload_group("air_defense", ["radar", "missile", "laser"])
    controller.create_payload_group("ground_attack", ["radar", "bomb", "drone_swarm"])
    
    # Set mission parameters
    controller.set_mission_parameters({
        "mission_type": "air_defense",
        "priority_targets": ["fighter", "bomber"],
        "engagement_range": 50.0,
        "rules_of_engagement": "defensive",
        "environmental_conditions": {
            "weather": "clear",
            "time_of_day": "day"
        }
    })
    
    # Create a coordinator
    coordinator = PayloadCoordinator(controller)
    
    # Define a coordination strategy
    strategy = CoordinationStrategy(
        name="Air Defense Strategy",
        priority={"radar": 3, "missile": 2, "laser": 1, "drone_swarm": 0},
        timing={"radar": 0.0, "missile": 2.0, "laser": 1.0, "drone_swarm": 3.0},
        dependencies={"missile": ["radar"], "laser": ["radar"]},
        constraints={"max_simultaneous": 2}
    )
    
    # Register and activate the strategy
    coordinator.register_strategy("air_defense_strategy", strategy)
    coordinator.set_active_strategy("air_defense_strategy")
    
    # Simulate a target
    target_data = {
        "type": "fighter",
        "distance": 35.0,
        "speed": 450.0,
        "heading": 270.0,
        "altitude": 8000.0,
        "threat_level": "high"
    }
    
    # Optimize payload configuration
    print("Optimizing payload configuration...")
    optimization_result = controller.optimize_payload_configuration()
    print(f"Optimization result: {optimization_result}")
    
    # Execute coordinated operation
    print("\nExecuting coordinated operation...")
    operation_result = coordinator.execute_coordinated_operation(target_data)
    print(f"Operation result: {operation_result}")
    
    # Deploy a payload group
    print("\nDeploying payload group...")
    group_result = controller.deploy_payload_group("air_defense", target_data)
    print(f"Group deployment result: {group_result}")
    
    # Get controller status
    print("\nController status:")
    print(controller.get_controller_status())
    
    # Shutdown
    controller.shutdown()
    print("\nController shutdown.")


if __name__ == "__main__":
    run_payload_control_example()
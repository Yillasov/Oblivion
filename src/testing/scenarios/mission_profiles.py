#!/usr/bin/env python3
"""
Represents a mission profile for scenario-based testing.
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

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import json
import yaml
from enum import Enum
from dataclasses import dataclass

from src.testing.hil.framework import HILTestFramework

class MissionType(Enum):
    RECONNAISSANCE = "reconnaissance"
    STRIKE = "strike"
    ESCORT = "escort"
    PATROL = "patrol"
    INTERCEPT = "intercept"
    RETURN_TO_BASE = "return_to_base"

@dataclass
class Waypoint:
    position: np.ndarray
    altitude: float
    speed: float
    loiter_time: float = 0.0
    action: Optional[str] = None

class MissionProfile:
    
    
    def __init__(self, 
                mission_type: MissionType,
                waypoints: List[Waypoint],
                duration: float,
                environment_conditions: Dict[str, Any],
                threat_scenarios: List[Dict[str, Any]] = []):
        self.mission_type = mission_type
        self.waypoints = waypoints
        self.duration = duration
        self.environment_conditions = environment_conditions
        self.threat_scenarios = threat_scenarios or []
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MissionProfile':
        """Create a mission profile from dictionary data."""
        mission_type = MissionType(data.get("mission_type", "reconnaissance"))
        
        waypoints = []
        for wp_data in data.get("waypoints", []):
            position = np.array(wp_data.get("position", [0, 0, 0]))
            waypoints.append(Waypoint(
                position=position,
                altitude=wp_data.get("altitude", position[2]),
                speed=wp_data.get("speed", 100.0),
                loiter_time=wp_data.get("loiter_time", 0.0),
                action=wp_data.get("action")
            ))
        
        return cls(
            mission_type=mission_type,
            waypoints=waypoints,
            duration=data.get("duration", 600.0),
            environment_conditions=data.get("environment_conditions", {}),
            threat_scenarios=data.get("threat_scenarios", [])
        )
    
    def to_scenario(self) -> Dict[str, Any]:
        """Convert mission profile to a test scenario."""
        waypoint_positions = [wp.position.tolist() for wp in self.waypoints]
        waypoint_altitudes = [wp.altitude for wp in self.waypoints]
        waypoint_speeds = [wp.speed for wp in self.waypoints]
        
        scenario = {
            "name": f"{self.mission_type.value}_mission",
            "duration": self.duration,
            "environment": self.environment_conditions,
            "mission_params": {
                "mission_type": self.mission_type.value,
                "waypoints": waypoint_positions,
                "waypoint_altitudes": waypoint_altitudes,
                "waypoint_speeds": waypoint_speeds,
                "target_position": waypoint_positions[-1] if waypoint_positions else [0, 0, 0],
                "base_position": waypoint_positions[0] if waypoint_positions else [0, 0, 0]
            },
            "threats": self.threat_scenarios
        }
        
        return scenario

class ScenarioGenerator:
    """Generates test scenarios for various mission profiles."""
    
    def __init__(self, profiles_dir: str):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
    
    def load_profile(self, profile_name: str) -> MissionProfile:
        """Load a mission profile from file."""
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.yaml")
        
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Mission profile not found: {profile_path}")
        
        with open(profile_path, 'r') as f:
            profile_data = yaml.safe_load(f)
        
        return MissionProfile.from_dict(profile_data)
    
    def generate_reconnaissance_mission(self, area_center: List[float], area_radius: float) -> MissionProfile:
        """Generate a reconnaissance mission profile."""
        # Create waypoints for a reconnaissance pattern
        waypoints = []
        
        # Starting point (base)
        base_pos = np.array([area_center[0] - area_radius*2, area_center[1], 0])
        waypoints.append(Waypoint(base_pos, altitude=100.0, speed=150.0))
        
        # Entry point
        entry_pos = np.array([area_center[0] - area_radius, area_center[1], 0])
        waypoints.append(Waypoint(entry_pos, altitude=500.0, speed=200.0))
        
        # Reconnaissance pattern (simple grid)
        grid_size = 4
        for i in range(grid_size):
            x = area_center[0] - area_radius/2 + (i % 2) * area_radius
            y = area_center[1] - area_radius/2 + (i // 2) * area_radius
            pos = np.array([x, y, 0])
            waypoints.append(Waypoint(pos, altitude=500.0, speed=150.0, loiter_time=30.0))
        
        # Exit point
        exit_pos = np.array([area_center[0] + area_radius, area_center[1], 0])
        waypoints.append(Waypoint(exit_pos, altitude=500.0, speed=200.0))
        
        # Return to base
        waypoints.append(Waypoint(base_pos, altitude=100.0, speed=150.0))
        
        # Environment conditions
        environment = {
            "wind": [5.0, 2.0, 0.0],
            "turbulence_intensity": 0.2,
            "temperature": 288.15,
            "pressure": 101325
        }
        
        return MissionProfile(
            mission_type=MissionType.RECONNAISSANCE,
            waypoints=waypoints,
            duration=1200.0,  # 20 minutes
            environment_conditions=environment
        )
    
    def generate_strike_mission(self, target_position: List[float], defenses: List[Dict[str, Any]]) -> MissionProfile:
        """Generate a strike mission profile."""
        # Create waypoints for a strike mission
        waypoints = []
        
        # Base position
        base_pos = np.array([target_position[0] - 100000, target_position[1], 0])
        waypoints.append(Waypoint(base_pos, altitude=100.0, speed=150.0))
        
        # Approach waypoints
        approach_distance = 50000
        approach_pos = np.array([target_position[0] - approach_distance, target_position[1], 0])
        waypoints.append(Waypoint(approach_pos, altitude=5000.0, speed=250.0))
        
        # Pre-strike waypoint
        pre_strike_pos = np.array([target_position[0] - 10000, target_position[1], 0])
        waypoints.append(Waypoint(pre_strike_pos, altitude=1000.0, speed=300.0))
        
        # Target waypoint
        target_pos = np.array(target_position)
        waypoints.append(Waypoint(target_pos, altitude=500.0, speed=350.0, action="strike"))
        
        # Egress waypoint
        egress_pos = np.array([target_position[0] + 10000, target_position[1], 0])
        waypoints.append(Waypoint(egress_pos, altitude=1000.0, speed=350.0))
        
        # Return to base
        waypoints.append(Waypoint(base_pos, altitude=100.0, speed=150.0))
        
        # Environment conditions
        environment = {
            "wind": [3.0, 1.0, 0.0],
            "turbulence_intensity": 0.1,
            "temperature": 288.15,
            "pressure": 101325
        }
        
        # Threat scenarios based on defenses
        threats = []
        for defense in defenses:
            threat = {
                "type": defense.get("type", "radar"),
                "position": defense.get("position", [0, 0, 0]),
                "activation_time": defense.get("activation_time", 300.0),
                "duration": defense.get("duration", 60.0)
            }
            threats.append(threat)
        
        return MissionProfile(
            mission_type=MissionType.STRIKE,
            waypoints=waypoints,
            duration=1800.0,  # 30 minutes
            environment_conditions=environment,
            threat_scenarios=threats
        )
    
    def save_profile(self, profile: MissionProfile, name: str) -> str:
        """Save a mission profile to file."""
        # Convert profile to dictionary
        profile_dict = {
            "mission_type": profile.mission_type.value,
            "duration": profile.duration,
            "environment_conditions": profile.environment_conditions,
            "waypoints": [
                {
                    "position": wp.position.tolist(),
                    "altitude": wp.altitude,
                    "speed": wp.speed,
                    "loiter_time": wp.loiter_time,
                    "action": wp.action
                }
                for wp in profile.waypoints
            ],
            "threat_scenarios": profile.threat_scenarios
        }
        
        # Save to YAML file
        profile_path = os.path.join(self.profiles_dir, f"{name}.yaml")
        with open(profile_path, 'w') as f:
            yaml.dump(profile_dict, f, default_flow_style=False)
        
        return profile_path

class ScenarioTestRunner:
    """Runs scenario-based tests for various mission profiles."""
    
    def __init__(self, hil_framework: HILTestFramework, profiles_dir: str):
        self.hil_framework = hil_framework
        self.scenario_generator = ScenarioGenerator(profiles_dir)
        self.results = {}
    
    def run_mission_profile(self, profile_name: str) -> Dict[str, Any]:
        """Run a test based on a mission profile."""
        # Load the mission profile
        profile = self.scenario_generator.load_profile(profile_name)
        
        # Convert profile to scenario
        scenario = profile.to_scenario()
        
        # Run the scenario
        results = self.hil_framework.run_benchmark(scenario)
        
        # Store results
        self.results[profile_name] = results
        
        return results
    
    def generate_and_run_reconnaissance(self, 
                                       area_center: List[float], 
                                       area_radius: float) -> Dict[str, Any]:
        """Generate and run a reconnaissance mission."""
        profile = self.scenario_generator.generate_reconnaissance_mission(area_center, area_radius)
        
        # Save the profile
        profile_name = f"recon_{area_center[0]}_{area_center[1]}_{area_radius}"
        self.scenario_generator.save_profile(profile, profile_name)
        
        # Run the profile
        return self.run_mission_profile(profile_name)
    
    def generate_and_run_strike(self, 
                               target_position: List[float], 
                               defenses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate and run a strike mission."""
        profile = self.scenario_generator.generate_strike_mission(target_position, defenses)
        
        # Save the profile
        profile_name = f"strike_{target_position[0]}_{target_position[1]}"
        self.scenario_generator.save_profile(profile, profile_name)
        
        # Run the profile
        return self.run_mission_profile(profile_name)
    
    def analyze_mission_results(self, profile_name: str) -> Dict[str, Any]:
        """Analyze results from a mission profile test."""
        if profile_name not in self.results:
            return {"error": "No results found for this profile"}
        
        results = self.results[profile_name]
        
        # Load the profile to get mission parameters
        profile = self.scenario_generator.load_profile(profile_name)
        
        # Basic analysis
        analysis = {
            "mission_type": profile.mission_type.value,
            "duration": results.get("duration", 0),
            "waypoints_reached": 0,
            "mission_success": False,
            "threats_encountered": 0,
            "performance_metrics": results.get("benchmark", {})
        }
        
        # Analyze state history to determine waypoints reached
        if "state_history" in results:
            state_history = results["state_history"]
            positions = np.array([state["position"] for state in state_history])
            
            # Count waypoints reached
            waypoints_reached = 0
            for waypoint in profile.waypoints:
                # Check if aircraft came within 100m of waypoint
                distances = np.linalg.norm(positions - waypoint.position, axis=1)
                if np.min(distances) < 100.0:
                    waypoints_reached += 1
            
            analysis["waypoints_reached"] = waypoints_reached
            
            # Mission success if all waypoints reached
            analysis["mission_success"] = waypoints_reached == len(profile.waypoints)
        
        # Count threats encountered
        if "threats" in results.get("scenario", {}):
            analysis["threats_encountered"] = len(results["scenario"]["threats"])
        
        return analysis
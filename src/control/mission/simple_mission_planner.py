#!/usr/bin/env python3
"""
Simple Mission Planning System

Provides basic mission planning capabilities with waypoints and simple actions.
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

import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("mission_planner")


class MissionActionType(Enum):
    """Simple mission action types."""
    MOVE_TO = "move_to"
    HOVER = "hover"
    LAND = "land"
    TAKEOFF = "takeoff"


@dataclass
class MissionWaypoint:
    """Simple waypoint definition."""
    
    position: List[float]  # [x, y, z] in meters
    heading: float = 0.0   # heading in degrees
    speed: float = 5.0     # speed in m/s


@dataclass
class MissionAction:
    """Simple mission action."""
    
    action_type: MissionActionType
    waypoint: Optional[MissionWaypoint] = None
    duration: float = 0.0  # duration in seconds (for HOVER)


class SimpleMission:
    """Simple mission definition."""
    
    def __init__(self, name: str):
        """Initialize mission."""
        self.name = name
        self.actions: List[MissionAction] = []
        self.current_action_index = 0
        
        logger.info(f"Created mission: {name}")
    
    def add_action(self, action: MissionAction) -> None:
        """Add action to mission."""
        self.actions.append(action)
        logger.debug(f"Added {action.action_type.value} action to mission {self.name}")
    
    def reset(self) -> None:
        """Reset mission to beginning."""
        self.current_action_index = 0
        logger.info(f"Reset mission: {self.name}")


class SimpleMissionPlanner:
    """Simple mission planning system."""
    
    def __init__(self):
        """Initialize mission planner."""
        self.missions: Dict[str, SimpleMission] = {}
        self.active_mission: Optional[SimpleMission] = None
        self.mission_complete = False
        
        logger.info("Initialized simple mission planner")
    
    def create_mission(self, name: str) -> SimpleMission:
        """Create a new mission."""
        mission = SimpleMission(name)
        self.missions[name] = mission
        return mission
    
    def start_mission(self, name: str) -> bool:
        """Start a mission by name."""
        if name not in self.missions:
            logger.error(f"Mission not found: {name}")
            return False
        
        self.active_mission = self.missions[name]
        self.active_mission.reset()
        self.mission_complete = False
        logger.info(f"Started mission: {name}")
        return True
    
    def get_current_action(self) -> Optional[MissionAction]:
        """Get current mission action."""
        if not self.active_mission or self.mission_complete:
            return None
        
        if self.active_mission.current_action_index >= len(self.active_mission.actions):
            self.mission_complete = True
            logger.info(f"Mission complete: {self.active_mission.name}")
            return None
        
        return self.active_mission.actions[self.active_mission.current_action_index]
    
    def advance_mission(self) -> bool:
        """Advance to next mission action."""
        if not self.active_mission or self.mission_complete:
            return False
        
        self.active_mission.current_action_index += 1
        
        if self.active_mission.current_action_index >= len(self.active_mission.actions):
            self.mission_complete = True
            logger.info(f"Mission complete: {self.active_mission.name}")
            return False
        
        logger.debug(f"Advanced to action {self.active_mission.current_action_index} in mission {self.active_mission.name}")
        return True


# Example usage
def create_example_mission() -> SimpleMission:
    """Create an example mission."""
    mission = SimpleMission("simple_patrol")
    
    # Takeoff
    mission.add_action(MissionAction(
        action_type=MissionActionType.TAKEOFF
    ))
    
    # Move to first waypoint
    mission.add_action(MissionAction(
        action_type=MissionActionType.MOVE_TO,
        waypoint=MissionWaypoint(
            position=[100.0, 0.0, -50.0],
            heading=0.0,
            speed=10.0
        )
    ))
    
    # Hover for 30 seconds
    mission.add_action(MissionAction(
        action_type=MissionActionType.HOVER,
        duration=30.0
    ))
    
    # Move to second waypoint
    mission.add_action(MissionAction(
        action_type=MissionActionType.MOVE_TO,
        waypoint=MissionWaypoint(
            position=[100.0, 100.0, -50.0],
            heading=90.0,
            speed=5.0
        )
    ))
    
    # Land
    mission.add_action(MissionAction(
        action_type=MissionActionType.LAND
    ))
    
    return mission
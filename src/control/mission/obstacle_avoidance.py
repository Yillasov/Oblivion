#!/usr/bin/env python3
"""
Simple Obstacle Avoidance System

Provides basic obstacle detection and avoidance capabilities.
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
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("obstacle_avoidance")


@dataclass
class Obstacle:
    """Simple obstacle representation."""
    
    position: np.ndarray  # [x, y, z] in meters
    radius: float         # Obstacle radius in meters


class SimpleObstacleAvoidance:
    """Simple obstacle avoidance implementation."""
    
    def __init__(self, safety_distance: float = 10.0):
        """Initialize obstacle avoidance system."""
        self.obstacles: List[Obstacle] = []
        self.safety_distance = safety_distance
        
        logger.info("Initialized simple obstacle avoidance system")
    
    def add_obstacle(self, position: List[float], radius: float) -> None:
        """Add obstacle to the environment."""
        self.obstacles.append(Obstacle(
            position=np.array(position),
            radius=radius
        ))
        logger.debug(f"Added obstacle at {position} with radius {radius}m")
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles = []
        logger.debug("Cleared all obstacles")
    
    def check_path(self, start: List[float], end: List[float]) -> bool:
        """
        Check if path between start and end is clear of obstacles.
        
        Returns:
            bool: True if path is clear, False if obstacle detected
        """
        start_pos = np.array(start)
        end_pos = np.array(end)
        
        # Direction vector
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
        
        # Check for collisions along path
        for obstacle in self.obstacles:
            # Vector from start to obstacle
            to_obstacle = obstacle.position - start_pos
            
            # Project onto path direction
            projection = np.dot(to_obstacle, direction)
            
            # Clamp projection to path length
            projection = max(0, min(distance, projection))
            
            # Find closest point on path to obstacle
            closest_point = start_pos + projection * direction
            
            # Check distance to obstacle
            obstacle_distance = np.linalg.norm(closest_point - obstacle.position)
            
            # If too close, path is not clear
            if obstacle_distance < (obstacle.radius + self.safety_distance):
                logger.debug(f"Path blocked by obstacle at {obstacle.position}")
                return False
        
        return True
    
    def find_avoidance_waypoint(self, current: List[float], 
                              target: List[float]) -> Optional[List[float]]:
        """
        Find a waypoint to avoid obstacles between current and target positions.
        
        Returns:
            Optional[List[float]]: Avoidance waypoint or None if no path found
        """
        current_pos = np.array(current)
        target_pos = np.array(target)
        
        # If path is clear, no need for avoidance
        if self.check_path(current, target):
            return None
        
        # Simple avoidance: try to go around obstacles
        for obstacle in self.obstacles:
            # Vector from obstacle to current position
            to_current = current_pos - obstacle.position
            
            # Vector from obstacle to target
            to_target = target_pos - obstacle.position
            
            # Normalized vectors
            if np.linalg.norm(to_current) > 0:
                to_current = to_current / np.linalg.norm(to_current)
            
            if np.linalg.norm(to_target) > 0:
                to_target = to_target / np.linalg.norm(to_target)
            
            # Compute avoidance direction (perpendicular to path)
            avoidance_dir = np.cross(np.cross(to_current, to_target), to_target)
            
            if np.linalg.norm(avoidance_dir) > 0:
                avoidance_dir = avoidance_dir / np.linalg.norm(avoidance_dir)
            else:
                # If vectors are parallel, use a default avoidance direction
                avoidance_dir = np.array([to_target[1], -to_target[0], 0])
                if np.linalg.norm(avoidance_dir) > 0:
                    avoidance_dir = avoidance_dir / np.linalg.norm(avoidance_dir)
            
            # Compute avoidance waypoint
            avoidance_distance = obstacle.radius + self.safety_distance * 2
            avoidance_point = obstacle.position + avoidance_dir * avoidance_distance
            
            # Check if this avoidance path is clear
            if self.check_path(current, avoidance_point.tolist()) and \
               self.check_path(avoidance_point.tolist(), target):
                logger.debug(f"Found avoidance waypoint at {avoidance_point}")
                return avoidance_point.tolist()
        
        logger.warning("Could not find valid avoidance path")
        return None
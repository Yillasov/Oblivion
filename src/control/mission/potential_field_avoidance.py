#!/usr/bin/env python3
"""
Potential Field Obstacle Avoidance System

Implements a simple potential field approach for obstacle avoidance.
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
from typing import List, Optional
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("potential_field_avoidance")


@dataclass
class PotentialFieldConfig:
    """Configuration for potential field algorithm."""
    
    # Attractive force parameters
    attractive_gain: float = 1.0
    goal_threshold: float = 5.0
    
    # Repulsive force parameters
    repulsive_gain: float = 100.0
    influence_radius: float = 50.0


class PotentialFieldAvoidance:
    """Potential field based obstacle avoidance."""
    
    def __init__(self, config: Optional[PotentialFieldConfig] = None):
        """Initialize potential field avoidance system."""
        self.config = config if config else PotentialFieldConfig()
        self.obstacles = []
        
        logger.info("Initialized potential field avoidance system")
    
    def add_obstacle(self, position: List[float], radius: float) -> None:
        """Add obstacle to the environment."""
        self.obstacles.append({
            "position": np.array(position),
            "radius": radius
        })
        logger.debug(f"Added obstacle at {position} with radius {radius}m")
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles = []
        logger.debug("Cleared all obstacles")
    
    def _attractive_force(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Calculate attractive force toward goal."""
        distance = np.linalg.norm(goal - current)
        
        # If we're close enough to goal, no force needed
        if distance < self.config.goal_threshold:
            return np.zeros(3)
        
        # Direction to goal
        direction = (goal - current) / distance
        
        # Linear attractive force
        force = self.config.attractive_gain * direction
        
        return force
    
    def _repulsive_force(self, current: np.ndarray) -> np.ndarray:
        """Calculate repulsive force from obstacles."""
        total_force = np.zeros(3)
        
        for obstacle in self.obstacles:
            # Vector from obstacle to current position
            to_current = current - obstacle["position"]
            distance = np.linalg.norm(to_current)
            
            # If outside influence radius, skip
            if distance > self.config.influence_radius:
                continue
            
            # Adjust for obstacle radius
            distance = max(distance - obstacle["radius"], 0.1)  # Avoid division by zero
            
            # Direction away from obstacle
            if distance > 0:
                direction = to_current / np.linalg.norm(to_current)
            else:
                # If we're at the obstacle center (unlikely), pick random direction
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
            
            # Force is stronger when closer to obstacle
            magnitude = self.config.repulsive_gain * (1.0/distance - 1.0/self.config.influence_radius)
            magnitude = max(0, magnitude)  # Ensure non-negative
            
            # Add to total force
            total_force += magnitude * direction
        
        return total_force
    
    def compute_avoidance_vector(self, current: List[float], 
                               goal: List[float]) -> List[float]:
        """
        Compute avoidance vector using potential field method.
        
        Args:
            current: Current position [x, y, z]
            goal: Goal position [x, y, z]
            
        Returns:
            List[float]: Direction vector for safe movement
        """
        current_pos = np.array(current)
        goal_pos = np.array(goal)
        
        # Calculate forces
        attractive = self._attractive_force(current_pos, goal_pos)
        repulsive = self._repulsive_force(current_pos)
        
        # Combine forces
        total_force = attractive + repulsive
        
        # Normalize if non-zero
        magnitude = np.linalg.norm(total_force)
        if magnitude > 0:
            total_force = total_force / magnitude
        
        logger.debug(f"Computed avoidance vector: {total_force}")
        return total_force.tolist()
    
    def get_next_waypoint(self, current: List[float], goal: List[float], 
                        step_size: float = 5.0) -> List[float]:
        """
        Get next waypoint that avoids obstacles.
        
        Args:
            current: Current position [x, y, z]
            goal: Goal position [x, y, z]
            step_size: Distance to move in computed direction
            
        Returns:
            List[float]: Next waypoint position
        """
        # Get avoidance direction
        direction = self.compute_avoidance_vector(current, goal)
        
        # Compute next position
        next_pos = np.array(current) + np.array(direction) * step_size
        
        logger.debug(f"Next waypoint: {next_pos}")
        return next_pos.tolist()
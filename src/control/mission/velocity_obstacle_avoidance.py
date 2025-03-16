"""
Velocity Obstacle Avoidance System

Implements a simplified velocity obstacle approach for dynamic obstacle avoidance.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("velocity_obstacle_avoidance")


@dataclass
class DynamicObstacle:
    """Dynamic obstacle representation."""
    
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    radius: float         # Obstacle radius in meters


class VelocityObstacleAvoidance:
    """Simple velocity obstacle based avoidance."""
    
    def __init__(self, safety_margin: float = 5.0, time_horizon: float = 5.0):
        """Initialize velocity obstacle avoidance system."""
        self.obstacles: List[DynamicObstacle] = []
        self.safety_margin = safety_margin
        self.time_horizon = time_horizon
        self.max_speed = 20.0  # Maximum speed in m/s
        
        logger.info("Initialized velocity obstacle avoidance system")
    
    def add_obstacle(self, position: List[float], velocity: List[float], radius: float) -> None:
        """Add dynamic obstacle to the environment."""
        self.obstacles.append(DynamicObstacle(
            position=np.array(position),
            velocity=np.array(velocity),
            radius=radius
        ))
        logger.debug(f"Added obstacle at {position} with velocity {velocity} and radius {radius}m")
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles = []
        logger.debug("Cleared all obstacles")
    
    def is_velocity_safe(self, current_pos: np.ndarray, velocity: np.ndarray) -> bool:
        """Check if a velocity is safe from collisions."""
        for obstacle in self.obstacles:
            # Relative position and velocity
            rel_pos = obstacle.position - current_pos
            rel_vel = obstacle.velocity - velocity
            
            # Distance to closest approach
            if np.linalg.norm(rel_vel) < 0.001:
                # If relative velocity is near zero, use current distance
                closest_dist = np.linalg.norm(rel_pos)
            else:
                # Time to closest approach
                tc = max(0, -np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel))
                tc = min(tc, self.time_horizon)  # Limit to time horizon
                
                # Position at closest approach
                closest_pos = rel_pos + tc * rel_vel
                closest_dist = np.linalg.norm(closest_pos)
            
            # Check if distance is less than combined radii plus safety margin
            if closest_dist < (obstacle.radius + self.safety_margin):
                return False
        
        return True
    
    def find_safe_velocity(self, current_pos: List[float], 
                         desired_vel: List[float], 
                         num_samples: int = 10) -> List[float]:
        """
        Find a safe velocity close to the desired velocity.
        
        Args:
            current_pos: Current position [x, y, z]
            desired_vel: Desired velocity [vx, vy, vz]
            num_samples: Number of alternative velocities to sample
            
        Returns:
            List[float]: Safe velocity vector
        """
        current_pos_np = np.array(current_pos)
        desired_vel_np = np.array(desired_vel)
        
        # Check if desired velocity is already safe
        if self.is_velocity_safe(current_pos_np, desired_vel_np):
            return desired_vel
        
        # If not safe, sample alternative velocities
        best_vel = np.zeros(3)
        best_score = float('-inf')
        
        # Desired speed
        desired_speed = np.linalg.norm(desired_vel_np)
        if desired_speed < 0.001:
            desired_speed = 0.001
        
        # Desired direction
        desired_dir = desired_vel_np / desired_speed
        
        # Sample velocities at different angles and speeds
        for speed_factor in np.linspace(0.5, 1.0, 5):
            speed = desired_speed * speed_factor
            
            for i in range(num_samples):
                # Random perturbation to direction
                angle = 2 * np.pi * i / num_samples
                
                # Create rotation matrix around vertical axis
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                rotation = np.array([
                    [cos_angle, 0, sin_angle],
                    [0, 1, 0],
                    [-sin_angle, 0, cos_angle]
                ])
                
                # Apply rotation to desired direction
                new_dir = rotation @ desired_dir
                new_vel = new_dir * speed
                
                # Check if velocity is safe
                if self.is_velocity_safe(current_pos_np, new_vel):
                    # Score based on similarity to desired velocity
                    similarity = np.dot(new_vel, desired_vel_np) / (np.linalg.norm(new_vel) * np.linalg.norm(desired_vel_np))
                    score = similarity * speed / desired_speed
                    
                    if score > best_score:
                        best_score = score
                        best_vel = new_vel
        
        # If no safe velocity found, slow down
        if best_score == float('-inf'):
            logger.warning("No safe velocity found, slowing down")
            return (desired_vel_np * 0.5).tolist()
        
        logger.debug(f"Found safe velocity: {best_vel}")
        return best_vel.tolist()
    
    def compute_avoidance_velocity(self, current_pos: List[float], 
                                 goal_pos: List[float],
                                 current_vel: List[float],
                                 max_speed: Optional[float] = None) -> List[float]:
        """
        Compute a safe velocity to avoid obstacles while moving toward goal.
        
        Args:
            current_pos: Current position [x, y, z]
            goal_pos: Goal position [x, y, z]
            current_vel: Current velocity [vx, vy, vz]
            max_speed: Maximum allowed speed (optional)
            
        Returns:
            List[float]: Safe velocity vector
        """
        if max_speed is None:
            max_speed = self.max_speed
        
        # Compute desired velocity toward goal
        current_pos_np = np.array(current_pos)
        goal_pos_np = np.array(goal_pos)
        
        to_goal = goal_pos_np - current_pos_np
        distance = np.linalg.norm(to_goal)
        
        if distance < 0.001:
            # Already at goal
            return [0, 0, 0]
        
        # Desired direction
        desired_dir = to_goal / distance
        
        # Desired speed (proportional to distance, up to max_speed)
        desired_speed = min(distance, max_speed)
        
        # Desired velocity
        desired_vel = desired_dir * desired_speed
        
        # Find safe velocity
        safe_vel = self.find_safe_velocity(current_pos, desired_vel.tolist())
        
        return safe_vel
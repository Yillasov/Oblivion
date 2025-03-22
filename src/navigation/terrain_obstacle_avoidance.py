"""
Terrain-Based Obstacle Avoidance for UCAV platforms.

Integrates terrain data with obstacle avoidance algorithms for enhanced
terrain-following navigation capabilities.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from src.navigation.terrain_database import TerrainDatabase
from src.navigation.predictive_terrain import PredictiveTerrainModeling, TerrainRiskLevel
from src.navigation.error_handling import safe_navigation_operation

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class TerrainObstacle:
    """Terrain obstacle representation."""
    position: np.ndarray  # [x, y, z] in meters
    radius: float         # Obstacle radius in meters
    height: float         # Height of the obstacle
    risk_level: TerrainRiskLevel


class TerrainObstacleAvoidance:
    """
    Terrain-based obstacle avoidance system.
    
    Integrates terrain data with obstacle avoidance algorithms.
    """
    
    def __init__(self, 
                terrain_db: TerrainDatabase,
                safety_margin: float = 50.0,
                lookahead_distance: float = 3000.0):
        """
        Initialize terrain obstacle avoidance.
        
        Args:
            terrain_db: Terrain database
            safety_margin: Safety margin in meters
            lookahead_distance: Distance to look ahead in meters
        """
        self.terrain_db = terrain_db
        self.safety_margin = safety_margin
        self.lookahead_distance = lookahead_distance
        self.terrain_predictor = PredictiveTerrainModeling(
            terrain_db=terrain_db,
            lookahead_distance=lookahead_distance
        )
        self.obstacles: List[TerrainObstacle] = []
        
        logger.info(f"Initialized terrain obstacle avoidance system")
    
    def update_terrain_obstacles(self, 
                               position: Tuple[float, float, float],
                               heading: float,
                               terrain_name: str = "default") -> None:
        """
        Update terrain obstacles based on current position and heading.
        
        Args:
            position: Current [x, y, z] position
            heading: Current heading in degrees
            terrain_name: Name of terrain to use
        """
        # Clear previous obstacles
        self.obstacles = []
        
        # Get terrain predictions
        predictions = self.terrain_predictor.predict_terrain(
            position=(position[0], position[1]),
            heading=heading,
            terrain_name=terrain_name
        )
        
        # Convert high-risk terrain areas to obstacles
        for prediction in predictions:
            if prediction.risk_level in [TerrainRiskLevel.HIGH, TerrainRiskLevel.EXTREME]:
                # Create obstacle at this position
                obstacle = TerrainObstacle(
                    position=np.array([prediction.position[0], prediction.position[1], prediction.height]),
                    radius=100.0,  # Default obstacle radius
                    height=prediction.height,
                    risk_level=prediction.risk_level
                )
                self.obstacles.append(obstacle)
        
        logger.debug(f"Updated terrain obstacles: {len(self.obstacles)} high-risk areas identified")
    
    def find_safe_path(self, 
                     current_pos: Tuple[float, float, float],
                     target_pos: Tuple[float, float, float],
                     heading: float,
                     terrain_name: str = "default") -> Dict[str, Any]:
        """
        Find a safe path avoiding terrain obstacles.
        
        Args:
            current_pos: Current [x, y, z] position
            target_pos: Target [x, y, z] position
            heading: Current heading in degrees
            terrain_name: Name of terrain to use
            
        Returns:
            Dictionary with safe path information
        """
        # Update terrain obstacles
        self.update_terrain_obstacles(current_pos, heading, terrain_name)
        
        # Get optimal path from terrain predictor
        optimal_path = self.terrain_predictor.get_optimal_path(
            position=(current_pos[0], current_pos[1]),
            heading=heading,
            terrain_name=terrain_name
        )
        
        # Check if path is clear
        is_path_clear = self.check_path(current_pos, target_pos)
        
        # If path is not clear, find avoidance waypoint
        waypoints = []
        if not is_path_clear:
            avoidance_point = self.find_avoidance_waypoint(current_pos, target_pos)
            if avoidance_point:
                waypoints.append(avoidance_point)
        
        return {
            "optimal_altitude": optimal_path.get("optimal_altitude", current_pos[2] + self.safety_margin),
            "is_path_clear": is_path_clear,
            "avoidance_waypoints": waypoints,
            "high_risk_areas": optimal_path.get("high_risk_areas", []),
            "recommended_heading_change": self._calculate_heading_change(current_pos, target_pos, heading)
        }
    
    def check_path(self, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> bool:
        """
        Check if path between start and end is clear of terrain obstacles.
        
        Args:
            start: Start position [x, y, z]
            end: End position [x, y, z]
            
        Returns:
            True if path is clear, False if obstacle detected
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
            if obstacle_distance < (obstacle.radius + self.safety_margin):
                logger.debug(f"Path blocked by terrain obstacle at {obstacle.position}")
                return False
        
        return True
    
    def find_avoidance_waypoint(self, 
                              current: Tuple[float, float, float], 
                              target: Tuple[float, float, float]) -> Optional[List[float]]:
        """
        Find a waypoint to avoid terrain obstacles.
        
        Args:
            current: Current position [x, y, z]
            target: Target position [x, y, z]
            
        Returns:
            Avoidance waypoint or None if no path found
        """
        current_pos = np.array(current)
        target_pos = np.array(target)
        
        # If path is clear, no need for avoidance
        if self.check_path(current, target):
            return None
        
        # Try vertical avoidance first (fly higher)
        vertical_point = np.array([
            current_pos[0], 
            current_pos[1],
            current_pos[2] + self.safety_margin * 2
        ])
        
        if self.check_path(tuple(vertical_point), target):
            return vertical_point.tolist()
        
        # Try lateral avoidance for each obstacle
        for obstacle in self.obstacles:
            # Vector from obstacle to current position
            to_current = current_pos - obstacle.position
            
            # Vector from obstacle to target
            to_target = target_pos - obstacle.position
            
            # Compute avoidance direction (perpendicular to path)
            avoidance_dir = np.cross(np.cross(to_current, to_target), to_target)
            
            if np.linalg.norm(avoidance_dir) > 0:
                avoidance_dir = avoidance_dir / np.linalg.norm(avoidance_dir)
                
                # Compute avoidance waypoint
                avoidance_distance = obstacle.radius + self.safety_margin * 2
                avoidance_point = obstacle.position + avoidance_dir * avoidance_distance
                
                # Set altitude to be safe
                avoidance_point[2] = max(avoidance_point[2], obstacle.height + self.safety_margin)
                
                # Check if this avoidance path is clear
                if self.check_path(current, avoidance_point.tolist()) and \
                   self.check_path(avoidance_point.tolist(), target):
                    return avoidance_point.tolist()
        
        # If no good path found, just go higher
        return [current_pos[0], current_pos[1], current_pos[2] + self.safety_margin * 3]
    
    def _calculate_heading_change(self, 
                                current: Tuple[float, float, float],
                                target: Tuple[float, float, float],
                                current_heading: float) -> float:
        """Calculate recommended heading change to avoid obstacles."""
        # If path is clear, no heading change needed
        if self.check_path(current, target):
            return 0.0
        
        # Find avoidance waypoint
        waypoint = self.find_avoidance_waypoint(current, target)
        if not waypoint:
            return 0.0
        
        # Calculate heading to waypoint
        current_pos = np.array(current)
        waypoint_pos = np.array(waypoint)
        
        # Get 2D direction vector
        direction = waypoint_pos[:2] - current_pos[:2]
        
        # Calculate heading in degrees
        waypoint_heading = np.degrees(np.arctan2(direction[1], direction[0]))
        
        # Calculate heading change (normalized to -180 to 180)
        heading_change = waypoint_heading - current_heading
        while heading_change > 180:
            heading_change -= 360
        while heading_change < -180:
            heading_change += 360
            
        return heading_change
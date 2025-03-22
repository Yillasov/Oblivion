"""
Predictive Terrain Modeling for UCAV platforms.

Provides functionality to analyze and predict terrain features ahead of
the aircraft for terrain-following navigation and obstacle avoidance.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.navigation.terrain_database import TerrainDatabase, TerrainFeature, TerrainFeatureType
from src.navigation.error_handling import safe_navigation_operation

# Configure logger
logger = logging.getLogger(__name__)


class TerrainRiskLevel(Enum):
    """Risk levels for terrain features."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class TerrainPrediction:
    """Terrain prediction data."""
    position: Tuple[float, float]  # x, y coordinates
    distance: float  # distance from current position
    height: float  # terrain height
    slope: float  # terrain slope in degrees
    risk_level: TerrainRiskLevel
    features: List[Dict[str, Any]]  # nearby terrain features
    recommended_altitude: float  # recommended flight altitude


class PredictiveTerrainModeling:
    """
    Predictive terrain modeling system.
    
    Analyzes terrain ahead of the aircraft and predicts optimal flight paths.
    """
    
    def __init__(self, terrain_db: TerrainDatabase, lookahead_distance: float = 5000.0):
        """
        Initialize predictive terrain modeling.
        
        Args:
            terrain_db: Terrain database
            lookahead_distance: Distance to look ahead in meters
        """
        self.terrain_db = terrain_db
        self.lookahead_distance = lookahead_distance
        self.sample_points = 20  # Number of points to sample along path
        self.current_terrain = "default"
        self.safety_margin = 100.0  # meters above terrain
        
        # Risk level thresholds
        self.slope_thresholds = {
            TerrainRiskLevel.LOW: 10.0,      # degrees
            TerrainRiskLevel.MEDIUM: 20.0,   # degrees
            TerrainRiskLevel.HIGH: 30.0      # degrees
        }
        
        logger.info(f"Initialized predictive terrain modeling with {lookahead_distance}m lookahead")
    
    @safe_navigation_operation
    def predict_terrain(self, 
                       position: Tuple[float, float], 
                       heading: float,
                       terrain_name: str = None) -> List[TerrainPrediction]:
        """
        Predict terrain ahead of the aircraft.
        
        Args:
            position: Current (x, y) position
            heading: Current heading in degrees
            terrain_name: Name of terrain to use
            
        Returns:
            List of terrain predictions along the path
        """
        # Use provided terrain name or default
        terrain_name = terrain_name or self.current_terrain
        
        # Convert heading to radians
        heading_rad = np.radians(heading)
        
        # Calculate direction vector
        direction = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        
        # Generate sample points along the path
        predictions = []
        
        for i in range(1, self.sample_points + 1):
            # Calculate distance for this sample
            distance = (i / self.sample_points) * self.lookahead_distance
            
            # Calculate position
            sample_position = (
                position[0] + direction[0] * distance,
                position[1] + direction[1] * distance
            )
            
            # Get terrain data at this position
            terrain_data = self.terrain_db.get_terrain_at_position(
                terrain_name, sample_position, radius=100.0
            )
            
            if "error" in terrain_data:
                logger.warning(f"Error getting terrain data: {terrain_data['error']}")
                continue
            
            # Calculate slope
            if i > 1 and len(predictions) > 0:
                prev_height = predictions[-1].height
                height_diff = terrain_data["height"] - prev_height
                horizontal_dist = distance - ((i-1) / self.sample_points) * self.lookahead_distance
                slope = np.degrees(np.arctan2(height_diff, horizontal_dist))
            else:
                slope = 0.0
            
            # Determine risk level based on slope and terrain features
            risk_level = self._calculate_risk_level(slope, terrain_data)
            
            # Calculate recommended altitude
            recommended_altitude = self._calculate_recommended_altitude(
                terrain_data, risk_level
            )
            
            # Create prediction
            prediction = TerrainPrediction(
                position=sample_position,
                distance=distance,
                height=terrain_data["height"],
                slope=slope,
                risk_level=risk_level,
                features=terrain_data.get("nearby_features", []),
                recommended_altitude=recommended_altitude
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _calculate_risk_level(self, slope: float, terrain_data: Dict[str, Any]) -> TerrainRiskLevel:
        """Calculate risk level based on slope and terrain features."""
        # Start with low risk
        risk_level = TerrainRiskLevel.LOW
        
        # Adjust based on absolute slope
        abs_slope = abs(slope)
        if abs_slope > self.slope_thresholds[TerrainRiskLevel.HIGH]:
            risk_level = TerrainRiskLevel.EXTREME
        elif abs_slope > self.slope_thresholds[TerrainRiskLevel.MEDIUM]:
            risk_level = TerrainRiskLevel.HIGH
        elif abs_slope > self.slope_thresholds[TerrainRiskLevel.LOW]:
            risk_level = TerrainRiskLevel.MEDIUM
        
        # Check for nearby features that might increase risk
        for feature in terrain_data.get("nearby_features", []):
            feature_type = feature.get("type", "")
            distance = feature.get("distance", 1000.0)
            
            # Increase risk for mountains and canyons that are close
            if (feature_type in ["mountain", "canyon", "ridge"] and 
                distance < 200.0 and 
                risk_level.value != TerrainRiskLevel.EXTREME.value):
                # Increase risk by one level
                if risk_level == TerrainRiskLevel.LOW:
                    risk_level = TerrainRiskLevel.MEDIUM
                elif risk_level == TerrainRiskLevel.MEDIUM:
                    risk_level = TerrainRiskLevel.HIGH
                elif risk_level == TerrainRiskLevel.HIGH:
                    risk_level = TerrainRiskLevel.EXTREME
        
        return risk_level
    
    def _calculate_recommended_altitude(self, 
                                      terrain_data: Dict[str, Any],
                                      risk_level: TerrainRiskLevel) -> float:
        """Calculate recommended altitude based on terrain and risk."""
        # Base altitude is terrain height plus safety margin
        base_altitude = terrain_data["height"] + self.safety_margin
        
        # Add additional margin based on risk level
        if risk_level == TerrainRiskLevel.MEDIUM:
            base_altitude += 50.0
        elif risk_level == TerrainRiskLevel.HIGH:
            base_altitude += 100.0
        elif risk_level == TerrainRiskLevel.EXTREME:
            base_altitude += 200.0
        
        # Add additional margin based on terrain variability
        height_range = terrain_data.get("height_range", 0.0)
        base_altitude += height_range * 0.2
        
        return base_altitude
    
    def get_optimal_path(self, 
                        position: Tuple[float, float],
                        heading: float,
                        terrain_name: str = None) -> Dict[str, Any]:
        """
        Calculate optimal path through terrain.
        
        Args:
            position: Current (x, y) position
            heading: Current heading in degrees
            terrain_name: Name of terrain to use
            
        Returns:
            Dictionary with optimal path information
        """
        # Predict terrain along current heading
        predictions = self.predict_terrain(position, heading, terrain_name)
        
        if not predictions:
            return {"error": "No terrain predictions available"}
        
        # Find highest risk areas
        high_risk_areas = [p for p in predictions if p.risk_level in 
                          [TerrainRiskLevel.HIGH, TerrainRiskLevel.EXTREME]]
        
        # Calculate optimal altitude profile
        altitude_profile = [p.recommended_altitude for p in predictions]
        
        # Find maximum recommended altitude
        max_altitude = max(altitude_profile) if altitude_profile else 0.0
        
        # Calculate average risk level (numeric representation)
        risk_values = {
            TerrainRiskLevel.LOW: 1,
            TerrainRiskLevel.MEDIUM: 2,
            TerrainRiskLevel.HIGH: 3,
            TerrainRiskLevel.EXTREME: 4
        }
        avg_risk = sum(risk_values[p.risk_level] for p in predictions) / len(predictions)
        
        return {
            "optimal_altitude": max_altitude,
            "altitude_profile": altitude_profile,
            "high_risk_areas": [
                {
                    "distance": area.distance,
                    "position": area.position,
                    "risk_level": area.risk_level.value
                } for area in high_risk_areas
            ],
            "average_risk": avg_risk,
            "predictions": [
                {
                    "distance": p.distance,
                    "height": p.height,
                    "recommended_altitude": p.recommended_altitude,
                    "risk_level": p.risk_level.value
                } for p in predictions
            ]
        }
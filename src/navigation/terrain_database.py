"""
Terrain Database Management for UCAV platforms.

Provides functionality to store, retrieve, and analyze terrain data
for terrain-following navigation and mission planning.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.simulation.environment.terrain import TerrainModel, TerrainConfig
from src.navigation.error_handling import safe_navigation_operation

# Configure logger
logger = logging.getLogger(__name__)


class TerrainFeatureType(Enum):
    """Types of terrain features."""
    MOUNTAIN = "mountain"
    VALLEY = "valley"
    RIDGE = "ridge"
    PLATEAU = "plateau"
    CANYON = "canyon"
    PLAIN = "plain"
    WATER = "water"


@dataclass
class TerrainFeature:
    """Terrain feature data."""
    feature_id: int
    feature_type: TerrainFeatureType
    center_position: Tuple[float, float]  # x, y coordinates
    elevation: float
    size: float  # approximate radius in meters
    steepness: float  # average slope in degrees
    metadata: Dict[str, Any] = {}


class TerrainDatabase:
    """
    Terrain database management system.
    
    Stores and retrieves terrain data for navigation and mission planning.
    """
    
    def __init__(self, database_path: str = "/Users/yessine/Oblivion/data/terrain"):
        """
        Initialize terrain database.
        
        Args:
            database_path: Path to terrain database files
        """
        self.database_path = database_path
        self.terrain_models: Dict[str, TerrainModel] = {}
        self.terrain_features: Dict[str, List[TerrainFeature]] = {}
        
        # Create database directory if it doesn't exist
        os.makedirs(self.database_path, exist_ok=True)
        
        logger.info(f"Initialized terrain database at {self.database_path}")
    
    @safe_navigation_operation
    def load_terrain(self, terrain_name: str) -> Optional[TerrainModel]:
        """
        Load a terrain model by name.
        
        Args:
            terrain_name: Name of the terrain to load
            
        Returns:
            TerrainModel if successful, None otherwise
        """
        # Check if already loaded
        if terrain_name in self.terrain_models:
            return self.terrain_models[terrain_name]
        
        # Determine file path
        terrain_path = os.path.join(self.database_path, f"{terrain_name}.json")
        
        # Check if file exists
        if not os.path.exists(terrain_path):
            logger.warning(f"Terrain file not found: {terrain_path}")
            return None
        
        try:
            # Load terrain configuration from file
            with open(terrain_path, 'r') as f:
                terrain_data = json.load(f)
            
            # Create terrain config
            config = TerrainConfig(
                width=terrain_data.get("width", 10000.0),
                length=terrain_data.get("length", 10000.0),
                resolution=terrain_data.get("resolution", 129),
                min_height=terrain_data.get("min_height", 0.0),
                max_height=terrain_data.get("max_height", 1000.0),
                roughness=terrain_data.get("roughness", 0.5),
                seed=terrain_data.get("seed", 42)
            )
            
            # Create terrain model
            terrain_model = TerrainModel(config)
            
            # Store in cache
            self.terrain_models[terrain_name] = terrain_model
            
            # Load features if available
            features_path = os.path.join(self.database_path, f"{terrain_name}_features.json")
            if os.path.exists(features_path):
                self._load_terrain_features(terrain_name, features_path)
            
            logger.info(f"Loaded terrain model '{terrain_name}'")
            return terrain_model
            
        except Exception as e:
            logger.error(f"Error loading terrain '{terrain_name}': {str(e)}")
            return None
    
    def save_terrain(self, terrain_name: str, terrain_model: TerrainModel) -> bool:
        """
        Save a terrain model to file.
        
        Args:
            terrain_name: Name of the terrain to save
            terrain_model: Terrain model to save
            
        Returns:
            Success status
        """
        # Determine file path
        terrain_path = os.path.join(self.database_path, f"{terrain_name}.json")
        
        try:
            # Convert terrain model to dictionary
            terrain_data = {
                "width": terrain_model.config.width,
                "length": terrain_model.config.length,
                "resolution": terrain_model.config.resolution,
                "min_height": terrain_model.config.min_height,
                "max_height": terrain_model.config.max_height,
                "roughness": terrain_model.config.roughness,
                "seed": terrain_model.config.seed
            }
            
            # Save to file
            with open(terrain_path, 'w') as f:
                json.dump(terrain_data, f, indent=2)
            
            # Store in cache
            self.terrain_models[terrain_name] = terrain_model
            
            logger.info(f"Saved terrain model '{terrain_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error saving terrain '{terrain_name}': {str(e)}")
            return False
    
    def _load_terrain_features(self, terrain_name: str, features_path: str) -> None:
        """Load terrain features from file."""
        try:
            with open(features_path, 'r') as f:
                features_data = json.load(f)
            
            features = []
            for feature_data in features_data:
                feature = TerrainFeature(
                    feature_id=feature_data.get("id", 0),
                    feature_type=TerrainFeatureType(feature_data.get("type", "plain")),
                    center_position=(
                        feature_data.get("position", [0, 0])[0],
                        feature_data.get("position", [0, 0])[1]
                    ),
                    elevation=feature_data.get("elevation", 0.0),
                    size=feature_data.get("size", 100.0),
                    steepness=feature_data.get("steepness", 0.0),
                    metadata=feature_data.get("metadata", {})
                )
                features.append(feature)
            
            self.terrain_features[terrain_name] = features
            logger.info(f"Loaded {len(features)} terrain features for '{terrain_name}'")
            
        except Exception as e:
            logger.error(f"Error loading terrain features for '{terrain_name}': {str(e)}")
    
    def extract_terrain_features(self, terrain_name: str, terrain_model: TerrainModel) -> List[TerrainFeature]:
        """
        Extract terrain features from a terrain model.
        
        Args:
            terrain_name: Name of the terrain
            terrain_model: Terrain model to analyze
            
        Returns:
            List of terrain features
        """
        # Simple feature extraction algorithm
        features = []
        feature_id = 0
        
        # Sample the terrain at lower resolution for feature detection
        sample_res = min(50, terrain_model.config.resolution // 4)
        height_samples = np.zeros((sample_res, sample_res))
        
        for i in range(sample_res):
            for j in range(sample_res):
                # Convert to terrain coordinates
                x = terrain_model.origin_x + (i / sample_res) * terrain_model.config.width
                y = terrain_model.origin_y + (j / sample_res) * terrain_model.config.length
                height_samples[i, j] = terrain_model.get_height(x, y)
        
        # Find local maxima (mountains)
        for i in range(1, sample_res-1):
            for j in range(1, sample_res-1):
                # Check if higher than all neighbors
                is_peak = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if height_samples[i, j] <= height_samples[i+di, j+dj]:
                            is_peak = False
                            break
                    if not is_peak:
                        break
                
                if is_peak:
                    # Convert to terrain coordinates
                    x = terrain_model.origin_x + (i / sample_res) * terrain_model.config.width
                    y = terrain_model.origin_y + (j / sample_res) * terrain_model.config.length
                    
                    # Calculate steepness (average slope to neighbors)
                    slopes = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            
                            # Calculate horizontal distance
                            dx = di * (terrain_model.config.width / sample_res)
                            dy = dj * (terrain_model.config.length / sample_res)
                            dist = np.sqrt(dx**2 + dy**2)
                            
                            # Calculate height difference
                            dh = height_samples[i, j] - height_samples[i+di, j+dj]
                            
                            # Calculate slope in degrees
                            slope = np.degrees(np.arctan2(dh, dist))
                            slopes.append(slope)
                    
                    avg_slope = np.mean(slopes)
                    
                    # Create mountain feature
                    feature = TerrainFeature(
                        feature_id=feature_id,
                        feature_type=TerrainFeatureType.MOUNTAIN,
                        center_position=(x, y),
                        elevation=height_samples[i, j],
                        size=200.0,  # Approximate size
                        steepness=float(avg_slope)
                    )
                    
                    features.append(feature)
                    feature_id += 1
        
        # Store features
        self.terrain_features[terrain_name] = features
        
        # Save features to file
        features_path = os.path.join(self.database_path, f"{terrain_name}_features.json")
        try:
            features_data = []
            for feature in features:
                features_data.append({
                    "id": feature.feature_id,
                    "type": feature.feature_type.value,
                    "position": list(feature.center_position),
                    "elevation": feature.elevation,
                    "size": feature.size,
                    "steepness": feature.steepness,
                    "metadata": feature.metadata or {}
                })
            
            with open(features_path, 'w') as f:
                json.dump(features_data, f, indent=2)
            
            logger.info(f"Saved {len(features)} terrain features for '{terrain_name}'")
            
        except Exception as e:
            logger.error(f"Error saving terrain features for '{terrain_name}': {str(e)}")
        
        return features
    
    def get_terrain_at_position(self, terrain_name: str, position: Tuple[float, float], radius: float = 1000.0) -> Dict[str, Any]:
        """
        Get terrain data around a specific position.
        
        Args:
            terrain_name: Name of the terrain
            position: (x, y) coordinates
            radius: Radius around position to analyze
            
        Returns:
            Dictionary with terrain data
        """
        terrain_model = self.load_terrain(terrain_name)
        if not terrain_model:
            return {"error": "Terrain not found"}
        
        # Get height at position
        x, y = position
        height = terrain_model.get_height(x, y)
        
        # Sample heights in a grid around position
        sample_points = 20
        heights = []
        
        for i in range(sample_points):
            for j in range(sample_points):
                # Calculate sample position
                sample_x = x + (i - sample_points/2) * (radius / sample_points)
                sample_y = y + (j - sample_points/2) * (radius / sample_points)
                
                # Get height
                sample_height = terrain_model.get_height(sample_x, sample_y)
                heights.append(sample_height)
        
        # Calculate statistics
        min_height = min(heights)
        max_height = max(heights)
        avg_height = sum(heights) / len(heights)
        
        # Find nearby features
        nearby_features = []
        if terrain_name in self.terrain_features:
            for feature in self.terrain_features[terrain_name]:
                feature_x, feature_y = feature.center_position
                distance = np.sqrt((x - feature_x)**2 + (y - feature_y)**2)
                
                if distance <= radius + feature.size:
                    nearby_features.append({
                        "id": feature.feature_id,
                        "type": feature.feature_type.value,
                        "distance": distance,
                        "elevation": feature.elevation,
                        "steepness": feature.steepness
                    })
        
        return {
            "position": position,
            "height": height,
            "min_height": min_height,
            "max_height": max_height,
            "avg_height": avg_height,
            "height_range": max_height - min_height,
            "nearby_features": nearby_features
        }
    
    def create_default_terrain(self, terrain_name: str = "default") -> TerrainModel:
        """
        Create a default terrain model.
        
        Args:
            terrain_name: Name for the terrain
            
        Returns:
            Created terrain model
        """
        from src.simulation.environment.terrain import create_default_terrain
        
        # Create default terrain
        terrain_model = create_default_terrain()
        
        # Save to database
        self.save_terrain(terrain_name, terrain_model)
        
        # Extract features
        self.extract_terrain_features(terrain_name, terrain_model)
        
        return terrain_model
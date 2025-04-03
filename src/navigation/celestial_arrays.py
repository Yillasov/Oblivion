"""
Celestial Navigation Arrays for UCAV platforms.

Provides navigation capabilities using arrays of celestial bodies
for precise positioning in GPS-denied environments.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.navigation.error_handling import safe_navigation_operation
from src.navigation.common import PositionProvider
from src.navigation.base import NavigationSystem, NavigationSpecs
from src.navigation.celestial_database import CelestialDatabase
from src.navigation.star_tracker import StarCatalogEntry

# Configure logger
logger = logging.getLogger(__name__)


class CelestialArrayType(Enum):
    """Types of celestial navigation arrays."""
    STAR_ARRAY = "star_array"
    PLANET_ARRAY = "planet_array"
    MIXED_ARRAY = "mixed_array"


@dataclass
class CelestialArrayConfig:
    """Configuration for celestial navigation arrays."""
    array_type: CelestialArrayType = CelestialArrayType.STAR_ARRAY
    min_celestial_bodies: int = 3
    max_celestial_bodies: int = 7
    integration_time: float = 2.0  # seconds
    catalog_name: str = "navigation"
    position_filter_strength: float = 0.4


class CelestialNavigationArrays(NavigationSystem, PositionProvider):
    """
    Celestial Navigation Arrays system.
    
    Uses arrays of celestial bodies for precise navigation
    in GPS-denied environments.
    """
    
    def __init__(self, config: CelestialArrayConfig = CelestialArrayConfig()):
        """
        Initialize celestial navigation arrays.
        
        Args:
            config: System configuration
        """
        specs = NavigationSpecs(
            weight=1.8,  # kg
            volume={"value": 0.005},  # m³
            power_requirements=3.6,  # Watts (12V * 0.3A)
            drift_rate=0.0005,  # m/s
            initialization_time=15.0,  # seconds
            accuracy={"value": 0.5},  # km
            update_rate=0.5  # Hz
        )
        super().__init__(specs)
        
        self.config = config
        
        # Celestial database
        self.celestial_db = CelestialDatabase()
        
        # Current celestial bodies used for navigation
        self.active_bodies: List[StarCatalogEntry] = []
        
        # Current position and uncertainty
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.position_accuracy = 500.0  # meters
        
        # Kalman filter state
        self.position_filter = np.zeros(3)
        self.position_covariance = np.eye(3) * 1000.0
        
        logger.info(f"Initialized celestial navigation arrays with {config.array_type.value}")
    
    @safe_navigation_operation
    def initialize(self) -> bool:
        """
        Initialize the celestial navigation arrays.
        
        Returns:
            Success status
        """
        if self.initialized:
            return True
            
        logger.info("Initializing celestial navigation arrays")
        
        # Initialize base system
        super().initialize()
        
        # Load celestial catalog
        self._load_celestial_catalog()
        
        # Select initial celestial bodies for navigation
        self._select_celestial_bodies()
        
        # Update status
        self.status["operational"] = True
        self.initialized = True
        
        logger.info("Celestial navigation arrays initialized successfully")
        return True
    
    def _load_celestial_catalog(self) -> None:
        """Load celestial catalog data."""
        catalog = self.celestial_db.load_catalog(self.config.catalog_name)
        
        # If catalog doesn't exist, create default
        if not catalog:
            logger.info(f"Creating default celestial catalog '{self.config.catalog_name}'")
            catalog = self.celestial_db.create_default_catalog(self.config.catalog_name)
    
    def _select_celestial_bodies(self) -> None:
        """Select optimal celestial bodies for navigation."""
        catalog = self.celestial_db.load_catalog(self.config.catalog_name)
        
        # Sort by brightness (lower magnitude is brighter)
        sorted_catalog = sorted(catalog, key=lambda x: x.magnitude)
        
        # Select brightest bodies within limit
        num_bodies = min(self.config.max_celestial_bodies, len(sorted_catalog))
        self.active_bodies = sorted_catalog[:num_bodies]
        
        logger.info(f"Selected {len(self.active_bodies)} celestial bodies for navigation")
    
    @safe_navigation_operation
    def update(self, delta_time: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update position estimate using celestial navigation.
        
        Args:
            delta_time: Time since last update in seconds
            environment_data: Optional environment data
            
        Returns:
            Current position state
        """
        if not self.active:
            return {"error": "Celestial navigation arrays not active"}
        
        # Get celestial observations from environment data
        observations = self._extract_observations(environment_data)
        
        # If we have enough observations, calculate position
        if len(observations) >= self.config.min_celestial_bodies:
            # Calculate position from observations
            self._calculate_position(observations)
            
            # Update position accuracy
            self._update_position_accuracy(len(observations))
        else:
            # Not enough celestial bodies visible
            self.position_accuracy = min(self.position_accuracy * 1.2, 10000.0)  # Degrade accuracy
        
        return {
            "position": self.position,
            "accuracy": self.position_accuracy,
            "active_bodies": len(self.active_bodies),
            "visible_bodies": len(observations)
        }
    
    def _extract_observations(self, environment_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract celestial observations from environment data."""
        observations = []
        
        if not environment_data or "celestial_observations" not in environment_data:
            return observations
            
        # Process each observation
        for obs in environment_data["celestial_observations"]:
            if "id" in obs and "azimuth" in obs and "elevation" in obs:
                observations.append(obs)
                
        return observations
    
    def _calculate_position(self, observations: List[Dict[str, Any]]) -> None:
        """Calculate position from celestial observations."""
        # In a real system, this would use triangulation algorithms
        # For simulation, we'll use a simplified model
        
        # Simple position update based on observations
        position_update = np.zeros(3)
        
        for obs in observations:
            # Find corresponding celestial body
            body = next((b for b in self.active_bodies if b.id == obs["id"]), None)
            if not body:
                continue
                
            # Convert spherical to cartesian coordinates (simplified)
            azimuth = np.radians(obs["azimuth"])
            elevation = np.radians(obs["elevation"])
            
            # Unit vector pointing to celestial body
            direction = np.array([
                np.cos(elevation) * np.sin(azimuth),
                np.cos(elevation) * np.cos(azimuth),
                np.sin(elevation)
            ])
            
            # Weight by brightness (brighter stars have more influence)
            weight = 1.0 / (body.magnitude + 2.0)
            position_update += direction * weight
        
        # Normalize and scale position update
        if np.any(position_update):
            position_update = position_update / np.linalg.norm(position_update) * 0.1
            
            # Apply Kalman filter
            self.position_filter = (
                (1.0 - self.config.position_filter_strength) * self.position_filter + 
                self.config.position_filter_strength * position_update
            )
            
            # Update position
            self.position["x"] += self.position_filter[0]
            self.position["y"] += self.position_filter[1]
            self.position["z"] += self.position_filter[2]
    
    def _update_position_accuracy(self, num_observations: int) -> None:
        """Update position accuracy based on number of observations."""
        # More observations = better accuracy
        base_accuracy = 500.0  # meters
        observation_factor = max(0.5, min(1.0, num_observations / self.config.max_celestial_bodies))
        
        # Update accuracy (lower is better)
        self.position_accuracy = base_accuracy / observation_factor
    
    def get_position(self) -> Dict[str, float]:
        """
        Get current position using celestial navigation.
        
        Returns:
            Position dictionary with x, y, z coordinates
        """
        if not self.active:
            return {"x": float('nan'), "y": float('nan'), "z": float('nan')}
            
        return self.position
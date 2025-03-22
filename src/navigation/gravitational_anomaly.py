"""
Gravitational Anomaly Sensor for UCAV platforms.

Provides functionality to detect and map gravitational anomalies
for enhanced navigation in GPS-denied environments.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.navigation.error_handling import safe_navigation_operation

# Configure logger
logger = logging.getLogger(__name__)


class GravityAnomalyType(Enum):
    """Types of gravitational anomalies."""
    MASS_CONCENTRATION = "mass_concentration"
    MASS_DEFICIT = "mass_deficit"
    GEOLOGICAL = "geological"
    UNKNOWN = "unknown"


@dataclass
class GravityAnomalyConfig:
    """Configuration for gravitational anomaly detection."""
    grid_size: Tuple[int, int] = (40, 40)
    resolution: float = 20.0  # meters per grid cell
    update_rate: float = 0.5  # Hz
    filter_strength: float = 0.4  # Kalman filter strength
    anomaly_threshold: float = 0.05  # mGal threshold for anomaly detection


@dataclass
class GravityAnomaly:
    """Gravitational anomaly data."""
    position: Tuple[float, float]  # x, y coordinates
    strength: float  # gravity anomaly in mGal
    gradient: float  # gravity gradient
    anomaly_type: GravityAnomalyType
    confidence: float  # detection confidence


class GravitationalAnomalySensor:
    """
    Gravitational anomaly sensor system.
    
    Maps gravitational field variations and detects anomalies for navigation.
    """
    
    def __init__(self, config: GravityAnomalyConfig = GravityAnomalyConfig()):
        """
        Initialize gravitational anomaly sensor.
        
        Args:
            config: Sensor configuration
        """
        self.config = config
        
        # Initialize gravity field grid
        self.gravity_map = np.zeros(config.grid_size)  # gravity anomaly in mGal
        self.gradient_map = np.zeros(config.grid_size + (2,))  # x, y gradients
        self.confidence_map = np.zeros(config.grid_size)
        
        # Anomaly detection
        self.anomalies: List[GravityAnomaly] = []
        
        # Reference gravity (Earth's standard gravity)
        self.reference_gravity = 9.80665  # m/s²
        
        logger.info(f"Initialized gravitational anomaly sensor with {config.grid_size} grid")
    
    @safe_navigation_operation
    def update_gravity_data(self, 
                          position: Tuple[float, float], 
                          gravity_reading: float,
                          confidence: float = 1.0) -> None:
        """
        Update gravity map with new sensor reading.
        
        Args:
            position: Current position (x, y)
            gravity_reading: Gravity measurement in m/s²
            confidence: Confidence in the reading (0-1)
        """
        # Convert position to grid coordinates
        grid_x, grid_y = self._position_to_grid(position)
        
        # Check if within grid bounds
        if not (0 <= grid_x < self.config.grid_size[0] and 0 <= grid_y < self.config.grid_size[1]):
            return
        
        # Convert to mGal anomaly (1 mGal = 10^-5 m/s²)
        anomaly_mgal = (gravity_reading - self.reference_gravity) * 100000
        
        # Update gravity map with Kalman filter
        k = self.config.filter_strength * confidence
        self.gravity_map[grid_y, grid_x] = (1 - k) * self.gravity_map[grid_y, grid_x] + k * anomaly_mgal
        
        # Update confidence map
        self.confidence_map[grid_y, grid_x] = min(1.0, self.confidence_map[grid_y, grid_x] + 0.1 * confidence)
        
        # Update gradients if we have neighboring cells
        self._update_gradients(grid_x, grid_y)
        
        # Check for anomalies
        self._detect_anomalies(position, anomaly_mgal)
    
    def _update_gradients(self, grid_x: int, grid_y: int) -> None:
        """Update gravity gradient at the specified grid position."""
        # Calculate x gradient if possible
        if 0 < grid_x < self.config.grid_size[0] - 1:
            self.gradient_map[grid_y, grid_x, 0] = (
                self.gravity_map[grid_y, grid_x + 1] - self.gravity_map[grid_y, grid_x - 1]
            ) / (2 * self.config.resolution)
        
        # Calculate y gradient if possible
        if 0 < grid_y < self.config.grid_size[1] - 1:
            self.gradient_map[grid_y, grid_x, 1] = (
                self.gravity_map[grid_y + 1, grid_x] - self.gravity_map[grid_y - 1, grid_x]
            ) / (2 * self.config.resolution)
    
    def _detect_anomalies(self, position: Tuple[float, float], anomaly_mgal: float) -> None:
        """Detect gravitational anomalies from readings."""
        # If anomaly exceeds threshold, record it
        if abs(anomaly_mgal) > self.config.anomaly_threshold:
            # Check if anomaly already exists at this position
            for anomaly in self.anomalies:
                dist = np.sqrt((position[0] - anomaly.position[0])**2 + 
                              (position[1] - anomaly.position[1])**2)
                if dist < self.config.resolution:
                    # Update existing anomaly
                    anomaly.strength = (anomaly.strength + anomaly_mgal) / 2
                    anomaly.confidence = min(1.0, anomaly.confidence + 0.1)
                    return
            
            # Determine anomaly type
            if anomaly_mgal > 0:
                anomaly_type = GravityAnomalyType.MASS_CONCENTRATION
            else:
                anomaly_type = GravityAnomalyType.MASS_DEFICIT
            
            # Create new anomaly
            grid_x, grid_y = self._position_to_grid(position)
            gradient = 0.0
            if (0 <= grid_x < self.config.grid_size[0] and 
                0 <= grid_y < self.config.grid_size[1]):
                gradient = np.linalg.norm(self.gradient_map[grid_y, grid_x])
            
            anomaly = GravityAnomaly(
                position=position,
                strength=anomaly_mgal,
                gradient=float(gradient),
                anomaly_type=anomaly_type,
                confidence=0.6
            )
            self.anomalies.append(anomaly)
            logger.debug(f"Detected new gravitational anomaly at {position}")
    
    def get_gravity_at_position(self, position: Tuple[float, float]) -> Dict[str, Any]:
        """
        Get gravity data at a specific position.
        
        Args:
            position: Position (x, y) to query
            
        Returns:
            Dictionary with gravity data
        """
        grid_x, grid_y = self._position_to_grid(position)
        
        # Check if within grid bounds
        if not (0 <= grid_x < self.config.grid_size[0] and 0 <= grid_y < self.config.grid_size[1]):
            return {
                "gravity": self.reference_gravity,
                "anomaly_mgal": 0.0,
                "gradient": [0.0, 0.0],
                "confidence": 0.0,
                "anomalies": []
            }
        
        # Get gravity data
        anomaly_mgal = self.gravity_map[grid_y, grid_x]
        gradient = self.gradient_map[grid_y, grid_x].tolist()
        confidence = self.confidence_map[grid_y, grid_x]
        
        # Find nearby anomalies
        nearby_anomalies = []
        for anomaly in self.anomalies:
            dist = np.sqrt((position[0] - anomaly.position[0])**2 + 
                          (position[1] - anomaly.position[1])**2)
            if dist < self.config.resolution * 2:
                nearby_anomalies.append({
                    "position": anomaly.position,
                    "strength": anomaly.strength,
                    "type": anomaly.anomaly_type.value,
                    "distance": dist
                })
        
        return {
            "gravity": self.reference_gravity + (anomaly_mgal / 100000),
            "anomaly_mgal": float(anomaly_mgal),
            "gradient": gradient,
            "confidence": float(confidence),
            "anomalies": nearby_anomalies
        }
    
    def get_navigation_features(self) -> List[Dict[str, Any]]:
        """
        Get gravitational features useful for navigation.
        
        Returns:
            List of gravitational features for navigation
        """
        features = []
        
        # Only include high-confidence anomalies
        for anomaly in self.anomalies:
            if anomaly.confidence > 0.7:
                features.append({
                    "position": anomaly.position,
                    "strength": anomaly.strength,
                    "type": anomaly.anomaly_type.value,
                    "confidence": anomaly.confidence,
                    "gradient": anomaly.gradient
                })
        
        return features
    
    def _position_to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world position to grid coordinates."""
        # Assuming grid is centered at origin
        grid_size_meters = (
            self.config.grid_size[0] * self.config.resolution,
            self.config.grid_size[1] * self.config.resolution
        )
        
        grid_x = int((position[0] + grid_size_meters[0]/2) / self.config.resolution)
        grid_y = int((position[1] + grid_size_meters[1]/2) / self.config.resolution)
        
        return grid_x, grid_y
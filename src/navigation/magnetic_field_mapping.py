"""
Magnetic Field Mapping for UCAV platforms.

Provides functionality to detect, map, and navigate using Earth's magnetic field
and magnetic anomalies for GPS-denied navigation.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.navigation.error_handling import safe_navigation_operation

# Configure logger
logger = logging.getLogger(__name__)


class MagneticAnomalyType(Enum):
    """Types of magnetic anomalies."""
    NATURAL = "natural"
    ARTIFICIAL = "artificial"
    GEOLOGICAL = "geological"
    UNKNOWN = "unknown"


@dataclass
class MagneticFieldConfig:
    """Configuration for magnetic field mapping."""
    grid_size: Tuple[int, int] = (50, 50)
    resolution: float = 10.0  # meters per grid cell
    update_rate: float = 1.0  # Hz
    filter_strength: float = 0.3  # Kalman filter strength
    anomaly_threshold: float = 0.2  # Threshold for anomaly detection


@dataclass
class MagneticAnomaly:
    """Magnetic anomaly data."""
    position: Tuple[float, float]  # x, y coordinates
    strength: float  # magnetic field strength in nT
    gradient: float  # field gradient
    anomaly_type: MagneticAnomalyType
    confidence: float  # detection confidence


class MagneticFieldMap:
    """
    Magnetic field mapping system.
    
    Maps the Earth's magnetic field and detects anomalies for navigation.
    """
    
    def __init__(self, config: MagneticFieldConfig = MagneticFieldConfig()):
        """
        Initialize magnetic field mapping.
        
        Args:
            config: Mapping configuration
        """
        self.config = config
        
        # Initialize magnetic field grid
        self.field_map = np.zeros(config.grid_size + (3,))  # x, y, z components
        self.field_strength_map = np.zeros(config.grid_size)
        self.confidence_map = np.zeros(config.grid_size)
        
        # Anomaly detection
        self.anomalies: List[MagneticAnomaly] = []
        
        # Reference field (Earth's background field)
        self.reference_field = np.array([20000.0, 0.0, 43000.0])  # nT (north, east, down)
        
        logger.info(f"Initialized magnetic field mapping with {config.grid_size} grid")
    
    @safe_navigation_operation
    def update_field_data(self, 
                        position: Tuple[float, float], 
                        magnetic_reading: np.ndarray,
                        confidence: float = 1.0) -> None:
        """
        Update magnetic field map with new sensor reading.
        
        Args:
            position: Current position (x, y)
            magnetic_reading: Magnetic field vector [x, y, z] in nT
            confidence: Confidence in the reading (0-1)
        """
        # Convert position to grid coordinates
        grid_x, grid_y = self._position_to_grid(position)
        
        # Check if within grid bounds
        if not (0 <= grid_x < self.config.grid_size[0] and 0 <= grid_y < self.config.grid_size[1]):
            return
        
        # Update field map with Kalman filter
        k = self.config.filter_strength * confidence
        self.field_map[grid_y, grid_x] = (1 - k) * self.field_map[grid_y, grid_x] + k * magnetic_reading
        
        # Update field strength map
        field_strength = np.linalg.norm(magnetic_reading)
        self.field_strength_map[grid_y, grid_x] = (1 - k) * self.field_strength_map[grid_y, grid_x] + k * field_strength
        
        # Update confidence map
        self.confidence_map[grid_y, grid_x] = min(1.0, self.confidence_map[grid_y, grid_x] + 0.1 * confidence)
        
        # Check for anomalies
        self._detect_anomalies(position, magnetic_reading, float(field_strength))
    
    def _detect_anomalies(self, 
                         position: Tuple[float, float],
                         magnetic_reading: np.ndarray,
                         field_strength: float) -> None:
        """Detect magnetic anomalies from readings."""
        # Calculate expected field at this position (simplified model)
        expected_field = self.reference_field
        
        # Calculate deviation from expected field
        deviation = np.linalg.norm(magnetic_reading - expected_field) / np.linalg.norm(expected_field)
        
        # If deviation exceeds threshold, record anomaly
        if deviation > self.config.anomaly_threshold:
            # Check if anomaly already exists at this position
            for anomaly in self.anomalies:
                dist = np.sqrt((position[0] - anomaly.position[0])**2 + 
                              (position[1] - anomaly.position[1])**2)
                if dist < self.config.resolution:
                    # Update existing anomaly
                    anomaly.strength = (anomaly.strength + field_strength) / 2
                    anomaly.confidence = min(1.0, anomaly.confidence + 0.1)
                    return
            
            # Create new anomaly
            anomaly = MagneticAnomaly(
                position=position,
                strength=field_strength,
                gradient=float(deviation),
                anomaly_type=MagneticAnomalyType.UNKNOWN,
                confidence=0.6
            )
            self.anomalies.append(anomaly)
            logger.debug(f"Detected new magnetic anomaly at {position}")
    
    def get_field_at_position(self, position: Tuple[float, float]) -> Dict[str, Any]:
        """
        Get magnetic field data at a specific position.
        
        Args:
            position: Position (x, y) to query
            
        Returns:
            Dictionary with field data
        """
        grid_x, grid_y = self._position_to_grid(position)
        
        # Check if within grid bounds
        if not (0 <= grid_x < self.config.grid_size[0] and 0 <= grid_y < self.config.grid_size[1]):
            return {
                "field_vector": self.reference_field.tolist(),
                "field_strength": np.linalg.norm(self.reference_field),
                "confidence": 0.0,
                "anomalies": []
            }
        
        # Get field data
        field_vector = self.field_map[grid_y, grid_x]
        field_strength = self.field_strength_map[grid_y, grid_x]
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
            "field_vector": field_vector.tolist(),
            "field_strength": float(field_strength),
            "confidence": float(confidence),
            "anomalies": nearby_anomalies
        }
    
    def get_navigation_features(self) -> List[Dict[str, Any]]:
        """
        Get magnetic features useful for navigation.
        
        Returns:
            List of magnetic features for navigation
        """
        features = []
        
        # Only include high-confidence anomalies
        for anomaly in self.anomalies:
            if anomaly.confidence > 0.7:
                features.append({
                    "position": anomaly.position,
                    "strength": anomaly.strength,
                    "type": anomaly.anomaly_type.value,
                    "confidence": anomaly.confidence
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
    
    def _grid_to_position(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world position."""
        # Assuming grid is centered at origin
        grid_size_meters = (
            self.config.grid_size[0] * self.config.resolution,
            self.config.grid_size[1] * self.config.resolution
        )
        
        x = grid_x * self.config.resolution - grid_size_meters[0]/2
        y = grid_y * self.config.resolution - grid_size_meters[1]/2
        
        return x, y
"""
Multi-Sensor Data Fusion Module

Provides capabilities for integrating data from multiple sensors to create
a unified perception of the environment.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import logging

from src.simulation.sensors.sensor_framework import SensorType

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Available fusion methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    KALMAN_FILTER = "kalman_filter"
    BAYESIAN = "bayesian"
    DEMPSTER_SHAFER = "dempster_shafer"


@dataclass
class FusionConfig:
    """Configuration for sensor fusion."""
    method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE
    temporal_window: int = 5  # Number of time steps to consider
    confidence_threshold: float = 0.3
    grid_size: Tuple[int, int] = (20, 20)  # Size of fusion grid


class SensorFusion:
    """Main class for multi-sensor data fusion."""
    
    def __init__(self, config: FusionConfig = FusionConfig()):
        """Initialize the sensor fusion system."""
        self.config = config
        self.sensor_weights = {}
        self.history = []
        self.current_grid = np.zeros(self.config.grid_size)
        self.confidence_grid = np.zeros(self.config.grid_size)
        
    def set_sensor_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for different sensor types."""
        self.sensor_weights = weights
        
    def process(self, sensor_data: Dict[str, Dict[str, Any]], 
               timestamp: float) -> Dict[str, Any]:
        """Process and fuse data from multiple sensors."""
        # Create empty grids for this fusion cycle
        detection_grid = np.zeros(self.config.grid_size)
        confidence_grid = np.zeros(self.config.grid_size)
        
        # Process each sensor's data
        for sensor_type, data in sensor_data.items():
            # Skip empty data
            if not data:
                continue
                
            # Get weight for this sensor type
            weight = self.sensor_weights.get(sensor_type, 1.0)
            
            # Extract and process data based on sensor type
            processed_grid = self._process_sensor_data(sensor_type, data)
            
            if processed_grid is not None:
                # Add to detection grid with weight
                detection_grid += weight * processed_grid
                
                # Update confidence based on sensor reliability
                confidence = data.get('detection_confidence', 0.8)
                confidence_grid += weight * (processed_grid > 0) * confidence
        
        # Normalize grids
        total_weight = sum(self.sensor_weights.values()) if self.sensor_weights else len(sensor_data)
        if total_weight > 0:
            detection_grid /= total_weight
            confidence_grid /= total_weight
        
        # Apply temporal fusion if history exists
        if self.history:
            detection_grid = self._apply_temporal_fusion(detection_grid)
        
        # Store current result in history
        self.history.append({
            'grid': detection_grid.copy(),
            'confidence': confidence_grid.copy(),
            'timestamp': timestamp
        })
        
        # Keep history limited to temporal window
        if len(self.history) > self.config.temporal_window:
            self.history = self.history[-self.config.temporal_window:]
        
        # Update current grids
        self.current_grid = detection_grid
        self.confidence_grid = confidence_grid
        
        # Extract detections from grid
        detections = self._extract_detections(detection_grid, confidence_grid)
        
        return {
            'fused_grid': detection_grid.tolist(),
            'confidence_grid': confidence_grid.tolist(),
            'detections': detections,
            'timestamp': timestamp
        }
    
    def _process_sensor_data(self, sensor_type: str, 
                           data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Process data from a specific sensor type into a grid."""
        grid = None
        
        if sensor_type == SensorType.NEUROMORPHIC_VISION.name:
            # Vision sensors provide feature maps
            if 'feature_map' in data:
                grid = np.array(data['feature_map'])
        
        elif sensor_type == SensorType.SYNTHETIC_APERTURE_RADAR.name:
            # SAR provides detection maps
            if 'detection_map' in data:
                grid = np.array(data['detection_map'])
            elif 'sar_detections' in data:
                # Convert detections to grid
                grid = np.zeros(self.config.grid_size)
                for detection in data['sar_detections']:
                    x, y = self._map_to_grid(detection.get('position', [0, 0, 0]))
                    if 0 <= x < self.config.grid_size[1] and 0 <= y < self.config.grid_size[0]:
                        grid[y, x] = detection.get('detection_confidence', 0.5)
        
        elif sensor_type == SensorType.LIDAR.name:
            # LiDAR provides point clouds
            if 'obstacle_map' in data:
                grid = np.array(data['obstacle_map'])
            elif 'point_cloud' in data:
                # Convert point cloud to grid
                grid = np.zeros(self.config.grid_size)
                for point in data['point_cloud']:
                    x, y = self._map_to_grid(point.get('position', [0, 0, 0]))
                    if 0 <= x < self.config.grid_size[1] and 0 <= y < self.config.grid_size[0]:
                        grid[y, x] = point.get('intensity', 1.0)
        
        elif sensor_type == SensorType.TERAHERTZ.name or sensor_type == SensorType.QUANTUM_RADAR.name:
            # Other sensors may provide their own maps
            for key in ['material_map', 'entanglement_map', 'detection_map']:
                if key in data:
                    grid = np.array(data[key])
                    break
        
        # Resize grid if needed
        if grid is not None and grid.shape != self.config.grid_size:
            grid = self._resize_grid(grid, self.config.grid_size)
            
        return grid
    
    def _resize_grid(self, grid: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize a grid to the target size."""
        if grid.shape == target_size:
            return grid
            
        # Simple resize by repeating or sampling
        h, w = grid.shape
        h_ratio = target_size[0] / h
        w_ratio = target_size[1] / w
        
        resized_grid = np.zeros(target_size)
        for i in range(target_size[0]):
            for j in range(target_size[1]):
                src_i = min(h-1, int(i / h_ratio))
                src_j = min(w-1, int(j / w_ratio))
                resized_grid[i, j] = grid[src_i, src_j]
        
        return resized_grid
    
    def _map_to_grid(self, position: List[float]) -> Tuple[int, int]:
        """Map a 3D position to grid coordinates."""
        # Simple mapping assuming position is in meters and grid covers 100x100m
        # with the platform at the center
        x, y = position[0], position[1]
        
        # Map to grid coordinates
        grid_x = int((x + 50) / 100 * self.config.grid_size[1])
        grid_y = int((y + 50) / 100 * self.config.grid_size[0])
        
        return grid_x, grid_y
    
    def _apply_temporal_fusion(self, current_grid: np.ndarray) -> np.ndarray:
        """Apply temporal fusion with historical data."""
        if not self.history:
            return current_grid
            
        # Simple exponential decay
        result = current_grid.copy()
        
        for i, past in enumerate(reversed(self.history)):
            # Weight decreases with age
            weight = 0.7 ** (i + 1)
            result += weight * past['grid']
            
        # Normalize
        total_weight = 1.0 + sum(0.7 ** (i + 1) for i in range(len(self.history)))
        result /= total_weight
        
        return result
    
    def _extract_detections(self, detection_grid: np.ndarray, 
                          confidence_grid: np.ndarray) -> List[Dict[str, Any]]:
        """Extract discrete detections from the fusion grid."""
        detections = []
        
        # Find local maxima in the grid
        for y in range(1, self.config.grid_size[0] - 1):
            for x in range(1, self.config.grid_size[1] - 1):
                value = detection_grid[y, x]
                confidence = confidence_grid[y, x]
                
                # Skip low values
                if value < self.config.confidence_threshold:
                    continue
                    
                # Check if local maximum
                neighborhood = detection_grid[y-1:y+2, x-1:x+2]
                if value >= np.max(neighborhood):
                    # Convert grid coordinates to world position
                    pos_x = (x / self.config.grid_size[1]) * 100 - 50
                    pos_y = (y / self.config.grid_size[0]) * 100 - 50
                    
                    detections.append({
                        'position': [pos_x, pos_y, 0.0],
                        'confidence': float(confidence),
                        'value': float(value)
                    })
        
        return detections


# Create a fusion system with default weights
def create_fusion_system() -> SensorFusion:
    """Create a sensor fusion system with default configuration."""
    fusion = SensorFusion()
    
    # Set default weights for different sensor types
    fusion.set_sensor_weights({
        SensorType.NEUROMORPHIC_VISION.name: 0.8,
        SensorType.SYNTHETIC_APERTURE_RADAR.name: 1.2,
        SensorType.LIDAR.name: 1.0,
        SensorType.TERAHERTZ.name: 0.7,
        SensorType.QUANTUM_RADAR.name: 1.1
    })
    
    return fusion
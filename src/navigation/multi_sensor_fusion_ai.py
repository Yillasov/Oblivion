"""
Multi-Sensor Fusion AI for UCAV platforms.

Provides advanced fusion capabilities using neuromorphic computing
to integrate data from multiple sensor types.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass

from src.core.fusion.sensor_fusion import SensorFusion, FusionConfig, FusionMethod
from src.core.neuromorphic.sensor_algorithms import MultimodalFusionAlgorithm
from src.simulation.sensors.sensor_framework import SensorType

# Configure logger
logger = logging.getLogger(__name__)


class FusionAIMode(Enum):
    """Operating modes for the fusion AI."""
    STANDARD = "standard"
    NEUROMORPHIC = "neuromorphic"
    HYBRID = "hybrid"


@dataclass
class FusionAIConfig:
    """Configuration for the fusion AI."""
    mode: FusionAIMode = FusionAIMode.HYBRID
    confidence_threshold: float = 0.6
    grid_size: Tuple[int, int] = (30, 30)
    temporal_window: int = 5


class MultiSensorFusionAI:
    """
    Multi-Sensor Fusion AI system.
    
    Integrates data from multiple sensors using neuromorphic computing.
    """
    
    def __init__(self, config: FusionAIConfig = FusionAIConfig()):
        """Initialize the multi-sensor fusion AI."""
        self.config = config
        
        # Create standard fusion system
        fusion_config = FusionConfig(
            method=FusionMethod.KALMAN_FILTER,
            temporal_window=config.temporal_window,
            confidence_threshold=config.confidence_threshold,
            grid_size=config.grid_size
        )
        self.standard_fusion = SensorFusion(fusion_config)
        
        # Create neuromorphic fusion system
        self.neuromorphic_fusion = MultimodalFusionAlgorithm()
        
        # Set default sensor weights
        self._set_default_weights()
        
        # Fusion results
        self.fusion_result = {}
        self.confidence_map = np.zeros(config.grid_size)
        
        logger.info(f"Initialized Multi-Sensor Fusion AI in {config.mode.value} mode")
    
    def _set_default_weights(self) -> None:
        """Set default weights for different sensor types."""
        weights = {
            SensorType.NEUROMORPHIC_VISION.name: 1.0,
            SensorType.SYNTHETIC_APERTURE_RADAR.name: 1.2,
            SensorType.LIDAR.name: 0.9,
            SensorType.TERAHERTZ.name: 0.8,
            SensorType.QUANTUM_RADAR.name: 1.5,
            SensorType.INFRARED.name: 0.7
        }
        
        self.standard_fusion.set_sensor_weights(weights)
        self.neuromorphic_fusion.set_sensor_weights(weights)
    
    def process(self, sensor_data: Dict[str, Dict[str, Any]], timestamp: float) -> Dict[str, Any]:
        """
        Process and fuse data from multiple sensors.
        
        Args:
            sensor_data: Dictionary of sensor data by sensor type
            timestamp: Current timestamp
            
        Returns:
            Dictionary with fusion results
        """
        if not sensor_data:
            return {"error": "No sensor data provided"}
        
        # Process based on mode
        if self.config.mode == FusionAIMode.STANDARD:
            result = self.standard_fusion.process(sensor_data, timestamp)
        
        elif self.config.mode == FusionAIMode.NEUROMORPHIC:
            neuro_result = self.neuromorphic_fusion.process(sensor_data)
            result = {
                "fused_grid": neuro_result.get("fused_map", []),
                "confidence_grid": neuro_result.get("confidence_map", []),
                "timestamp": timestamp
            }
        
        else:  # HYBRID mode
            # Process with both systems
            std_result = self.standard_fusion.process(sensor_data, timestamp)
            neuro_result = self.neuromorphic_fusion.process(sensor_data)
            
            # Combine results (simple weighted average)
            std_grid = np.array(std_result.get("fused_grid", np.zeros(self.config.grid_size)))
            neuro_grid = np.array(neuro_result.get("fused_map", np.zeros(self.config.grid_size)))
            
            # Resize neuromorphic grid if needed
            if neuro_grid.shape != std_grid.shape:
                neuro_grid = self._resize_grid(neuro_grid, (std_grid.shape[0], std_grid.shape[1]))
            
            # Weighted fusion
            fused_grid = 0.4 * std_grid + 0.6 * neuro_grid
            
            # Combine confidence
            std_conf = np.array(std_result.get("confidence_grid", np.zeros(self.config.grid_size)))
            neuro_conf = np.array(neuro_result.get("confidence_map", np.zeros(self.config.grid_size)))
            
            if neuro_conf.shape != std_conf.shape:
                neuro_conf = self._resize_grid(neuro_conf, (std_conf.shape[0], std_conf.shape[1]))
            
            confidence_grid = np.maximum(std_conf, neuro_conf)
            
            result = {
                "fused_grid": fused_grid.tolist(),
                "confidence_grid": confidence_grid.tolist(),
                "detections": std_result.get("detections", []),
                "timestamp": timestamp
            }
        
        # Store result
        self.fusion_result = result
        self.confidence_map = np.array(result.get("confidence_grid", np.zeros(self.config.grid_size)))
        
        return result
    
    def _resize_grid(self, grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize a grid to the target shape."""
        if grid.shape == target_shape:
            return grid
            
        h, w = grid.shape
        th, tw = target_shape
        
        h_ratio = th / h
        w_ratio = tw / w
        
        resized = np.zeros(target_shape)
        for i in range(th):
            for j in range(tw):
                src_i = min(h-1, int(i / h_ratio))
                src_j = min(w-1, int(j / w_ratio))
                resized[i, j] = grid[src_i, src_j]
        
        return resized
    
    def get_high_confidence_regions(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get regions with high confidence."""
        if threshold is None:
            threshold = self.config.confidence_threshold
            
        regions = []
        confidence = self.confidence_map
        
        # Find connected regions above threshold
        visited = np.zeros_like(confidence, dtype=bool)
        
        for i in range(confidence.shape[0]):
            for j in range(confidence.shape[1]):
                if confidence[i, j] >= threshold and not visited[i, j]:
                    # Found a new region
                    region_points = []
                    self._flood_fill(confidence, visited, i, j, threshold, region_points)
                    
                    if region_points:
                        # Calculate region properties
                        points = np.array(region_points)
                        center = points.mean(axis=0)
                        size = len(points)
                        avg_confidence = confidence[tuple(zip(*region_points))].mean()
                        
                        regions.append({
                            "center": center.tolist(),
                            "size": size,
                            "confidence": float(avg_confidence),
                            "points": region_points
                        })
        
        return regions
    
    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, 
                   i: int, j: int, threshold: float, 
                   points: List[Tuple[int, int]]) -> None:
        """Flood fill algorithm to find connected regions."""
        if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1]:
            return
            
        if visited[i, j] or grid[i, j] < threshold:
            return
            
        # Mark as visited and add to region
        visited[i, j] = True
        points.append((i, j))
        
        # Check neighbors
        self._flood_fill(grid, visited, i+1, j, threshold, points)
        self._flood_fill(grid, visited, i-1, j, threshold, points)
        self._flood_fill(grid, visited, i, j+1, threshold, points)
        self._flood_fill(grid, visited, i, j-1, threshold, points)
"""
Distributed Aperture System implementation for 360-degree situational awareness.

This module provides a distributed aperture system that uses multiple sensors
positioned around the platform to provide complete spherical coverage.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass, field

from src.simulation.sensors.advanced_sensors import DistributedApertureSensor
from src.simulation.sensors.sensor_framework import SensorConfig, SensorType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


@dataclass
class SensorLocation:
    """Location and orientation of a single sensor in the distributed array."""
    position: List[float]  # [x, y, z] relative to platform center
    orientation: List[float]  # [pitch, yaw, roll] in degrees
    field_of_view: float  # degrees


@dataclass
class DASConfig:
    """Configuration for distributed aperture system."""
    sensor_count: int = 6
    sensor_locations: List[SensorLocation] = field(default_factory=list)
    fusion_method: str = "weighted"  # weighted, bayesian, neural
    overlap_percentage: float = 15.0  # percentage of overlap between sensors
    
    def __post_init__(self):
        if not self.sensor_locations:
            # Default to 6 sensors in hexagonal arrangement
            self.sensor_locations = [
                SensorLocation([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 70.0),    # Front
                SensorLocation([-1.0, 0.0, 0.0], [0.0, 180.0, 0.0], 70.0),  # Rear
                SensorLocation([0.0, 1.0, 0.0], [0.0, 90.0, 0.0], 70.0),    # Right
                SensorLocation([0.0, -1.0, 0.0], [0.0, 270.0, 0.0], 70.0),  # Left
                SensorLocation([0.0, 0.0, 1.0], [90.0, 0.0, 0.0], 70.0),    # Top
                SensorLocation([0.0, 0.0, -1.0], [-90.0, 0.0, 0.0], 70.0)   # Bottom
            ]


class DistributedApertureImplementation(DistributedApertureSensor):
    """
    Distributed Aperture System implementation.
    
    This system uses multiple sensors positioned around the platform to provide
    complete spherical coverage and situational awareness.
    """
    
    def __init__(self, 
                config: SensorConfig, 
                das_config: DASConfig = DASConfig(),
                neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """Initialize distributed aperture system."""
        super().__init__(config, neuromorphic_system)
        self.das_config = das_config
        self.num_sensors = das_config.sensor_count
        self.sensor_locations = das_config.sensor_locations
        self.fusion_method = das_config.fusion_method
        
        # DAS specific data
        self.data.update({
            'sensor_detections': [[] for _ in range(self.num_sensors)],
            'fused_detections': [],
            'coverage_map': np.zeros((180, 360)),  # Elevation x Azimuth
            'blind_spots': []
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update distributed aperture sensor data."""
        # Get targets from environment
        targets = environment.get('targets', [])
        
        # Clear previous detections
        for i in range(self.num_sensors):
            self.data['sensor_detections'][i] = []
        self.data['fused_detections'] = []
        
        # Platform position and orientation
        position = platform_state.get('position', np.zeros(3))
        orientation = platform_state.get('orientation', np.zeros(3))
        
        # Process each target for each sensor
        for target in targets:
            target_pos = target.get('position', np.zeros(3))
            target_id = target.get('id', 0)
            
            # Calculate relative position
            rel_pos = target_pos - position
            distance = np.linalg.norm(rel_pos)
            
            # Check if in range
            if distance < self.config.min_range or distance > self.config.max_range:
                continue
            
            # Process each sensor
            for i, sensor_loc in enumerate(self.sensor_locations):
                # Calculate target position in sensor's local frame
                sensor_rel_pos = self._transform_to_sensor_frame(rel_pos, sensor_loc, orientation)
                
                # Calculate angles in sensor frame
                sensor_azimuth = np.arctan2(sensor_rel_pos[1], sensor_rel_pos[0])
                sensor_elevation = np.arcsin(sensor_rel_pos[2] / max(0.1, np.linalg.norm(sensor_rel_pos)))
                
                # Convert to degrees
                sensor_azimuth_deg = np.degrees(sensor_azimuth)
                sensor_elevation_deg = np.degrees(sensor_elevation)
                
                # Check if target is in sensor's field of view
                if abs(sensor_azimuth_deg) > sensor_loc.field_of_view/2 or abs(sensor_elevation_deg) > sensor_loc.field_of_view/2:
                    continue
                
                # Calculate detection probability
                detection_prob = self._calculate_detection_probability(
                    target, float(distance), sensor_azimuth_deg, sensor_elevation_deg
                )
                
                # Random detection based on probability
                if self.rng.random() < detection_prob:
                    # Calculate global angles
                    azimuth = np.arctan2(rel_pos[1], rel_pos[0])
                    elevation = np.arcsin(rel_pos[2] / max(0.1, distance))
                    
                    # Convert to degrees
                    azimuth_deg = np.degrees(azimuth)
                    elevation_deg = np.degrees(elevation)
                    
                    self.data['sensor_detections'][i].append({
                        'id': target_id,
                        'distance': float(distance),
                        'azimuth': float(azimuth_deg),
                        'elevation': float(elevation_deg),
                        'sensor_azimuth': float(sensor_azimuth_deg),
                        'sensor_elevation': float(sensor_elevation_deg),
                        'detection_confidence': float(detection_prob),
                        'sensor_id': i
                    })
        
        # Update coverage map
        self._update_coverage_map()
        
        # Fuse detections from all sensors
        fused_detections = self._fuse_detections()
        self.data['fused_detections'] = fused_detections
    
    def _transform_to_sensor_frame(self, rel_pos: np.ndarray, 
                                 sensor_loc: SensorLocation,
                                 platform_orientation: np.ndarray) -> np.ndarray:
        """Transform a position from platform frame to sensor local frame."""
        # Simple implementation - in a real system this would use proper rotation matrices
        sensor_orientation = np.array(sensor_loc.orientation) + platform_orientation
        
        # Convert to radians
        pitch = np.radians(sensor_orientation[0])
        yaw = np.radians(sensor_orientation[1])
        roll = np.radians(sensor_orientation[2])
        
        # Apply rotations (simplified)
        # This is a basic approximation - a real system would use proper 3D transformations
        rotated_pos = rel_pos.copy()
        
        # Yaw rotation (around z-axis)
        rotated_pos[0] = rel_pos[0] * np.cos(yaw) + rel_pos[1] * np.sin(yaw)
        rotated_pos[1] = -rel_pos[0] * np.sin(yaw) + rel_pos[1] * np.cos(yaw)
        
        # Pitch rotation (around y-axis)
        rotated_pos_x = rotated_pos[0] * np.cos(pitch) - rotated_pos[2] * np.sin(pitch)
        rotated_pos_z = rotated_pos[0] * np.sin(pitch) + rotated_pos[2] * np.cos(pitch)
        rotated_pos[0] = rotated_pos_x
        rotated_pos[2] = rotated_pos_z
        
        return rotated_pos
    
    def _calculate_detection_probability(self, target: Dict[str, Any], 
                                       distance: float,
                                       azimuth: float,
                                       elevation: float) -> float:
        """Calculate detection probability based on target and sensor parameters."""
        # Base detection probability
        base_prob = self.config.accuracy * (1.0 - distance / self.config.max_range)
        
        # Angle factor - detection probability decreases toward edge of FOV
        angle_factor = 1.0 - (abs(azimuth) + abs(elevation)) / 180.0
        
        # Target signature factor
        signature = target.get('visual_signature', 0.7)
        signature_factor = min(1.0, signature * 1.5)
        
        # Final probability
        detection_prob = base_prob * angle_factor * signature_factor
        
        return min(0.95, max(0.1, detection_prob))
    
    def _update_coverage_map(self) -> None:
        """Update the coverage map based on sensor positions and orientations."""
        # Reset coverage map
        coverage_map = np.zeros((180, 360))
        
        # For each sensor, mark its coverage area
        for sensor_loc in self.sensor_locations:
            # Convert sensor orientation to indices
            center_el = int(90 + sensor_loc.orientation[0])
            center_az = int(180 + sensor_loc.orientation[1]) % 360
            
            # Mark coverage area
            fov_half = int(sensor_loc.field_of_view / 2)
            for el in range(max(0, center_el - fov_half), min(180, center_el + fov_half)):
                for az in range(360):
                    # Calculate angular distance from center
                    az_diff = min((az - center_az) % 360, (center_az - az) % 360)
                    if az_diff <= fov_half:
                        coverage_map[el, az] = 1
        
        self.data['coverage_map'] = coverage_map
        
        # Find blind spots
        blind_spots = []
        for el in range(0, 180, 10):
            for az in range(0, 360, 10):
                if coverage_map[el, az] == 0:
                    blind_spots.append((el - 90, az - 180))
        
        self.data['blind_spots'] = blind_spots
    
    def _fuse_detections(self) -> List[Dict[str, Any]]:
        """Fuse detections from multiple sensors."""
        # Group detections by target ID
        targets = {}
        
        for sensor_id, detections in enumerate(self.data['sensor_detections']):
            for detection in detections:
                target_id = detection['id']
                if target_id not in targets:
                    targets[target_id] = {
                        'sensors': [],
                        'detections': []
                    }
                
                targets[target_id]['sensors'].append(sensor_id)
                targets[target_id]['detections'].append(detection)
        
        # Create fused detections
        fused_detections = []
        
        for target_id, target_data in targets.items():
            if not target_data['detections']:
                continue
                
            # Average position across sensors
            avg_distance = np.mean([d['distance'] for d in target_data['detections']])
            avg_azimuth = np.mean([d['azimuth'] for d in target_data['detections']])
            avg_elevation = np.mean([d['elevation'] for d in target_data['detections']])
            
            # Confidence increases with number of sensors that detected the target
            sensor_count = len(set(target_data['sensors']))
            confidence_boost = min(0.3, 0.05 * sensor_count)
            
            # Average confidence across sensors with boost for multi-sensor detection
            avg_confidence = np.mean([d['detection_confidence'] for d in target_data['detections']])
            fused_confidence = min(0.99, avg_confidence + confidence_boost)
            
            # Create fused detection
            fused_detections.append({
                'id': target_id,
                'distance': float(avg_distance),
                'azimuth': float(avg_azimuth),
                'elevation': float(avg_elevation),
                'detection_confidence': float(fused_confidence),
                'sensor_count': sensor_count,
                'sensors': list(set(target_data['sensors']))
            })
        
        return fused_detections
    
    def get_coverage_analysis(self) -> Dict[str, Any]:
        """Get coverage analysis results."""
        coverage_map = self.data.get('coverage_map', np.zeros((180, 360)))
        blind_spots = self.data.get('blind_spots', [])
        
        # Calculate coverage percentage
        coverage_percentage = np.sum(coverage_map) / (180 * 360) * 100
        
        return {
            'coverage_percentage': float(coverage_percentage),
            'blind_spot_count': len(blind_spots),
            'sensor_count': self.num_sensors,
            'fusion_method': self.fusion_method,
            'detection_count': len(self.data.get('fused_detections', []))
        }
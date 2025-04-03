#!/usr/bin/env python3
"""
Synthetic Aperture Radar implementation for high-resolution imaging.

This module provides a SAR implementation that uses platform motion to create
a synthetic aperture for high-resolution imaging through clouds, smoke, and foliage.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from src.simulation.sensors.advanced_sensors import SyntheticApertureRadarSensor
from src.simulation.sensors.sensor_framework import SensorConfig, SensorType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


class SARMode(Enum):
    """Operating modes for synthetic aperture radar."""
    SPOTLIGHT = "spotlight"  # High resolution, small area
    STRIPMAP = "stripmap"    # Medium resolution, continuous strip
    SCANSAR = "scansar"      # Lower resolution, wide area
    TOPSAR = "topsar"        # Terrain observation


class Polarization(Enum):
    """Polarization modes for SAR."""
    HH = "hh"  # Horizontal transmit, horizontal receive
    VV = "vv"  # Vertical transmit, vertical receive
    HV = "hv"  # Horizontal transmit, vertical receive
    VH = "vh"  # Vertical transmit, horizontal receive


@dataclass
class SARConfig:
    """Configuration for synthetic aperture radar."""
    mode: SARMode = SARMode.SPOTLIGHT
    polarization: Polarization = Polarization.HH
    frequency: float = 10.0  # GHz (X-band)
    bandwidth: float = 0.3   # GHz
    resolution: Tuple[float, float] = (0.3, 0.3)  # Range, azimuth resolution in meters
    swath_width: float = 5000.0  # meters
    pulse_length: float = 10.0e-6  # seconds
    pulse_repetition_frequency: float = 1500.0  # Hz
    integration_time: float = 2.0  # seconds


class SyntheticApertureRadarImplementation(SyntheticApertureRadarSensor):
    """
    Synthetic Aperture Radar implementation.
    
    This sensor uses platform motion to create a synthetic aperture for
    high-resolution imaging through clouds, smoke, and foliage.
    """
    
    def __init__(self, 
                config: SensorConfig, 
                sar_config: SARConfig = SARConfig(),
                neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """Initialize synthetic aperture radar implementation."""
        super().__init__(config, neuromorphic_system)
        self.sar_config = sar_config
        self.resolution_mode = sar_config.mode.value
        self.polarization = sar_config.polarization.value
        self.frequency = sar_config.frequency
        self.bandwidth = sar_config.bandwidth
        
        # Track platform positions for synthetic aperture
        self.platform_positions = []
        self.last_position_time = 0.0
        self.position_interval = 1.0 / sar_config.pulse_repetition_frequency
        
        # SAR specific data
        self.data.update({
            'sar_detections': [],
            'image_resolution': sar_config.resolution,
            'current_swath': [],
            'penetration_detections': []
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update synthetic aperture radar data."""
        # Get current time
        current_time = environment.get('time', 0.0)
        
        # Record platform position for synthetic aperture
        if current_time - self.last_position_time >= self.position_interval:
            self.platform_positions.append({
                'position': platform_state.get('position', np.zeros(3)).copy(),
                'orientation': platform_state.get('orientation', np.zeros(3)).copy(),
                'time': current_time
            })
            self.last_position_time = current_time
            
            # Limit history based on integration time
            max_positions = int(self.sar_config.integration_time / self.position_interval)
            if len(self.platform_positions) > max_positions:
                self.platform_positions = self.platform_positions[-max_positions:]
        
        # Get targets from environment
        targets = environment.get('targets', [])
        obstacles = environment.get('obstacles', [])
        terrain = environment.get('terrain', {})
        
        # Platform position
        position = platform_state.get('position', np.zeros(3))
        velocity = platform_state.get('velocity', np.zeros(3))
        
        # Need sufficient platform movement for SAR processing
        if len(self.platform_positions) < 3:
            return
            
        # Calculate synthetic aperture length
        aperture_positions = np.array([p['position'] for p in self.platform_positions])
        aperture_length = np.linalg.norm(aperture_positions[-1] - aperture_positions[0])
        
        # Skip if aperture is too small
        if aperture_length < 10.0:  # Minimum aperture length in meters
            return
            
        # Clear previous detections
        sar_detections = []
        penetration_detections = []
        
        # Process each target
        for target in targets:
            target_pos = target.get('position', np.zeros(3))
            target_id = target.get('id', 0)
            
            # Calculate relative position
            rel_pos = target_pos - position
            distance = np.linalg.norm(rel_pos)
            
            # Check if in range
            if distance < self.config.min_range or distance > self.config.max_range:
                continue
                
            # Check if in swath width (perpendicular to flight path)
            # Simplified: just check if within a certain distance perpendicular to velocity
            if np.linalg.norm(velocity) > 1.0:
                velocity_unit = velocity / np.linalg.norm(velocity)
                perpendicular_distance = np.linalg.norm(rel_pos - np.dot(rel_pos, velocity_unit) * velocity_unit)
                if perpendicular_distance > self.sar_config.swath_width / 2:
                    continue
            
            # Calculate angles
            azimuth = np.arctan2(rel_pos[1], rel_pos[0])
            elevation = np.arcsin(rel_pos[2] / max(0.1, distance))
            
            # Convert to degrees
            azimuth_deg = np.degrees(azimuth)
            elevation_deg = np.degrees(elevation)
            
            # Check if target is behind an obstacle (SAR can penetrate some materials)
            is_obscured = False
            penetration_material = None
            
            for obstacle in obstacles:
                obstacle_pos = obstacle.get('position', np.zeros(3))
                obstacle_size = obstacle.get('size', np.zeros(3))
                obstacle_material = obstacle.get('material', 'unknown')
                
                # Simple check if obstacle is between sensor and target
                obstacle_distance = np.linalg.norm(obstacle_pos - position)
                
                if obstacle_distance < distance:
                    # Check if obstacle is in line of sight
                    obstacle_direction = (obstacle_pos - position) / max(0.1, obstacle_distance)
                    target_direction = rel_pos / max(0.1, distance)
                    
                    # Simple dot product to check alignment
                    alignment = np.dot(obstacle_direction, target_direction)
                    
                    if alignment > 0.95:  # Roughly aligned
                        is_obscured = True
                        penetration_material = obstacle_material
                        break
            
            # Calculate detection probability based on SAR parameters
            detection_prob = self._calculate_sar_detection_probability(
                float(distance), is_obscured, penetration_material, float(aperture_length)
            )
            
            # Random detection based on probability
            if self.rng.random() < detection_prob:
                # Calculate resolution based on mode and aperture
                range_res, azimuth_res = self._calculate_resolution(float(distance), float(aperture_length))
                
                # Add position noise based on resolution
                position_noise = np.random.normal(0, range_res / 2, 3)
                
                detection = {
                    'id': target_id,
                    'distance': float(distance),
                    'azimuth': float(azimuth_deg),
                    'elevation': float(elevation_deg),
                    'detection_confidence': float(detection_prob),
                    'range_resolution': float(range_res),
                    'azimuth_resolution': float(azimuth_res),
                    'position': (target_pos + position_noise).tolist()
                }
                
                sar_detections.append(detection)
                
                if is_obscured:
                    penetration_detections.append({
                        'id': target_id,
                        'distance': float(distance),
                        'material_penetrated': penetration_material,
                        'detection_confidence': float(detection_prob)
                    })
        
        # Update sensor data
        self.data['sar_detections'] = sar_detections
        self.data['penetration_detections'] = penetration_detections
        
        # Generate current swath data (simplified)
        self._generate_swath_data(platform_state, environment)
    
    def _calculate_sar_detection_probability(self, distance: float, is_obscured: bool,
                                          material: Optional[str], aperture_length: float) -> float:
        """Calculate detection probability based on SAR parameters."""
        # Base detection probability based on distance
        base_prob = self.config.accuracy * (1.0 - distance / self.config.max_range)
        
        # Adjust for aperture length - longer aperture improves detection
        aperture_factor = min(1.0, aperture_length / 100.0)
        base_prob *= (0.7 + 0.3 * aperture_factor)
        
        # Adjust for obscuration
        if is_obscured:
            # SAR can penetrate some materials
            if material in ['foliage', 'light_vegetation', 'clouds', 'smoke', 'fabric']:
                # Good penetration
                base_prob *= 0.9
            elif material in ['wood', 'plastic', 'drywall']:
                # Moderate penetration
                base_prob *= 0.6
            elif material in ['concrete', 'stone']:
                # Poor penetration
                base_prob *= 0.3
            elif material in ['metal', 'water']:
                # Very poor penetration
                base_prob *= 0.1
            else:
                # Unknown material
                base_prob *= 0.5
        
        # Adjust for SAR mode
        if self.sar_config.mode == SARMode.SPOTLIGHT:
            # Spotlight mode has best detection for small areas
            base_prob *= 1.2
        elif self.sar_config.mode == SARMode.STRIPMAP:
            # Standard detection
            pass
        elif self.sar_config.mode == SARMode.SCANSAR:
            # Wider area, lower resolution
            base_prob *= 0.8
        
        return min(0.99, base_prob)
    
    def _calculate_resolution(self, distance: float, aperture_length: float) -> Tuple[float, float]:
        """Calculate range and azimuth resolution based on SAR parameters."""
        # Range resolution depends on bandwidth
        range_resolution = 3e8 / (2 * self.bandwidth * 1e9)
        
        # Azimuth resolution depends on aperture length and mode
        if self.sar_config.mode == SARMode.SPOTLIGHT:
            # Best resolution in spotlight mode
            azimuth_resolution = 0.5 * 3e8 / (self.frequency * 1e9 * aperture_length)
        elif self.sar_config.mode == SARMode.STRIPMAP:
            # Limited by antenna length
            azimuth_resolution = self.sar_config.resolution[1]
        else:
            # ScanSAR has worse resolution
            azimuth_resolution = self.sar_config.resolution[1] * 2
        
        return range_resolution, azimuth_resolution
    
    def _generate_swath_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Generate simplified swath data for visualization."""
        # This would normally generate a 2D image of the ground
        # For simulation purposes, we'll just create a simplified representation
        
        position = platform_state.get('position', np.zeros(3))
        velocity = platform_state.get('velocity', np.zeros(3))
        
        if np.linalg.norm(velocity) < 1.0:
            return
            
        # Create a simple grid representing the swath
        swath_width = self.sar_config.swath_width
        swath_length = 1000.0  # meters
        
        # Number of grid points
        width_points = 20
        length_points = 50
        
        # Create grid
        swath_data = []
        
        # Flight direction
        flight_dir = velocity / np.linalg.norm(velocity)
        
        # Perpendicular direction (simplified, assumes level flight)
        perp_dir = np.array([-flight_dir[1], flight_dir[0], 0])
        
        # Generate grid points
        for i in range(length_points):
            for j in range(width_points):
                # Position relative to aircraft
                rel_x = (i - length_points/2) * (swath_length / length_points)
                rel_y = (j - width_points/2) * (swath_width / width_points)
                
                # World position
                world_pos = position + rel_x * flight_dir + rel_y * perp_dir
                
                # Get height from terrain if available
                terrain_height = environment.get('terrain_height', {}).get('height', 0.0)
                world_pos[2] = terrain_height
                
                # Add to swath data
                swath_data.append({
                    'position': world_pos.tolist(),
                    'intensity': 0.5 + 0.5 * np.sin(i/5) * np.cos(j/3)  # Dummy intensity pattern
                })
        
        self.data['current_swath'] = swath_data
    
    def get_sar_image_metadata(self) -> Dict[str, Any]:
        """Get metadata about the SAR image."""
        return {
            'mode': self.sar_config.mode.value,
            'polarization': self.sar_config.polarization.value,
            'resolution': self.data['image_resolution'],
            'aperture_length': self._calculate_aperture_length(),
            'detection_count': len(self.data.get('sar_detections', [])),
            'penetration_count': len(self.data.get('penetration_detections', [])),
            'swath_width': self.sar_config.swath_width
        }
    
    def _calculate_aperture_length(self) -> float:
        """Calculate the current synthetic aperture length."""
        if len(self.platform_positions) < 2:
            return 0.0
            
        positions = np.array([p['position'] for p in self.platform_positions])
        return float(np.linalg.norm(positions[-1] - positions[0]))
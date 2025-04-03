#!/usr/bin/env python3
"""
Neuromorphic Vision Sensor implementation for event-based visual processing.

This module provides a neuromorphic vision sensor that mimics biological vision
with sparse, event-driven processing for high temporal resolution and efficiency.
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

from src.simulation.sensors.advanced_sensors import NeuromorphicVisionSensor
from src.simulation.sensors.sensor_framework import SensorConfig, SensorType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


@dataclass
class NeuromorphicVisionConfig:
    """Configuration for neuromorphic vision sensor."""
    resolution: Tuple[int, int] = (640, 480)  # width, height
    temporal_resolution: float = 0.001  # seconds (1ms)
    event_threshold: float = 0.15  # Brightness change threshold to trigger event
    bias_settings: Dict[str, float] = field(default_factory=lambda: {
        "diff_on": 0.3,      # ON event threshold
        "diff_off": 0.3,     # OFF event threshold
        "diff_amp": 2.0,     # Amplification factor
        "refractory": 0.001  # Refractory period in seconds
    })


class NeuromorphicVisionImplementation(NeuromorphicVisionSensor):
    """
    Neuromorphic Vision Sensor implementation.
    
    This sensor mimics biological vision with event-based processing,
    only registering changes in the visual field rather than full frames.
    """
    
    def __init__(self, 
                config: SensorConfig, 
                nvs_config: NeuromorphicVisionConfig = NeuromorphicVisionConfig(),
                neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """Initialize neuromorphic vision sensor."""
        super().__init__(config, neuromorphic_system)
        self.nvs_config = nvs_config
        self.resolution = nvs_config.resolution
        self.temporal_resolution = nvs_config.temporal_resolution
        self.event_threshold = nvs_config.event_threshold
        self.bias_settings = nvs_config.bias_settings
        
        # Previous frame for change detection
        self.previous_frame = np.zeros(self.resolution)
        self.last_update_time = 0.0
        
        # NVS specific data
        self.data.update({
            'events': [],
            'event_count': 0,
            'on_events': 0,
            'off_events': 0,
            'event_density': 0.0,
            'last_event_timestamp': 0.0
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update neuromorphic vision sensor data."""
        # Get current time
        current_time = environment.get('time', 0.0)
        
        # Skip if not enough time has passed
        if current_time - self.last_update_time < self.temporal_resolution:
            return
            
        # Get visual scene from environment
        scene = environment.get('visual_scene', None)
        if scene is None:
            # Generate simple scene if none provided
            scene = self._generate_simple_scene(environment)
        
        # Resize scene to match sensor resolution if needed
        if scene.shape[:2] != self.resolution[::-1]:  # Note: resolution is (width, height)
            scene = self._resize_scene(scene)
        
        # Convert to grayscale if color
        if len(scene.shape) > 2:
            scene = np.mean(scene, axis=2)
        
        # Normalize scene
        scene = scene / np.max(scene) if np.max(scene) > 0 else scene
        
        # Generate events by comparing with previous frame
        events = self._generate_events(scene, current_time)
        
        # Update previous frame
        self.previous_frame = scene.copy()
        self.last_update_time = current_time
        
        # Update sensor data
        self.data['events'] = events
        self.data['event_count'] = len(events)
        self.data['on_events'] = sum(1 for e in events if e['polarity'] > 0)
        self.data['off_events'] = sum(1 for e in events if e['polarity'] < 0)
        self.data['event_density'] = len(events) / (self.resolution[0] * self.resolution[1])
        self.data['last_event_timestamp'] = current_time
        
        # Process events neuromorphically if enabled
        if self.processing_mode == "neuromorphic" and self.neuromorphic_system:
            processed_data = self.process_data_neuromorphically({
                'events': events,
                'timestamp': current_time
            })
            self.data.update(processed_data)
    
    def _generate_simple_scene(self, environment: Dict[str, Any]) -> np.ndarray:
        """Generate a simple visual scene based on environment data."""
        # Create empty scene
        scene = np.zeros(self.resolution[::-1])  # (height, width)
        
        # Add targets as bright spots
        targets = environment.get('targets', [])
        for target in targets:
            # Get target position in sensor field of view
            target_pos = target.get('position', np.zeros(3))
            platform_pos = environment.get('platform_position', np.zeros(3))
            platform_orientation = environment.get('platform_orientation', np.zeros(3))
            
            # Calculate relative position
            rel_pos = target_pos - platform_pos
            
            # Skip if behind the sensor
            if rel_pos[0] < 0:
                continue
                
            # Simple projection to 2D
            distance = np.linalg.norm(rel_pos)
            if distance < 0.1:
                continue
                
            # Calculate angles
            azimuth = np.arctan2(rel_pos[1], rel_pos[0])
            elevation = np.arcsin(rel_pos[2] / distance)
            
            # Convert to image coordinates
            x = int((azimuth / np.pi + 0.5) * self.resolution[0])
            y = int((elevation / (np.pi/2) + 0.5) * self.resolution[1])
            
            # Check if in bounds
            if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1]:
                # Add bright spot with intensity based on distance
                intensity = min(1.0, 10.0 / distance)
                radius = max(1, int(5.0 / distance))
                
                # Draw circle
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        if dx*dx + dy*dy <= radius*radius:
                            px, py = x + dx, y + dy
                            if 0 <= px < self.resolution[0] and 0 <= py < self.resolution[1]:
                                scene[py, px] = intensity
        
        return scene
    
    def _resize_scene(self, scene: np.ndarray) -> np.ndarray:
        """Resize scene to match sensor resolution."""
        # Simple nearest-neighbor resize
        h, w = scene.shape[:2]
        target_h, target_w = self.resolution[::-1]
        
        # Calculate scaling factors
        scale_x = w / target_w
        scale_y = h / target_h
        
        # Create new array
        resized = np.zeros(self.resolution[::-1])
        
        # Simple resize
        for y in range(target_h):
            for x in range(target_w):
                src_x = min(w-1, int(x * scale_x))
                src_y = min(h-1, int(y * scale_y))
                if len(scene.shape) > 2:
                    resized[y, x] = np.mean(scene[src_y, src_x])
                else:
                    resized[y, x] = scene[src_y, src_x]
        
        return resized
    
    def _generate_events(self, current_frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """Generate events by comparing current frame with previous frame."""
        events = []
        
        # Calculate difference
        diff = current_frame - self.previous_frame
        
        # Apply thresholds
        on_events = diff > self.event_threshold
        off_events = diff < -self.event_threshold
        
        # Generate ON events
        y_coords, x_coords = np.where(on_events)
        for y, x in zip(y_coords, x_coords):
            events.append({
                'x': int(x),
                'y': int(y),
                'timestamp': timestamp,
                'polarity': 1,
                'intensity': float(diff[y, x])
            })
        
        # Generate OFF events
        y_coords, x_coords = np.where(off_events)
        for y, x in zip(y_coords, x_coords):
            events.append({
                'x': int(x),
                'y': int(y),
                'timestamp': timestamp,
                'polarity': -1,
                'intensity': float(-diff[y, x])
            })
        
        return events
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated events."""
        events = self.data.get('events', [])
        
        if not events:
            return {
                'event_count': 0,
                'on_ratio': 0.0,
                'off_ratio': 0.0,
                'spatial_density': 0.0,
                'average_intensity': 0.0
            }
        
        # Calculate statistics
        on_count = sum(1 for e in events if e['polarity'] > 0)
        off_count = sum(1 for e in events if e['polarity'] < 0)
        total_count = len(events)
        
        # Create spatial density map
        density_map = np.zeros(self.resolution[::-1])
        for event in events:
            x, y = event['x'], event['y']
            if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1]:
                density_map[y, x] += 1
        
        # Calculate average intensity
        avg_intensity = np.mean([e['intensity'] for e in events])
        
        return {
            'event_count': total_count,
            'on_ratio': on_count / total_count if total_count > 0 else 0.0,
            'off_ratio': off_count / total_count if total_count > 0 else 0.0,
            'spatial_density': np.mean(density_map > 0),
            'average_intensity': float(avg_intensity),
            'max_density': float(np.max(density_map))
        }
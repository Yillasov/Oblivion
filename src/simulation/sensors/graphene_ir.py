"""
Graphene-based infrared sensor implementation for ultra-sensitive thermal detection.

This module provides a graphene-based infrared sensor implementation that offers
superior sensitivity and response time compared to conventional IR sensors.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass

from src.simulation.sensors.advanced_sensors import GrapheneInfraredSensor
from src.simulation.sensors.sensor_framework import SensorConfig, SensorType
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.simulation.stealth.ir_signature_simulator import IRBand


@dataclass
class GrapheneIRConfig:
    """Configuration for graphene-based IR sensor."""
    temperature_sensitivity: float = 0.01  # Kelvin
    response_time: float = 0.001  # seconds
    spectral_band: IRBand = IRBand.MID_WAVE
    pixel_count: int = 1024  # 32x32 array
    cooling_required: bool = False  # Operates at room temperature


class GrapheneIRImplementation(GrapheneInfraredSensor):
    """
    Graphene-based infrared sensor implementation.
    
    This sensor uses graphene's unique properties to detect infrared radiation
    with exceptional sensitivity and response time.
    """
    
    def __init__(self, 
                config: SensorConfig, 
                graphene_config: GrapheneIRConfig = GrapheneIRConfig(),
                neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """Initialize graphene IR sensor implementation."""
        super().__init__(config, neuromorphic_system)
        self.graphene_config = graphene_config
        self.temperature_sensitivity = graphene_config.temperature_sensitivity
        self.response_time = graphene_config.response_time
        self.spectral_band = graphene_config.spectral_band
        
        # Graphene IR specific data
        self.data.update({
            'thermal_detections': [],
            'temperature_map': np.zeros((32, 32)),
            'hotspot_detections': []
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update graphene IR sensor data."""
        # Get targets from environment
        targets = environment.get('targets', [])
        thermal_detections = []
        hotspots = []
        
        # Platform position
        position = platform_state.get('position', np.zeros(3))
        
        # Get ambient temperature
        ambient_temp = environment.get('temperature', 20.0)
        
        # Initialize temperature map with ambient temperature
        temp_map = np.ones((32, 32)) * ambient_temp
        
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
            
            # Get target IR signature if available
            target_ir = target.get('ir_signature', 50.0)
            target_temp = ambient_temp + target_ir / 10.0  # Convert IR signature to temperature
            
            # Calculate detection probability
            detection_prob = self._calculate_detection_probability(target_temp, ambient_temp, float(distance))
            
            # Random detection based on probability
            if self.rng.random() < detection_prob:
                # Calculate angles
                azimuth = np.arctan2(rel_pos[1], rel_pos[0])
                elevation = np.arcsin(rel_pos[2] / max(0.1, distance))
                
                # Convert to degrees
                azimuth_deg = np.degrees(azimuth)
                elevation_deg = np.degrees(elevation)
                
                # Calculate temperature with noise
                measured_temp = target_temp + np.random.normal(0, self.temperature_sensitivity)
                
                thermal_detections.append({
                    'id': target_id,
                    'distance': float(distance),
                    'azimuth': float(azimuth_deg),
                    'elevation': float(elevation_deg),
                    'temperature': float(measured_temp),
                    'detection_confidence': float(detection_prob)
                })
                
                # Add to temperature map
                # Convert spherical to pixel coordinates (simplified)
                x = int((azimuth_deg + 180) / 360 * 32) % 32
                y = int((elevation_deg + 90) / 180 * 32) % 32
                
                # Update temperature map with target
                for i in range(max(0, x-2), min(32, x+3)):
                    for j in range(max(0, y-2), min(32, y+3)):
                        # Distance from center
                        d = np.sqrt((i-x)**2 + (j-y)**2)
                        if d < 3:
                            # Temperature falls off with distance from center
                            temp_map[j, i] = max(temp_map[j, i], 
                                               measured_temp * (1 - d/3))
                
                # Check if this is a hotspot
                if measured_temp > ambient_temp + 10:
                    hotspots.append({
                        'id': target_id,
                        'position': target_pos.tolist(),
                        'temperature': float(measured_temp),
                        'temperature_delta': float(measured_temp - ambient_temp)
                    })
        
        # Update sensor data
        self.data['thermal_detections'] = thermal_detections
        self.data['temperature_map'] = temp_map
        self.data['hotspot_detections'] = hotspots
    
    def _calculate_detection_probability(self, target_temp: float, 
                                       ambient_temp: float,
                                       distance: float) -> float:
        """Calculate detection probability based on temperature difference and distance."""
        # Temperature difference is key for IR detection
        temp_diff = abs(target_temp - ambient_temp)
        
        # Base detection probability based on temperature sensitivity
        base_prob = min(0.99, temp_diff / (self.temperature_sensitivity * 10))
        
        # Distance factor
        distance_factor = max(0.1, 1.0 - (distance / self.config.max_range))
        
        # Graphene sensors have better performance than traditional IR
        graphene_advantage = 1.5
        
        # Final probability
        detection_prob = min(0.99, base_prob * distance_factor * graphene_advantage)
        
        return detection_prob
    
    def get_thermal_analysis(self) -> Dict[str, Any]:
        """Get thermal analysis results."""
        detections = self.data.get('thermal_detections', [])
        hotspots = self.data.get('hotspot_detections', [])
        
        # Calculate average temperature if detections exist
        avg_temp = np.mean([d['temperature'] for d in detections]) if detections else 0.0
        
        return {
            'detection_count': len(detections),
            'hotspot_count': len(hotspots),
            'average_temperature': float(avg_temp),
            'max_temperature': float(np.max([d['temperature'] for d in detections]) if detections else 0.0),
            'temperature_sensitivity': self.temperature_sensitivity,
            'response_time': self.response_time
        }
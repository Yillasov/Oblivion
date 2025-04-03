#!/usr/bin/env python3
"""
Bio-mimetic sensor implementation inspired by natural sensing systems.

This module provides a bio-mimetic sensor implementation that mimics
natural sensing systems like bat echolocation, insect vision, or snake infrared sensing.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass

from src.simulation.sensors.advanced_sensors import BioMimeticSensor
from src.simulation.sensors.sensor_framework import SensorConfig, SensorType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


class BioInspirationSource(Enum):
    """Natural systems that inspire bio-mimetic sensors."""
    BAT = "bat"  # Echolocation
    INSECT = "insect"  # Compound eyes
    SNAKE = "snake"  # Infrared pit organs
    FISH = "fish"  # Lateral line
    BIRD = "bird"  # Magnetic field detection


@dataclass
class BioMimeticConfig:
    """Configuration for bio-mimetic sensor."""
    inspiration_source: BioInspirationSource = BioInspirationSource.BAT
    adaptation_rate: float = 0.5  # How quickly the sensor adapts to changes
    sensitivity: float = 0.8  # 0-1 scale
    energy_efficiency: float = 0.9  # 0-1 scale


class BioMimeticImplementation(BioMimeticSensor):
    """
    Bio-mimetic sensor implementation inspired by natural sensing systems.
    
    This sensor mimics the sensing capabilities of various animals to achieve
    efficient and effective detection in complex environments.
    """
    
    def __init__(self, 
                config: SensorConfig, 
                biomimetic_config: BioMimeticConfig = BioMimeticConfig(),
                neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """Initialize bio-mimetic sensor implementation."""
        super().__init__(config, neuromorphic_system)
        self.biomimetic_config = biomimetic_config
        self.bio_inspiration = biomimetic_config.inspiration_source.value
        self.adaptation_rate = biomimetic_config.adaptation_rate
        self.sensitivity = biomimetic_config.sensitivity
        
        # Bio-mimetic specific data
        self.data.update({
            'detections': [],
            'environment_features': {},
            'adaptation_level': 1.0
        })
        
        # Initialize sensing parameters based on inspiration source
        self._initialize_sensing_parameters()
    
    def _initialize_sensing_parameters(self) -> None:
        """Initialize sensing parameters based on inspiration source."""
        if self.bio_inspiration == BioInspirationSource.BAT.value:
            # Bat-inspired echolocation
            self.sensing_range = self.config.max_range * 0.8
            self.frequency_range = (20000, 120000)  # Hz
            self.pulse_duration = 0.003  # seconds
            self.data['sensing_type'] = 'echolocation'
            
        elif self.bio_inspiration == BioInspirationSource.INSECT.value:
            # Insect-inspired compound eyes
            self.sensing_range = self.config.max_range * 0.6
            self.visual_units = 100  # ommatidia
            self.motion_sensitivity = 0.9
            self.data['sensing_type'] = 'compound_vision'
            
        elif self.bio_inspiration == BioInspirationSource.SNAKE.value:
            # Snake-inspired infrared sensing
            self.sensing_range = self.config.max_range * 0.5
            self.temperature_sensitivity = 0.05  # Kelvin
            self.heat_source_detection = 0.95
            self.data['sensing_type'] = 'infrared_pit'
            
        elif self.bio_inspiration == BioInspirationSource.FISH.value:
            # Fish-inspired lateral line
            self.sensing_range = self.config.max_range * 0.3
            self.pressure_sensitivity = 0.01  # Pascal
            self.flow_detection = 0.9
            self.data['sensing_type'] = 'lateral_line'
            
        elif self.bio_inspiration == BioInspirationSource.BIRD.value:
            # Bird-inspired magnetic sensing
            self.sensing_range = self.config.max_range * 1.2
            self.magnetic_sensitivity = 0.001  # microTesla
            self.orientation_accuracy = 0.95
            self.data['sensing_type'] = 'magnetic_sense'
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update bio-mimetic sensor data."""
        # Get targets from environment
        targets = environment.get('targets', [])
        detections = []
        
        # Platform position
        position = platform_state.get('position', np.zeros(3))
        
        # Environmental features based on bio-inspiration
        environment_features = self._extract_environment_features(environment)
        
        # Process each target
        for target in targets:
            target_pos = target.get('position', np.zeros(3))
            target_id = target.get('id', 0)
            
            # Calculate relative position
            rel_pos = target_pos - position
            distance = np.linalg.norm(rel_pos)
            
            # Check if in range (based on bio-inspired sensing range)
            if distance < self.config.min_range or distance > self.sensing_range:
                continue
            
            # Calculate detection probability based on bio-inspiration
            detection_prob = self._calculate_bio_detection_probability(
                target, float(distance), environment_features
            )
            
            # Random detection based on probability
            if self.rng.random() < detection_prob:
                # Calculate angles
                azimuth = np.arctan2(rel_pos[1], rel_pos[0])
                elevation = np.arcsin(rel_pos[2] / max(0.1, distance))
                
                # Convert to degrees
                azimuth_deg = np.degrees(azimuth)
                elevation_deg = np.degrees(elevation)
                
                detections.append({
                    'id': target_id,
                    'distance': float(distance),
                    'azimuth': float(azimuth_deg),
                    'elevation': float(elevation_deg),
                    'detection_confidence': float(detection_prob),
                    'bio_features': self._extract_bio_features(target)
                })
        
        # Adapt to environment
        self._adapt_to_environment(environment_features)
        
        # Update sensor data
        self.data['detections'] = detections
        self.data['environment_features'] = environment_features
    
    def _extract_environment_features(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant environmental features based on bio-inspiration."""
        features = {}
        
        if self.bio_inspiration == BioInspirationSource.BAT.value:
            # Extract acoustic properties
            features['ambient_noise'] = environment.get('ambient_noise', 0.1)
            features['air_density'] = environment.get('air_density', 1.2)
            features['obstacles'] = len(environment.get('obstacles', []))
            
        elif self.bio_inspiration == BioInspirationSource.INSECT.value:
            # Extract visual properties
            features['light_level'] = environment.get('light_level', 0.5)
            features['motion_vectors'] = environment.get('motion_vectors', [])
            features['color_patterns'] = environment.get('color_patterns', [])
            
        elif self.bio_inspiration == BioInspirationSource.SNAKE.value:
            # Extract thermal properties
            features['ambient_temperature'] = environment.get('temperature', 20.0)
            features['thermal_gradients'] = environment.get('thermal_gradients', [])
            
        elif self.bio_inspiration == BioInspirationSource.FISH.value:
            # Extract fluid dynamics properties
            features['water_pressure'] = environment.get('pressure', 101.3)
            features['flow_velocity'] = environment.get('flow_velocity', 0.0)
            features['turbulence'] = environment.get('turbulence', 0.0)
            
        elif self.bio_inspiration == BioInspirationSource.BIRD.value:
            # Extract magnetic properties
            features['magnetic_field'] = environment.get('magnetic_field', [0.0, 0.0, 0.0])
            features['magnetic_anomalies'] = environment.get('magnetic_anomalies', [])
        
        return features
    
    def _calculate_bio_detection_probability(self, target: Dict[str, Any], 
                                           distance: float,
                                           environment_features: Dict[str, Any]) -> float:
        """Calculate detection probability based on bio-inspiration."""
        # Base detection probability
        base_prob = self.config.accuracy * (1.0 - distance / self.sensing_range)
        
        # Bio-inspired factors
        bio_factor = 1.0
        
        if self.bio_inspiration == BioInspirationSource.BAT.value:
            # Bat echolocation factors
            target_size = target.get('size', 1.0)
            target_material = target.get('material', 'solid')
            
            # Smaller targets are harder to detect
            size_factor = min(1.0, target_size)
            
            # Different materials reflect sound differently
            material_factor = {
                'solid': 0.9,
                'liquid': 0.7,
                'gas': 0.3,
                'stealth_coating': 0.4
            }.get(target_material, 0.8)
            
            # Ambient noise reduces detection
            noise_factor = max(0.5, 1.0 - environment_features.get('ambient_noise', 0.0))
            
            bio_factor = size_factor * material_factor * noise_factor
            
        elif self.bio_inspiration == BioInspirationSource.INSECT.value:
            # Insect compound eye factors
            light_level = environment_features.get('light_level', 0.5)
            target_motion = target.get('velocity', np.zeros(3))
            target_motion_magnitude = np.linalg.norm(target_motion)
            
            # Motion is easier to detect
            motion_factor = min(1.0, 0.5 + target_motion_magnitude / 10.0)
            
            # Light level affects vision
            light_factor = min(1.0, max(0.2, light_level))
            
            bio_factor = motion_factor * light_factor
            
        elif self.bio_inspiration == BioInspirationSource.SNAKE.value:
            # Snake infrared pit factors
            target_temp = target.get('temperature', 20.0)
            ambient_temp = environment_features.get('ambient_temperature', 20.0)
            
            # Temperature difference is key
            temp_diff = abs(target_temp - ambient_temp)
            temp_factor = min(1.0, temp_diff / 10.0)
            
            bio_factor = temp_factor
            
        elif self.bio_inspiration == BioInspirationSource.FISH.value:
            # Fish lateral line factors
            target_motion = target.get('velocity', np.zeros(3))
            target_motion_magnitude = np.linalg.norm(target_motion)
            
            # Motion creates pressure waves
            motion_factor = min(1.0, 0.3 + target_motion_magnitude / 5.0)
            
            # Turbulence reduces sensitivity
            turbulence = environment_features.get('turbulence', 0.0)
            turbulence_factor = max(0.5, 1.0 - turbulence)
            
            bio_factor = motion_factor * turbulence_factor
            
        elif self.bio_inspiration == BioInspirationSource.BIRD.value:
            # Bird magnetic sense factors
            target_magnetic = target.get('magnetic_signature', 0.0)
            
            # Magnetic signature affects detection
            magnetic_factor = min(1.0, 0.5 + target_magnetic)
            
            bio_factor = magnetic_factor
        
        # Apply sensitivity and adaptation
        detection_prob = base_prob * bio_factor * self.sensitivity * self.data['adaptation_level']
        
        return min(0.95, max(0.0, detection_prob))
    
    def _extract_bio_features(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Extract bio-relevant features from target."""
        features = {}
        
        if self.bio_inspiration == BioInspirationSource.BAT.value:
            features['echo_strength'] = target.get('size', 1.0) * 0.8
            features['echo_delay'] = target.get('distance', 0.0) / 340.0  # sound speed
            
        elif self.bio_inspiration == BioInspirationSource.INSECT.value:
            features['visual_contrast'] = target.get('contrast', 0.5)
            features['motion_direction'] = np.arctan2(
                target.get('velocity', [0, 0, 0])[1],
                target.get('velocity', [0, 0, 0])[0]
            )
            
        elif self.bio_inspiration == BioInspirationSource.SNAKE.value:
            features['heat_signature'] = target.get('temperature', 20.0)
            
        elif self.bio_inspiration == BioInspirationSource.FISH.value:
            features['pressure_wave'] = np.linalg.norm(target.get('velocity', [0, 0, 0])) * 0.1
            
        elif self.bio_inspiration == BioInspirationSource.BIRD.value:
            features['magnetic_deviation'] = target.get('magnetic_signature', 0.0)
        
        return features
    
    def _adapt_to_environment(self, environment_features: Dict[str, Any]) -> None:
        """Adapt sensor parameters to environment."""
        # Adaptation depends on bio-inspiration
        if self.bio_inspiration == BioInspirationSource.BAT.value:
            # Adapt to noise level
            noise = environment_features.get('ambient_noise', 0.1)
            self.data['adaptation_level'] = max(0.5, 1.0 - noise * 0.5)
            
        elif self.bio_inspiration == BioInspirationSource.INSECT.value:
            # Adapt to light level
            light = environment_features.get('light_level', 0.5)
            self.data['adaptation_level'] = max(0.3, min(1.0, light * 1.5))
            
        elif self.bio_inspiration == BioInspirationSource.SNAKE.value:
            # Adapt to ambient temperature
            temp = environment_features.get('ambient_temperature', 20.0)
            # Snakes work better in moderate temperatures
            self.data['adaptation_level'] = max(0.5, 1.0 - abs(temp - 25.0) / 20.0)
            
        elif self.bio_inspiration == BioInspirationSource.FISH.value:
            # Adapt to water pressure and turbulence
            turbulence = environment_features.get('turbulence', 0.0)
            self.data['adaptation_level'] = max(0.5, 1.0 - turbulence)
            
        elif self.bio_inspiration == BioInspirationSource.BIRD.value:
            # Birds adapt well to different conditions
            self.data['adaptation_level'] = 0.9
    
    def get_bio_sensing_stats(self) -> Dict[str, Any]:
        """Get bio-mimetic sensing statistics."""
        detections = self.data.get('detections', [])
        
        return {
            'detection_count': len(detections),
            'bio_inspiration': self.bio_inspiration,
            'adaptation_level': float(self.data.get('adaptation_level', 1.0)),
            'average_confidence': float(np.mean([d['detection_confidence'] for d in detections]) if detections else 0.0),
            'sensing_type': self.data.get('sensing_type', 'unknown'),
            'sensitivity': self.sensitivity
        }
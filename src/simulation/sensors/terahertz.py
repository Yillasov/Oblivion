#!/usr/bin/env python3
"""
Terahertz wave sensor implementation for through-material sensing.

This module provides a terahertz sensor implementation that can detect objects
through various materials with high resolution and penetration capabilities.
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

from src.simulation.sensors.advanced_sensors import TerahertzSensor
from src.simulation.sensors.sensor_framework import SensorConfig, SensorType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


class MaterialType(Enum):
    """Material types for terahertz penetration."""
    AIR = "air"
    PLASTIC = "plastic"
    FABRIC = "fabric"
    WOOD = "wood"
    DRYWALL = "drywall"
    CONCRETE = "concrete"
    METAL = "metal"


@dataclass
class TerahertzConfig:
    """Configuration for terahertz wave sensor."""
    frequency: float = 1.0  # THz
    bandwidth: float = 0.5  # THz
    penetration_depth: Dict[MaterialType, float] = field(default_factory=lambda: {
        MaterialType.AIR: 100.0,      # meters
        MaterialType.PLASTIC: 0.1,    # meters
        MaterialType.FABRIC: 0.05,    # meters
        MaterialType.WOOD: 0.02,      # meters
        MaterialType.DRYWALL: 0.03,   # meters
        MaterialType.CONCRETE: 0.01,  # meters
        MaterialType.METAL: 0.0       # meters (no penetration)
    })
    resolution: float = 0.005  # 5mm resolution
    scan_rate: float = 10.0    # Hz


class TerahertzImplementation(TerahertzSensor):
    """
    Terahertz wave sensor implementation.
    
    This sensor uses terahertz waves to detect objects through various materials
    with high resolution and penetration capabilities.
    """
    
    def __init__(self, 
                config: SensorConfig, 
                thz_config: TerahertzConfig = TerahertzConfig(),
                neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """Initialize terahertz sensor implementation."""
        super().__init__(config, neuromorphic_system)
        self.thz_config = thz_config
        self.frequency = thz_config.frequency
        self.bandwidth = thz_config.bandwidth
        self.penetration_depth = thz_config.penetration_depth
        self.resolution = thz_config.resolution
        
        # Terahertz specific data
        self.data.update({
            'thz_detections': [],
            'material_penetrations': [],
            'hidden_object_detections': []
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update terahertz sensor data."""
        # Get targets from environment
        targets = environment.get('targets', [])
        obstacles = environment.get('obstacles', [])
        
        # Platform position
        position = platform_state.get('position', np.zeros(3))
        
        # Clear previous detections
        thz_detections = []
        material_penetrations = []
        hidden_object_detections = []
        
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
            
            # Calculate angles
            azimuth = np.arctan2(rel_pos[1], rel_pos[0])
            elevation = np.arcsin(rel_pos[2] / max(0.1, distance))
            
            # Convert to degrees
            azimuth_deg = np.degrees(azimuth)
            elevation_deg = np.degrees(elevation)
            
            # Check if target is behind an obstacle
            is_hidden = False
            penetrated_material = None
            penetration_distance = 0.0
            
            for obstacle in obstacles:
                obstacle_pos = obstacle.get('position', np.zeros(3))
                obstacle_size = obstacle.get('size', np.zeros(3))
                obstacle_material = obstacle.get('material', MaterialType.CONCRETE.value)
                
                # Simple check if obstacle is between sensor and target
                obstacle_distance = np.linalg.norm(obstacle_pos - position)
                
                if obstacle_distance < distance:
                    # Check if obstacle is in line of sight
                    obstacle_direction = (obstacle_pos - position) / max(0.1, obstacle_distance)
                    target_direction = rel_pos / max(0.1, distance)
                    
                    # Simple dot product to check alignment
                    alignment = np.dot(obstacle_direction, target_direction)
                    
                    if alignment > 0.95:  # Roughly aligned
                        is_hidden = True
                        penetrated_material = obstacle_material
                        penetration_distance = obstacle_size[0]  # Simplified
                        break
            
            # Calculate detection probability
            detection_prob = self._calculate_detection_probability(
                float(distance), is_hidden, penetrated_material, float(penetration_distance)
            )
            
            # Random detection based on probability
            if self.rng.random() < detection_prob:
                # Add detection with position noise based on resolution
                position_noise = np.random.normal(0, self.resolution / distance, 2)
                
                detection = {
                    'id': target_id,
                    'distance': float(distance),
                    'azimuth': float(azimuth_deg + position_noise[0]),
                    'elevation': float(elevation_deg + position_noise[1]),
                    'detection_confidence': float(detection_prob),
                    'is_hidden': is_hidden
                }
                
                thz_detections.append(detection)
                
                if is_hidden:
                    material_penetrations.append({
                        'material': penetrated_material,
                        'thickness': float(penetration_distance),
                        'attenuation': self._calculate_attenuation(str(penetrated_material), penetration_distance)
                    })
                    
                    hidden_object_detections.append({
                        'id': target_id,
                        'distance': float(distance),
                        'material_penetrated': penetrated_material,
                        'detection_confidence': float(detection_prob)
                    })
        
        # Update sensor data
        self.data['thz_detections'] = thz_detections
        self.data['material_penetrations'] = material_penetrations
        self.data['hidden_object_detections'] = hidden_object_detections
    
    def _calculate_detection_probability(self, distance: float, is_hidden: bool,
                                       material: Optional[str], thickness: float) -> float:
        """Calculate detection probability based on distance and material penetration."""
        # Base detection probability based on distance
        base_prob = self.config.accuracy * (1.0 - distance / self.config.max_range)
        
        # If target is not hidden, return base probability
        if not is_hidden:
            return base_prob
        
        # Get material type
        material_type = None
        for mat in MaterialType:
            if mat.value == material:
                material_type = mat
                break
        
        if material_type is None:
            material_type = MaterialType.CONCRETE  # Default
        
        # Get penetration depth for this material
        max_penetration = self.penetration_depth.get(material_type, 0.01)
        
        # Calculate attenuation based on thickness
        if max_penetration <= 0.001:  # Effectively no penetration
            return 0.0
        
        # Exponential attenuation with thickness
        attenuation = np.exp(-thickness / max_penetration)
        
        # Final probability
        return base_prob * attenuation
    
    def _calculate_attenuation(self, material: str, thickness: float) -> float:
        """Calculate signal attenuation through material."""
        # Get material type
        material_type = None
        for mat in MaterialType:
            if mat.value == material:
                material_type = mat
                break
        
        if material_type is None:
            material_type = MaterialType.CONCRETE  # Default
        
        # Get penetration depth for this material
        max_penetration = self.penetration_depth.get(material_type, 0.01)
        
        # Calculate attenuation (dB)
        if max_penetration <= 0.001:  # Effectively no penetration
            return 100.0  # High attenuation
        
        # Exponential attenuation with thickness
        attenuation_factor = np.exp(-thickness / max_penetration)
        attenuation_db = -10 * np.log10(max(0.0001, attenuation_factor))
        
        return float(attenuation_db)
    
    def get_penetration_analysis(self) -> Dict[str, Any]:
        """Get analysis of material penetration capabilities."""
        material_penetrations = self.data.get('material_penetrations', [])
        
        if not material_penetrations:
            return {
                'materials_detected': 0,
                'max_thickness_penetrated': 0.0,
                'average_attenuation': 0.0
            }
        
        # Calculate statistics
        materials = set(mp['material'] for mp in material_penetrations)
        max_thickness = max(mp['thickness'] for mp in material_penetrations)
        avg_attenuation = np.mean([mp['attenuation'] for mp in material_penetrations])
        
        return {
            'materials_detected': len(materials),
            'max_thickness_penetrated': float(max_thickness),
            'average_attenuation': float(avg_attenuation),
            'hidden_objects_detected': len(self.data.get('hidden_object_detections', []))
        }
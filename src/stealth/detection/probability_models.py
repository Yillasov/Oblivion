#!/usr/bin/env python3
"""
Stealth Detection Probability Models

Simple models for calculating detection probabilities of stealth systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from enum import Enum
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import math

from src.simulation.sensors.stealth_detection import SignatureType
from src.stealth.effectiveness.stealth_effectiveness import EffectivenessRating


class DetectionModel(Enum):
    """Detection probability model types."""
    LINEAR = 0
    EXPONENTIAL = 1
    SIGMOID = 2
    THRESHOLD = 3
    RANGE_BASED = 4


class StealthDetectionProbability:
    """Models for calculating detection probability of stealth systems."""
    
    @staticmethod
    def calculate_detection_probability(
            signature_strength: float,
            sensor_sensitivity: float,
            distance: float,
            model: DetectionModel = DetectionModel.SIGMOID,
            environmental_factors: Optional[Dict[str, float]] = None
        ) -> float:
        """
        Calculate detection probability based on signature strength and sensor sensitivity.
        
        Args:
            signature_strength: Target signature strength
            sensor_sensitivity: Sensor sensitivity threshold
            distance: Distance to target in meters
            model: Detection probability model to use
            environmental_factors: Environmental factors affecting detection
            
        Returns:
            Detection probability (0.0 to 1.0)
        """
        # Apply environmental factors if provided
        env_factor = 1.0
        if environmental_factors:
            # Weather effects
            weather_factor = environmental_factors.get('weather', 1.0)
            # Time of day effects
            time_factor = environmental_factors.get('time_of_day', 1.0)
            # Terrain effects
            terrain_factor = environmental_factors.get('terrain', 1.0)
            
            env_factor = weather_factor * time_factor * terrain_factor
        
        # Adjust signature strength by environmental factors
        adjusted_signature = signature_strength * env_factor
        
        # Calculate detection probability based on selected model
        if model == DetectionModel.LINEAR:
            return StealthDetectionProbability._linear_model(
                adjusted_signature, sensor_sensitivity, distance)
        
        elif model == DetectionModel.EXPONENTIAL:
            return StealthDetectionProbability._exponential_model(
                adjusted_signature, sensor_sensitivity, distance)
        
        elif model == DetectionModel.SIGMOID:
            return StealthDetectionProbability._sigmoid_model(
                adjusted_signature, sensor_sensitivity, distance)
        
        elif model == DetectionModel.THRESHOLD:
            return StealthDetectionProbability._threshold_model(
                adjusted_signature, sensor_sensitivity, distance)
        
        elif model == DetectionModel.RANGE_BASED:
            return StealthDetectionProbability._range_based_model(
                adjusted_signature, sensor_sensitivity, distance)
        
        # Default to sigmoid model
        return StealthDetectionProbability._sigmoid_model(
            adjusted_signature, sensor_sensitivity, distance)
    
    @staticmethod
    def _linear_model(signature: float, sensitivity: float, distance: float) -> float:
        """Simple linear detection probability model."""
        # Normalize signature by sensitivity
        normalized = signature / max(0.001, sensitivity)
        # Apply distance factor (inverse square law)
        distance_factor = 1.0 / max(1.0, (distance / 1000.0) ** 2)
        # Calculate probability
        probability = normalized * distance_factor
        return min(1.0, max(0.0, probability))
    
    @staticmethod
    def _exponential_model(signature: float, sensitivity: float, distance: float) -> float:
        """Exponential detection probability model."""
        # Calculate ratio of signature to sensitivity
        ratio = signature / max(0.001, sensitivity)
        # Apply distance attenuation
        distance_factor = 1.0 / max(1.0, (distance / 1000.0) ** 2)
        # Calculate probability
        probability = 1.0 - math.exp(-2.0 * ratio * distance_factor)
        return min(1.0, max(0.0, probability))
    
    @staticmethod
    def _sigmoid_model(signature: float, sensitivity: float, distance: float) -> float:
        """Sigmoid detection probability model."""
        # Calculate normalized signature
        normalized = (signature / max(0.001, sensitivity)) - 1.0
        # Apply distance factor
        distance_factor = 1.0 / max(1.0, (distance / 1000.0) ** 2)
        # Apply sigmoid function
        probability = 1.0 / (1.0 + math.exp(-5.0 * normalized * distance_factor))
        return min(1.0, max(0.0, probability))
    
    @staticmethod
    def _threshold_model(signature: float, sensitivity: float, distance: float) -> float:
        """Threshold-based detection probability model."""
        # Calculate distance-adjusted sensitivity threshold
        distance_factor = max(1.0, (distance / 1000.0) ** 2)
        adjusted_threshold = sensitivity * distance_factor
        
        # If signature is below threshold, very low probability
        if signature < adjusted_threshold * 0.5:
            return 0.0
        # If signature is above threshold, high probability
        elif signature > adjusted_threshold * 1.5:
            return 1.0
        # Linear interpolation in the threshold region
        else:
            return (signature - adjusted_threshold * 0.5) / (adjusted_threshold * 1.0)
    
    @staticmethod
    def _range_based_model(signature: float, sensitivity: float, distance: float) -> float:
        """Range-based detection probability model."""
        # Calculate maximum detection range based on signature and sensitivity
        max_range = 1000.0 * math.sqrt(signature / max(0.001, sensitivity))
        
        # If target is beyond max range, zero probability
        if distance > max_range:
            return 0.0
        # If target is very close, high probability
        elif distance < max_range * 0.2:
            return 1.0
        # Linear decrease with range
        else:
            return 1.0 - ((distance - max_range * 0.2) / (max_range * 0.8))


class SignatureDetectionCalculator:
    """Calculator for signature-specific detection probabilities."""
    
    @staticmethod
    def calculate_radar_detection(
            rcs: float,
            radar_sensitivity: float,
            distance: float,
            environmental_factors: Optional[Dict[str, float]] = None
        ) -> float:
        """Calculate radar detection probability based on RCS."""
        # Radar typically uses exponential model
        return StealthDetectionProbability.calculate_detection_probability(
            rcs, radar_sensitivity, distance, 
            DetectionModel.EXPONENTIAL, environmental_factors
        )
    
    @staticmethod
    def calculate_ir_detection(
            ir_signature: float,
            ir_sensitivity: float,
            distance: float,
            environmental_factors: Optional[Dict[str, float]] = None
        ) -> float:
        """Calculate IR detection probability based on IR signature."""
        # IR typically uses sigmoid model
        return StealthDetectionProbability.calculate_detection_probability(
            ir_signature, ir_sensitivity, distance, 
            DetectionModel.SIGMOID, environmental_factors
        )
    
    @staticmethod
    def calculate_acoustic_detection(
            acoustic_signature: float,
            acoustic_sensitivity: float,
            distance: float,
            environmental_factors: Optional[Dict[str, float]] = None
        ) -> float:
        """Calculate acoustic detection probability based on acoustic signature."""
        # Acoustic typically uses range-based model
        return StealthDetectionProbability.calculate_detection_probability(
            acoustic_signature, acoustic_sensitivity, distance, 
            DetectionModel.RANGE_BASED, environmental_factors
        )
    
    @staticmethod
    def calculate_em_detection(
            em_signature: float,
            em_sensitivity: float,
            distance: float,
            environmental_factors: Optional[Dict[str, float]] = None
        ) -> float:
        """Calculate EM detection probability based on EM signature."""
        # EM typically uses threshold model
        return StealthDetectionProbability.calculate_detection_probability(
            em_signature, em_sensitivity, distance, 
            DetectionModel.THRESHOLD, environmental_factors
        )
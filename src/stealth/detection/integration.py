"""
Stealth Detection Integration

Integrates stealth detection probability models with sensor systems.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from src.simulation.sensors.stealth_detection import (
    SignatureType, 
    StealthDetectionSensor,
    MultiSignatureDetector
)
from src.stealth.detection.probability_models import (
    StealthDetectionProbability,
    SignatureDetectionCalculator
)


class StealthDetectionEnhancer:
    """Enhances stealth detection with probability models."""
    
    def __init__(self, multi_detector: Optional[MultiSignatureDetector] = None):
        """Initialize stealth detection enhancer."""
        self.multi_detector = multi_detector
        self.environmental_factors = {}
        
    def set_environmental_factors(self, factors: Dict[str, float]) -> None:
        """Set environmental factors affecting detection."""
        self.environmental_factors = factors
        
    def enhance_detection_results(self, 
                                 detection_results: Dict[str, Any],
                                 platform_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance detection results with probability models.
        
        Args:
            detection_results: Original detection results
            platform_state: Current platform state
            
        Returns:
            Enhanced detection results with probabilities
        """
        # Get detections
        enhanced_results = detection_results.copy()
        
        # Enhance radar detections
        if 'stealth_detections' in enhanced_results:
            for i, detection in enumerate(enhanced_results['stealth_detections']):
                # Calculate enhanced probability
                probability = SignatureDetectionCalculator.calculate_radar_detection(
                    detection.get('rcs', 1.0),
                    0.1,  # Default sensitivity
                    detection.get('distance', 1000.0),
                    self.environmental_factors
                )
                # Update detection with probability
                enhanced_results['stealth_detections'][i]['detection_probability'] = probability
        
        # Enhance IR detections
        if 'ir_detections' in enhanced_results:
            for i, detection in enumerate(enhanced_results['ir_detections']):
                # Calculate enhanced probability
                probability = SignatureDetectionCalculator.calculate_ir_detection(
                    detection.get('ir_signature', 50.0),
                    5.0,  # Default sensitivity
                    detection.get('distance', 1000.0),
                    self.environmental_factors
                )
                # Update detection with probability
                enhanced_results['ir_detections'][i]['detection_probability'] = probability
        
        # Enhance acoustic detections
        if 'acoustic_detections' in enhanced_results:
            for i, detection in enumerate(enhanced_results['acoustic_detections']):
                # Calculate enhanced probability
                probability = SignatureDetectionCalculator.calculate_acoustic_detection(
                    detection.get('acoustic_signature', 70.0),
                    10.0,  # Default sensitivity
                    detection.get('distance', 1000.0),
                    self.environmental_factors
                )
                # Update detection with probability
                enhanced_results['acoustic_detections'][i]['detection_probability'] = probability
        
        # Enhance EM detections
        if 'em_detections' in enhanced_results:
            for i, detection in enumerate(enhanced_results['em_detections']):
                # Calculate enhanced probability
                probability = SignatureDetectionCalculator.calculate_em_detection(
                    detection.get('em_signature', -60.0),
                    -80.0,  # Default sensitivity
                    detection.get('distance', 1000.0),
                    self.environmental_factors
                )
                # Update detection with probability
                enhanced_results['em_detections'][i]['detection_probability'] = probability
        
        return enhanced_results
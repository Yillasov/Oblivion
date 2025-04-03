#!/usr/bin/env python3
"""
Stealth Detection Module

Extends the sensor framework to interact with stealth systems and detect
various signatures (radar, IR, acoustic, electromagnetic).
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum

from src.simulation.sensors.sensor_framework import Sensor, SensorConfig, SensorType
from src.core.utils.logging_framework import get_logger

# Import stealth simulators
from src.simulation.stealth.rcs_simulator import RCSSimulator, RCSFrequencyBand
from src.simulation.stealth.ir_signature_simulator import IRSignatureSimulator, IRBand
from src.simulation.stealth.acoustic_simulator import AcousticSignatureSimulator, FrequencyRange
from src.simulation.stealth.em_signature_simulator import EMSignatureSimulator, EMBand

logger = get_logger("stealth_detection")


class SignatureType(Enum):
    """Types of signatures that can be detected."""
    RADAR = 0
    INFRARED = 1
    ACOUSTIC = 2
    ELECTROMAGNETIC = 3


class StealthDetectionSensor(Sensor):
    """Base class for sensors that can detect stealth signatures."""
    
    def __init__(self, config: SensorConfig, signature_type: SignatureType):
        """Initialize stealth detection sensor."""
        super().__init__(config)
        self.signature_type = signature_type
        self.detection_threshold = 0.1  # Minimum detectable signature
        self.signature_data = {}
        
    def detect_signature(self, 
                        target_signature: float,
                        distance: float,
                        environmental_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect a stealth signature based on sensor capabilities.
        
        Args:
            target_signature: Signature value of the target
            distance: Distance to target in meters
            environmental_conditions: Current environmental conditions
            
        Returns:
            Detection results
        """
        # Calculate attenuation due to distance
        attenuation = self._calculate_distance_attenuation(distance)
        
        # Calculate environmental effects
        env_factor = self._calculate_environmental_factor(environmental_conditions)
        
        # Calculate sensor effectiveness based on accuracy and noise
        sensor_factor = self.config.accuracy * (1.0 - self.config.noise_factor)
        
        # Calculate final detection probability
        attenuated_signature = target_signature * attenuation * env_factor
        detection_probability = min(1.0, max(0.0, 
            (attenuated_signature - self.detection_threshold) * sensor_factor))
        
        # Determine if detection occurred
        detected = self.rng.random() < detection_probability
        
        return {
            "detected": detected,
            "confidence": detection_probability,
            "signature_strength": attenuated_signature,
            "distance": distance,
            "attenuation_factor": attenuation,
            "environmental_factor": env_factor
        }
    
    def _calculate_distance_attenuation(self, distance: float) -> float:
        """Calculate signature attenuation due to distance."""
        if distance <= 0:
            return 1.0
            
        # Different signature types attenuate differently with distance
        if self.signature_type == SignatureType.RADAR:
            # Radar follows inverse fourth power law
            return min(1.0, (self.config.max_range / max(1.0, distance))**4)
            
        elif self.signature_type == SignatureType.INFRARED:
            # IR follows inverse square law with atmospheric attenuation
            return min(1.0, (self.config.max_range / max(1.0, distance))**2)
            
        elif self.signature_type == SignatureType.ACOUSTIC:
            # Acoustic follows inverse square law with additional attenuation
            return min(1.0, (self.config.max_range / max(1.0, distance))**2 * 0.8)
            
        elif self.signature_type == SignatureType.ELECTROMAGNETIC:
            # EM follows inverse square law
            return min(1.0, (self.config.max_range / max(1.0, distance))**2)
            
        return 1.0
    
    def _calculate_environmental_factor(self, environmental_conditions: Dict[str, Any]) -> float:
        """Calculate environmental effects on signature detection."""
        # Base factor
        factor = 1.0
        
        # Extract environmental conditions
        weather = environmental_conditions.get("weather", "CLEAR")
        precipitation = environmental_conditions.get("precipitation", 0.0)
        humidity = environmental_conditions.get("humidity", 0.5)
        temperature = environmental_conditions.get("temperature", 15.0)
        
        # Apply weather effects
        if weather in ["RAIN", "SNOW"]:
            if self.signature_type == SignatureType.RADAR:
                factor *= 0.8  # Rain/snow affects radar
            elif self.signature_type == SignatureType.INFRARED:
                factor *= 0.7  # Rain/snow affects IR more
            elif self.signature_type == SignatureType.ELECTROMAGNETIC:
                factor *= 0.9  # Rain/snow affects EM slightly
                
        if weather == "FOG":
            if self.signature_type == SignatureType.INFRARED:
                factor *= 0.6  # Fog affects IR significantly
            elif self.signature_type == SignatureType.RADAR:
                factor *= 0.95  # Fog affects radar slightly
                
        # Apply humidity effects
        if humidity > 0.7:
            if self.signature_type == SignatureType.INFRARED:
                factor *= 0.9  # High humidity affects IR
            elif self.signature_type == SignatureType.ELECTROMAGNETIC:
                factor *= 0.95  # High humidity affects EM slightly
                
        # Apply temperature effects
        if temperature < 0:
            if self.signature_type == SignatureType.INFRARED:
                factor *= 1.1  # Cold temperatures improve IR contrast
                
        return factor


class RadarStealthDetector(StealthDetectionSensor):
    """Radar sensor that can detect RCS signatures."""
    
    def __init__(self, config: SensorConfig, frequency_band: RCSFrequencyBand = RCSFrequencyBand.X_BAND):
        """Initialize radar stealth detector."""
        super().__init__(config, SignatureType.RADAR)
        self.frequency_band = frequency_band
        self.data.update({
            'stealth_detections': [],
            'rcs_values': {}
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update radar sensor data with stealth detection."""
        # First update regular radar data
        super()._update_sensor_data(platform_state, environment)
        
        # Get targets from environment
        targets = environment.get('targets', [])
        stealth_detections = []
        
        for target in targets:
            target_pos = target.get('position', np.zeros(3))
            target_id = target.get('id', 0)
            
            # Calculate relative position
            position = platform_state.get('position', np.zeros(3))
            rel_pos = target_pos - position
            distance = np.linalg.norm(rel_pos)
            
            # Check if in range
            if distance < self.config.min_range or distance > self.config.max_range:
                continue
            
            # Get target RCS if available
            target_rcs = target.get('rcs', 1.0)
            
            # Calculate azimuth and elevation for RCS lookup
            azimuth = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
            elevation = np.degrees(np.arcsin(rel_pos[2] / max(0.1, distance)))
            
            # Detect stealth signature
            detection_result = self.detect_signature(
                target_signature=target_rcs,
                distance=float(distance),
                environmental_conditions=environment
            )
            
            if detection_result["detected"]:
                stealth_detections.append({
                    'target_id': target_id,
                    'distance': distance,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'rcs': detection_result["signature_strength"],
                    'confidence': detection_result["confidence"]
                })
                
                # Store RCS value for this target
                self.data['rcs_values'][target_id] = detection_result["signature_strength"]
        
        # Update stealth detections
        self.data['stealth_detections'] = stealth_detections


class IRStealthDetector(StealthDetectionSensor):
    """Infrared sensor that can detect IR signatures."""
    
    def __init__(self, config: SensorConfig, ir_band: IRBand = IRBand.MID_WAVE):
        """Initialize IR stealth detector."""
        super().__init__(config, SignatureType.INFRARED)
        self.ir_band = ir_band
        self.data.update({
            'ir_detections': [],
            'ir_values': {}
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update IR sensor data with stealth detection."""
        # Get targets from environment
        targets = environment.get('targets', [])
        ir_detections = []
        
        for target in targets:
            target_pos = target.get('position', np.zeros(3))
            target_id = target.get('id', 0)
            
            # Calculate relative position
            position = platform_state.get('position', np.zeros(3))
            rel_pos = target_pos - position
            distance = np.linalg.norm(rel_pos)
            
            # Check if in range
            if distance < self.config.min_range or distance > self.config.max_range:
                continue
            
            # Get target IR signature if available
            target_ir = target.get('ir_signature', 50.0)
            
            # Calculate azimuth and elevation
            azimuth = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
            elevation = np.degrees(np.arcsin(rel_pos[2] / max(0.1, distance)))
            
            # Detect stealth signature
            detection_result = self.detect_signature(
                target_signature=target_ir,
                distance=float(distance),
                environmental_conditions=environment
            )
            
            if detection_result["detected"]:
                ir_detections.append({
                    'target_id': target_id,
                    'distance': distance,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'ir_signature': detection_result["signature_strength"],
                    'confidence': detection_result["confidence"]
                })
                
                # Store IR value for this target
                self.data['ir_values'][target_id] = detection_result["signature_strength"]
        
        # Update IR detections
        self.data['ir_detections'] = ir_detections


class AcousticStealthDetector(StealthDetectionSensor):
    """Acoustic sensor that can detect acoustic signatures."""
    
    def __init__(self, config: SensorConfig, frequency_range: FrequencyRange = FrequencyRange.FULL):
        """Initialize acoustic stealth detector."""
        super().__init__(config, SignatureType.ACOUSTIC)
        self.frequency_range = frequency_range
        self.data.update({
            'acoustic_detections': [],
            'acoustic_values': {}
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update acoustic sensor data with stealth detection."""
        # Get targets from environment
        targets = environment.get('targets', [])
        acoustic_detections = []
        
        for target in targets:
            target_pos = target.get('position', np.zeros(3))
            target_id = target.get('id', 0)
            
            # Calculate relative position
            position = platform_state.get('position', np.zeros(3))
            rel_pos = target_pos - position
            distance = np.linalg.norm(rel_pos)
            
            # Check if in range
            if distance < self.config.min_range or distance > self.config.max_range:
                continue
            
            # Get target acoustic signature if available
            target_acoustic = target.get('acoustic_signature', 70.0)
            
            # Calculate azimuth and elevation
            azimuth = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
            elevation = np.degrees(np.arcsin(rel_pos[2] / max(0.1, distance)))
            
            # Detect stealth signature
            detection_result = self.detect_signature(
                target_signature=target_acoustic,
                distance=float(distance),
                environmental_conditions=environment
            )
            
            if detection_result["detected"]:
                acoustic_detections.append({
                    'target_id': target_id,
                    'distance': distance,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'acoustic_signature': detection_result["signature_strength"],
                    'confidence': detection_result["confidence"]
                })
                
                # Store acoustic value for this target
                self.data['acoustic_values'][target_id] = detection_result["signature_strength"]
        
        # Update acoustic detections
        self.data['acoustic_detections'] = acoustic_detections


class EMStealthDetector(StealthDetectionSensor):
    """Electromagnetic sensor that can detect EM signatures."""
    
    def __init__(self, config: SensorConfig, em_band: EMBand = EMBand.UHF):
        """Initialize EM stealth detector."""
        super().__init__(config, SignatureType.ELECTROMAGNETIC)
        self.em_band = em_band
        self.data.update({
            'em_detections': [],
            'em_values': {}
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update EM sensor data with stealth detection."""
        # Get targets from environment
        targets = environment.get('targets', [])
        em_detections = []
        
        for target in targets:
            target_pos = target.get('position', np.zeros(3))
            target_id = target.get('id', 0)
            
            # Calculate relative position
            position = platform_state.get('position', np.zeros(3))
            rel_pos = target_pos - position
            distance = np.linalg.norm(rel_pos)
            
            # Check if in range
            if distance < self.config.min_range or distance > self.config.max_range:
                continue
            
            # Get target EM signature if available
            target_em = target.get('em_signature', -60.0)
            
            # Calculate azimuth and elevation
            azimuth = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
            elevation = np.degrees(np.arcsin(rel_pos[2] / max(0.1, distance)))
            
            # Detect stealth signature
            detection_result = self.detect_signature(
                target_signature=target_em,
                distance=float(distance),
                environmental_conditions=environment
            )
            
            if detection_result["detected"]:
                em_detections.append({
                    'target_id': target_id,
                    'distance': distance,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'em_signature': detection_result["signature_strength"],
                    'confidence': detection_result["confidence"]
                })
                
                # Store EM value for this target
                self.data['em_values'][target_id] = detection_result["signature_strength"]
        
        # Update EM detections
        self.data['em_detections'] = em_detections


class MultiSignatureDetector:
    """Integrates multiple stealth detection sensors for comprehensive signature analysis."""
    
    def __init__(self):
        """Initialize multi-signature detector."""
        self.detectors = {}
        self.fusion_data = {
            'integrated_detections': [],
            'signature_correlations': {}
        }
        
    def add_detector(self, detector_id: str, detector: StealthDetectionSensor) -> None:
        """Add a stealth detector to the multi-signature detector."""
        self.detectors[detector_id] = detector
        
    def update(self, time_now: float, platform_state: Dict[str, Any], 
              environment: Dict[str, Any]) -> Dict[str, Any]:
        """Update all detectors and perform signature fusion."""
        # Update all detectors
        for detector_id, detector in self.detectors.items():
            detector.update(time_now, platform_state, environment)
            
        # Perform signature fusion
        self._fuse_signatures()
        
        return self.fusion_data
        
    def _fuse_signatures(self) -> None:
        """Fuse signatures from different detectors."""
        # Clear previous fusion data
        self.fusion_data['integrated_detections'] = []
        self.fusion_data['signature_correlations'] = {}
        
        # Collect all target IDs from all detectors
        all_target_ids = set()
        for detector in self.detectors.values():
            if isinstance(detector, RadarStealthDetector):
                all_target_ids.update(detector.data.get('rcs_values', {}).keys())
            elif isinstance(detector, IRStealthDetector):
                all_target_ids.update(detector.data.get('ir_values', {}).keys())
            elif isinstance(detector, AcousticStealthDetector):
                all_target_ids.update(detector.data.get('acoustic_values', {}).keys())
            elif isinstance(detector, EMStealthDetector):
                all_target_ids.update(detector.data.get('em_values', {}).keys())
        
        # Process each target
        for target_id in all_target_ids:
            target_signatures = {}
            target_confidences = {}
            
            # Collect signatures from each detector
            for detector_id, detector in self.detectors.items():
                if isinstance(detector, RadarStealthDetector):
                    if target_id in detector.data.get('rcs_values', {}):
                        target_signatures['radar'] = detector.data['rcs_values'][target_id]
                        # Find confidence from detections
                        for detection in detector.data.get('stealth_detections', []):
                            if detection.get('target_id') == target_id:
                                target_confidences['radar'] = detection.get('confidence', 0.5)
                                break
                
                elif isinstance(detector, IRStealthDetector):
                    if target_id in detector.data.get('ir_values', {}):
                        target_signatures['ir'] = detector.data['ir_values'][target_id]
                        for detection in detector.data.get('ir_detections', []):
                            if detection.get('target_id') == target_id:
                                target_confidences['ir'] = detection.get('confidence', 0.5)
                                break
                
                elif isinstance(detector, AcousticStealthDetector):
                    if target_id in detector.data.get('acoustic_values', {}):
                        target_signatures['acoustic'] = detector.data['acoustic_values'][target_id]
                        for detection in detector.data.get('acoustic_detections', []):
                            if detection.get('target_id') == target_id:
                                target_confidences['acoustic'] = detection.get('confidence', 0.5)
                                break
                
                elif isinstance(detector, EMStealthDetector):
                    if target_id in detector.data.get('em_values', {}):
                        target_signatures['em'] = detector.data['em_values'][target_id]
                        for detection in detector.data.get('em_detections', []):
                            if detection.get('target_id') == target_id:
                                target_confidences['em'] = detection.get('confidence', 0.5)
                                break
            
            # Calculate integrated confidence
            if target_confidences:
                # Weighted average of confidences
                integrated_confidence = sum(target_confidences.values()) / len(target_confidences)
                
                # Boost confidence if multiple signature types detected
                signature_count = len(target_signatures)
                if signature_count > 1:
                    # Boost by up to 20% for all 4 signature types
                    confidence_boost = min(0.2, 0.05 * (signature_count - 1))
                    integrated_confidence = min(1.0, integrated_confidence * (1.0 + confidence_boost))
                
                # Add to integrated detections
                self.fusion_data['integrated_detections'].append({
                    'target_id': target_id,
                    'signatures': target_signatures,
                    'confidences': target_confidences,
                    'integrated_confidence': integrated_confidence,
                    'signature_count': signature_count
                })
                
                # Calculate signature correlations
                if len(target_signatures) > 1:
                    signature_types = list(target_signatures.keys())
                    for i in range(len(signature_types)):
                        for j in range(i+1, len(signature_types)):
                            type1 = signature_types[i]
                            type2 = signature_types[j]
                            
                            # Simple correlation - both detected
                            correlation_key = f"{type1}_{type2}"
                            if correlation_key not in self.fusion_data['signature_correlations']:
                                self.fusion_data['signature_correlations'][correlation_key] = 0
                            
                            self.fusion_data['signature_correlations'][correlation_key] += 1


def create_stealth_detection_sensors() -> MultiSignatureDetector:
    """
    Create a set of stealth detection sensors.
    
    Returns:
        MultiSignatureDetector: Multi-signature detector with stealth detection sensors
    """
    multi_detector = MultiSignatureDetector()
    
    # Create radar stealth detector
    radar_config = SensorConfig(
        type=SensorType.RADAR,
        name="stealth_radar",
        update_rate=5.0,
        fov_horizontal=120.0,
        fov_vertical=60.0,
        max_range=80000.0,
        accuracy=0.85,
        noise_factor=0.03
    )
    radar_detector = RadarStealthDetector(radar_config, RCSFrequencyBand.X_BAND)
    multi_detector.add_detector("radar", radar_detector)
    
    # Create IR stealth detector
    ir_config = SensorConfig(
        type=SensorType.INFRARED,
        name="stealth_ir",
        update_rate=10.0,
        fov_horizontal=60.0,
        fov_vertical=40.0,
        max_range=40000.0,
        accuracy=0.9,
        noise_factor=0.02
    )
    ir_detector = IRStealthDetector(ir_config, IRBand.MID_WAVE)
    multi_detector.add_detector("ir", ir_detector)
    
    # Create acoustic stealth detector
    acoustic_config = SensorConfig(
        type=SensorType.INFRARED,  # Reusing enum, no acoustic type available
        name="stealth_acoustic",
        update_rate=2.0,
        fov_horizontal=360.0,  # Omnidirectional
        fov_vertical=180.0,
        max_range=15000.0,
        accuracy=0.75,
        noise_factor=0.05
    )
    acoustic_detector = AcousticStealthDetector(acoustic_config, FrequencyRange.FULL)
    multi_detector.add_detector("acoustic", acoustic_detector)
    
    # Create EM stealth detector
    em_config = SensorConfig(
        type=SensorType.RADAR,  # Reusing enum, no EM type available
        name="stealth_em",
        update_rate=4.0,
        fov_horizontal=360.0,  # Omnidirectional
        fov_vertical=180.0,
        max_range=50000.0,
        accuracy=0.8,
        noise_factor=0.04
    )
    em_detector = EMStealthDetector(em_config, EMBand.UHF)
    multi_detector.add_detector("em", em_detector)
    
    return multi_detector
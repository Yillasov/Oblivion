"""
Multi-spectral EO/IR camera implementation for advanced target detection.

This module provides a multi-spectral camera implementation that can detect targets
across multiple bands from visible to far infrared simultaneously.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass, field

from src.simulation.sensors.advanced_sensors import MultiSpectralEOIRSensor
from src.simulation.sensors.sensor_framework import SensorConfig, SensorType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


class SpectralBand(Enum):
    """Spectral bands for multi-spectral imaging."""
    VISIBLE = "visible"
    NEAR_IR = "near_ir"
    MID_IR = "mid_ir"
    FAR_IR = "far_ir"


@dataclass
class MultiSpectralConfig:
    """Configuration for multi-spectral imaging system."""
    active_bands: List[SpectralBand] = field(default_factory=list)
    fusion_algorithm: str = "weighted"  # weighted, bayesian, neural
    resolution: Dict[str, int] = field(default_factory=dict)
    field_of_view: float = 30.0  # degrees
    integration_time: float = 0.01  # seconds
    
    def __post_init__(self):
        if not self.active_bands:
            self.active_bands = [SpectralBand.VISIBLE, SpectralBand.MID_IR]
        
        if not self.resolution:
            self.resolution = {
                "visible": 1920,
                "near_ir": 1280,
                "mid_ir": 640,
                "far_ir": 320
            }


class MultiSpectralImplementation(MultiSpectralEOIRSensor):
    """
    Multi-spectral EO/IR camera implementation.
    
    This sensor captures multiple spectral bands simultaneously, enabling
    enhanced target detection in various lighting and weather conditions.
    """
    
    def __init__(self, 
                config: SensorConfig, 
                multispectral_config: MultiSpectralConfig = MultiSpectralConfig(),
                neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """Initialize multi-spectral camera implementation."""
        super().__init__(config, neuromorphic_system)
        self.multispectral_config = multispectral_config
        self.active_bands = [band.value for band in multispectral_config.active_bands]
        self.fusion_algorithm = multispectral_config.fusion_algorithm
        self.resolution = multispectral_config.resolution
        
        # Multi-spectral specific data
        self.data.update({
            'band_detections': {band: [] for band in self.active_bands},
            'fused_detections': [],
            'band_images': {band: None for band in self.active_bands},
            'fused_image': None
        })
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update multi-spectral sensor data."""
        # Get targets from environment
        targets = environment.get('targets', [])
        
        # Clear previous detections
        for band in self.active_bands:
            self.data['band_detections'][band] = []
        self.data['fused_detections'] = []
        
        # Platform position
        position = platform_state.get('position', np.zeros(3))
        
        # Environmental conditions
        light_level = environment.get('light_level', 0.8)
        weather = environment.get('weather', 'clear')
        temperature = environment.get('temperature', 20.0)
        
        # Process each target for each active band
        band_detections = {band: [] for band in self.active_bands}
        
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
            
            # Process each band
            for band in self.active_bands:
                # Get band-specific target signature
                signature = self._get_band_signature(target, band)
                
                # Calculate detection probability for this band
                detection_prob = self._calculate_band_detection_probability(
                    band, signature, float(distance), light_level, weather, temperature
                )
                
                # Random detection based on probability
                if self.rng.random() < detection_prob:
                    # Add noise to position based on band resolution
                    resolution_factor = self._get_resolution_factor(band)
                    position_noise = np.random.normal(0, 0.5 / resolution_factor, 2)
                    
                    band_detections[band].append({
                        'id': target_id,
                        'distance': float(distance),
                        'azimuth': float(azimuth_deg + position_noise[0]),
                        'elevation': float(elevation_deg + position_noise[1]),
                        'signature': float(signature),
                        'detection_confidence': float(detection_prob)
                    })
        
        # Update band detections
        self.data['band_detections'] = band_detections
        
        # Generate simulated band images
        self._generate_band_images(band_detections, environment)
        
        # Fuse detections from all bands
        fused_detections = self._fuse_detections(band_detections)
        self.data['fused_detections'] = fused_detections
    
    def _get_band_signature(self, target: Dict[str, Any], band: str) -> float:
        """Get target signature for specific band."""
        # Default signatures by band
        default_signatures = {
            "visible": target.get('visual_signature', 0.7),
            "near_ir": target.get('near_ir_signature', 0.6),
            "mid_ir": target.get('ir_signature', 50.0) / 100.0,  # Normalize to 0-1
            "far_ir": target.get('thermal_signature', target.get('temperature', 20.0)) / 100.0
        }
        
        return default_signatures.get(band, 0.5)
    
    def _calculate_band_detection_probability(self, band: str, signature: float,
                                           distance: float, light_level: float,
                                           weather: str, temperature: float) -> float:
        """Calculate detection probability for specific band."""
        # Base detection probability
        base_prob = self.config.accuracy * (1.0 - distance / self.config.max_range)
        
        # Band-specific factors
        band_factors = {
            "visible": {
                "light_dependency": 0.8,  # Highly dependent on light
                "weather_impact": 0.7,    # Moderate impact from weather
                "temp_impact": 0.1        # Low impact from temperature
            },
            "near_ir": {
                "light_dependency": 0.4,  # Less dependent on light
                "weather_impact": 0.5,    # Moderate impact from weather
                "temp_impact": 0.2        # Low impact from temperature
            },
            "mid_ir": {
                "light_dependency": 0.1,  # Very low dependency on light
                "weather_impact": 0.3,    # Lower impact from weather
                "temp_impact": 0.6        # Moderate impact from temperature
            },
            "far_ir": {
                "light_dependency": 0.0,  # No dependency on light
                "weather_impact": 0.2,    # Low impact from weather
                "temp_impact": 0.9        # High impact from temperature
            }
        }
        
        factors = band_factors.get(band, {
            "light_dependency": 0.5,
            "weather_impact": 0.5,
            "temp_impact": 0.5
        })
        
        # Light level factor (affects mainly visible and near IR)
        light_factor = 1.0
        if factors["light_dependency"] > 0:
            light_factor = min(1.0, light_level / factors["light_dependency"])
        
        # Weather factor
        weather_factor = {
            "clear": 1.0,
            "cloudy": 0.8,
            "rain": 0.6,
            "snow": 0.5,
            "fog": 0.3,
            "sandstorm": 0.2
        }.get(weather, 0.7)
        
        # Apply weather impact based on band sensitivity
        weather_factor = 1.0 - (factors["weather_impact"] * (1.0 - weather_factor))
        
        # Temperature factor (affects mainly IR bands)
        temp_factor = 1.0
        if factors["temp_impact"] > 0:
            # Optimal temperature range depends on the band
            optimal_temp = 20.0
            temp_diff = abs(temperature - optimal_temp)
            temp_factor = max(0.5, 1.0 - (temp_diff / 50.0) * factors["temp_impact"])
        
        # Signature factor
        signature_factor = min(1.0, signature * 1.5)
        
        # Final probability
        detection_prob = base_prob * light_factor * weather_factor * temp_factor * signature_factor
        
        return min(0.95, max(0.1, detection_prob))
    
    def _get_resolution_factor(self, band: str) -> float:
        """Get resolution factor for band."""
        max_resolution = max(self.resolution.values())
        band_resolution = self.resolution.get(band, max_resolution)
        return band_resolution / max_resolution
    
    def _generate_band_images(self, band_detections: Dict[str, List[Dict[str, Any]]], 
                            environment: Dict[str, Any]) -> None:
        """Generate simulated images for each band."""
        # Simple placeholder implementation
        for band in self.active_bands:
            # Create empty image based on band resolution
            resolution = self.resolution.get(band, 640)
            image_size = min(resolution, 64)  # Limit size for simulation
            image = np.zeros((image_size, image_size))
            
            # Add detections to image
            for detection in band_detections.get(band, []):
                # Convert angles to image coordinates
                x = int((detection['azimuth'] + 180) / 360 * image_size) % image_size
                y = int((detection['elevation'] + 90) / 180 * image_size) % image_size
                
                # Add detection with gaussian blob
                radius = max(1, int(image_size / 20))
                intensity = detection['signature'] * 255
                
                for i in range(max(0, x-radius), min(image_size, x+radius+1)):
                    for j in range(max(0, y-radius), min(image_size, y+radius+1)):
                        dist = np.sqrt((i-x)**2 + (j-y)**2)
                        if dist <= radius:
                            # Gaussian falloff
                            image[j, i] = min(255, image[j, i] + intensity * np.exp(-(dist/radius)**2))
            
            self.data['band_images'][band] = image
        
        # Generate fused image if we have multiple active bands
        if len(self.active_bands) > 1:
            self._generate_fused_image()
    
    def _generate_fused_image(self) -> None:
        """Generate fused image from band images."""
        # Get first band image to determine size
        first_band = next(iter(self.data['band_images']))
        if self.data['band_images'][first_band] is None:
            return
            
        image_size = self.data['band_images'][first_band].shape[0]
        fused_image = np.zeros((image_size, image_size))
        
        if self.fusion_algorithm == "weighted":
            # Weighted average of all bands
            weights = {
                "visible": 0.4,
                "near_ir": 0.2,
                "mid_ir": 0.3,
                "far_ir": 0.1
            }
            
            total_weight = 0
            for band in self.active_bands:
                if self.data['band_images'][band] is not None:
                    weight = weights.get(band, 0.25)
                    fused_image += self.data['band_images'][band] * weight
                    total_weight += weight
            
            if total_weight > 0:
                fused_image /= total_weight
                
        elif self.fusion_algorithm == "bayesian":
            # Simple Bayesian fusion (product of normalized probabilities)
            fused_image = np.ones((image_size, image_size))
            for band in self.active_bands:
                if self.data['band_images'][band] is not None:
                    # Normalize to 0-1 probability
                    norm_image = self.data['band_images'][band] / 255.0
                    # Avoid zeros (use small epsilon)
                    norm_image = np.maximum(norm_image, 0.01)
                    fused_image *= norm_image
            
            # Scale back to 0-255
            fused_image = np.power(fused_image, 1.0/len(self.active_bands)) * 255
            
        elif self.fusion_algorithm == "neural":
            # Simplified neural fusion (max value across bands)
            for band in self.active_bands:
                if self.data['band_images'][band] is not None:
                    fused_image = np.maximum(fused_image, self.data['band_images'][band])
        
        self.data['fused_image'] = fused_image
    
    def _fuse_detections(self, band_detections: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Fuse detections from multiple bands."""
        # Group detections by target ID
        targets = {}
        
        for band, detections in band_detections.items():
            for detection in detections:
                target_id = detection['id']
                if target_id not in targets:
                    targets[target_id] = {
                        'bands': [],
                        'detections': []
                    }
                
                targets[target_id]['bands'].append(band)
                targets[target_id]['detections'].append(detection)
        
        # Create fused detections
        fused_detections = []
        
        for target_id, target_data in targets.items():
            if not target_data['detections']:
                continue
                
            # Average position across bands
            avg_distance = np.mean([d['distance'] for d in target_data['detections']])
            avg_azimuth = np.mean([d['azimuth'] for d in target_data['detections']])
            avg_elevation = np.mean([d['elevation'] for d in target_data['detections']])
            
            # Confidence increases with number of bands that detected the target
            band_count = len(set(target_data['bands']))
            confidence_boost = min(0.3, 0.1 * band_count)
            
            # Average confidence across bands with boost for multi-band detection
            avg_confidence = np.mean([d['detection_confidence'] for d in target_data['detections']])
            fused_confidence = min(0.99, avg_confidence + confidence_boost)
            
            # Create fused detection
            fused_detections.append({
                'id': target_id,
                'distance': float(avg_distance),
                'azimuth': float(avg_azimuth),
                'elevation': float(avg_elevation),
                'detection_confidence': float(fused_confidence),
                'detected_bands': list(set(target_data['bands'])),
                'band_count': band_count
            })
        
        return fused_detections
    
    def get_multispectral_analysis(self) -> Dict[str, Any]:
        """Get multi-spectral analysis results."""
        fused_detections = self.data.get('fused_detections', [])
        band_detections = self.data.get('band_detections', {})
        
        # Count detections per band
        band_counts = {band: len(detections) for band, detections in band_detections.items()}
        
        # Calculate average confidence per band
        band_confidences = {}
        for band, detections in band_detections.items():
            if detections:
                band_confidences[band] = float(np.mean([d['detection_confidence'] for d in detections]))
            else:
                band_confidences[band] = 0.0
        
        return {
            'fused_detection_count': len(fused_detections),
            'band_detection_counts': band_counts,
            'band_confidences': band_confidences,
            'active_bands': self.active_bands,
            'fusion_algorithm': self.fusion_algorithm,
            'average_fused_confidence': float(np.mean([d['detection_confidence'] for d in fused_detections]) if fused_detections else 0.0)
        }
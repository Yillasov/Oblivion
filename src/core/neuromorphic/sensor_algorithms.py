"""
Neuromorphic processing algorithms for different sensor types.

This module provides specialized neuromorphic processing algorithms
for each sensor type in the Oblivion system.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass, field

from src.simulation.sensors.sensor_framework import SensorType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


class ProcessingMode(Enum):
    """Processing modes for neuromorphic algorithms."""
    STANDARD = "standard"
    SPIKE_BASED = "spike_based"
    RATE_BASED = "rate_based"
    TEMPORAL_CODING = "temporal_coding"
    PHASE_CODING = "phase_coding"


@dataclass
class NeuromorphicAlgorithmConfig:
    """Configuration for neuromorphic processing algorithms."""
    mode: ProcessingMode = ProcessingMode.SPIKE_BASED
    time_steps: int = 100
    threshold: float = 0.5
    learning_enabled: bool = True
    noise_tolerance: float = 0.1
    adaptation_rate: float = 0.01
    temporal_window: int = 10  # Number of time steps to consider for temporal processing


class BaseSensorAlgorithm:
    """Base class for all neuromorphic sensor processing algorithms."""
    
    def __init__(self, config: NeuromorphicAlgorithmConfig = NeuromorphicAlgorithmConfig()):
        """Initialize the neuromorphic processing algorithm."""
        self.config = config
        self.time_step = 0
        self.neuron_states = {}
        self.spike_history = []
        self.initialized = False
        
    def initialize(self, input_shape: Tuple) -> None:
        """Initialize the algorithm with the input shape."""
        self.initialized = True
        
    def process(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data using neuromorphic computing."""
        # Base implementation just returns the data
        return sensor_data
    
    def encode_to_spikes(self, data: np.ndarray) -> np.ndarray:
        """Encode data to spikes based on threshold."""
        if self.config.mode == ProcessingMode.SPIKE_BASED:
            # Simple threshold-based encoding
            return (data > self.config.threshold).astype(np.int8)
        elif self.config.mode == ProcessingMode.RATE_BASED:
            # Rate-based encoding (higher values = more spikes)
            spike_rates = np.clip(data, 0, 1)
            spikes = np.random.random(data.shape) < spike_rates
            return spikes.astype(np.int8)
        elif self.config.mode == ProcessingMode.TEMPORAL_CODING:
            # Temporal coding (higher values fire earlier)
            max_time = self.config.time_steps
            firing_times = np.ceil((1.0 - np.clip(data, 0, 1)) * max_time)
            spikes = np.zeros_like(data, dtype=np.int8)
            spikes[firing_times <= self.time_step] = 1
            return spikes
        else:
            # Default to threshold-based
            return (data > self.config.threshold).astype(np.int8)
    
    def decode_from_spikes(self, spikes: List[np.ndarray]) -> np.ndarray:
        """Decode data from spike history."""
        if not spikes:
            return np.zeros((1,))
            
        if self.config.mode == ProcessingMode.SPIKE_BASED:
            # Simple decoding - just take the last spike pattern
            return spikes[-1].astype(float)
        elif self.config.mode == ProcessingMode.RATE_BASED:
            # Rate-based decoding - average spike rate over time window
            window = min(len(spikes), self.config.temporal_window)
            recent_spikes = spikes[-window:]
            return np.mean(recent_spikes, axis=0)
        elif self.config.mode == ProcessingMode.TEMPORAL_CODING:
            # Temporal decoding - earlier spikes = higher values
            window = min(len(spikes), self.config.temporal_window)
            if window == 0:
                return np.zeros_like(spikes[0], dtype=float)
                
            # For each position, find the earliest spike
            result = np.zeros_like(spikes[0], dtype=float)
            for t in range(window):
                # Positions that haven't spiked yet
                mask = (result == 0)
                # Set values based on when they first spike (earlier = higher value)
                result[mask & (spikes[-(t+1)] > 0)] = 1.0 - (t / window)
            return result
        else:
            # Default decoding
            return spikes[-1].astype(float)


class NeuromorphicVisionAlgorithm(BaseSensorAlgorithm):
    """Neuromorphic processing algorithm for vision sensors."""
    
    def initialize(self, input_shape: Tuple) -> None:
        """Initialize with vision sensor dimensions."""
        super().initialize(input_shape)
        # Create neuron grid matching input dimensions
        self.neuron_states = np.zeros(input_shape)
        # Initialize spike history
        self.spike_history = []
        
    def process(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process vision sensor data."""
        # Extract events from neuromorphic vision sensor
        events = sensor_data.get('events', [])
        
        # If no events, return empty result
        if not events:
            return {'processed_events': [], 'feature_map': np.zeros((10, 10))}
        
        # Convert events to a 2D grid
        width, height = sensor_data.get('resolution', (640, 480))
        event_grid = np.zeros((height, width))
        
        # Populate grid with event polarities
        for event in events:
            x, y = event.get('x', 0), event.get('y', 0)
            if 0 <= x < width and 0 <= y < height:
                event_grid[y, x] = event.get('polarity', 0)
        
        # Encode to spikes
        spikes = self.encode_to_spikes(event_grid)
        self.spike_history.append(spikes)
        
        # Keep history limited to temporal window
        if len(self.spike_history) > self.config.temporal_window:
            self.spike_history = self.spike_history[-self.config.temporal_window:]
        
        # Simple feature extraction (edge detection)
        if len(self.spike_history) > 1:
            # Temporal difference
            feature_map = np.abs(self.spike_history[-1] - self.spike_history[-2])
            # Downsample for efficiency
            feature_map = feature_map[::max(1, height//10), ::max(1, width//10)]
        else:
            feature_map = np.zeros((10, 10))
        
        # Increment time step
        self.time_step += 1
        
        return {
            'processed_events': self.spike_history[-1].tolist(),
            'feature_map': feature_map.tolist(),
            'active_regions': np.sum(feature_map > 0),
            'time_step': self.time_step
        }


class SyntheticApertureRadarAlgorithm(BaseSensorAlgorithm):
    """Neuromorphic processing algorithm for SAR sensors."""
    
    def initialize(self, input_shape: Tuple) -> None:
        """Initialize with SAR sensor parameters."""
        super().initialize(input_shape)
        # Create neuron array for SAR processing
        self.neuron_states = np.zeros(input_shape)
        
    def process(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process SAR sensor data."""
        # Extract SAR detections
        sar_detections = sensor_data.get('sar_detections', [])
        swath_data = sensor_data.get('current_swath', [])
        
        # If no detections, return empty result
        if not sar_detections:
            return {'processed_detections': [], 'detection_map': np.zeros((10, 10))}
        
        # Create a detection map (simplified 2D grid)
        detection_map = np.zeros((10, 10))
        
        # Process each detection
        processed_detections = []
        for detection in sar_detections:
            # Extract detection data
            confidence = detection.get('detection_confidence', 0.0)
            distance = detection.get('distance', 0.0)
            azimuth = detection.get('azimuth', 0.0)
            
            # Simple mapping to grid coordinates
            x = min(9, int(5 + 4 * np.sin(np.radians(azimuth))))
            y = min(9, int(9 - 9 * (distance / sensor_data.get('max_range', 1000.0))))
            
            # Update detection map with confidence
            if 0 <= x < 10 and 0 <= y < 10:
                detection_map[y, x] = max(detection_map[y, x], confidence)
            
            # Create processed detection with neuromorphic features
            processed_detection = detection.copy()
            processed_detection['spike_strength'] = self.encode_to_spikes(np.array([confidence]))[0]
            processed_detections.append(processed_detection)
        
        # Encode detection map to spikes
        spike_map = self.encode_to_spikes(detection_map)
        self.spike_history.append(spike_map)
        
        # Keep history limited
        if len(self.spike_history) > self.config.temporal_window:
            self.spike_history = self.spike_history[-self.config.temporal_window:]
        
        # Temporal integration of detections
        if len(self.spike_history) > 1:
            integrated_map = np.mean(self.spike_history, axis=0)
        else:
            integrated_map = spike_map
        
        # Increment time step
        self.time_step += 1
        
        return {
            'processed_detections': processed_detections,
            'detection_map': integrated_map.tolist(),
            'detection_count': len(processed_detections),
            'time_step': self.time_step
        }


class LidarAlgorithm(BaseSensorAlgorithm):
    """Neuromorphic processing algorithm for LiDAR sensors."""
    
    def process(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process LiDAR sensor data."""
        # Extract point cloud data
        point_cloud = sensor_data.get('point_cloud', [])
        
        # If no points, return empty result
        if not point_cloud:
            return {'processed_points': [], 'obstacle_map': np.zeros((10, 10))}
        
        # Create a simplified 2D obstacle map
        obstacle_map = np.zeros((10, 10))
        
        # Process each point
        for point in point_cloud:
            # Extract point data
            x, y, z = point.get('position', [0, 0, 0])
            intensity = point.get('intensity', 0.0)
            
            # Simple mapping to grid coordinates (top-down view)
            grid_x = min(9, int((x + 50) / 10))  # Assuming range of -50 to +50 meters
            grid_y = min(9, int((y + 50) / 10))
            
            # Update obstacle map with point intensity
            if 0 <= grid_x < 10 and 0 <= grid_y < 10:
                obstacle_map[grid_y, grid_x] = max(obstacle_map[grid_y, grid_x], intensity)
        
        # Encode obstacle map to spikes
        spike_map = self.encode_to_spikes(obstacle_map)
        self.spike_history.append(spike_map)
        
        # Keep history limited
        if len(self.spike_history) > self.config.temporal_window:
            self.spike_history = self.spike_history[-self.config.temporal_window:]
        
        # Simple obstacle detection
        obstacle_detected = np.max(obstacle_map) > self.config.threshold
        
        # Increment time step
        self.time_step += 1
        
        return {
            'obstacle_map': obstacle_map.tolist(),
            'spike_map': spike_map.tolist(),
            'obstacle_detected': obstacle_detected,
            'time_step': self.time_step
        }


class TerahertzAlgorithm(BaseSensorAlgorithm):
    """Neuromorphic processing algorithm for Terahertz sensors."""
    
    def process(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Terahertz sensor data."""
        # Extract spectral data
        spectral_data = sensor_data.get('spectral_data', [])
        material_detections = sensor_data.get('material_detections', [])
        
        # If no data, return empty result
        if not spectral_data and not material_detections:
            return {'processed_spectra': [], 'material_map': np.zeros((5, 5))}
        
        # Create a simplified material map
        material_map = np.zeros((5, 5))
        
        # Process material detections
        for detection in material_detections:
            material_type = detection.get('material_type', 0)
            confidence = detection.get('confidence', 0.0)
            position = detection.get('position', [0, 0, 0])
            
            # Simple mapping to grid coordinates
            grid_x = min(4, int((position[0] + 10) / 4))  # Assuming range of -10 to +10 meters
            grid_y = min(4, int((position[1] + 10) / 4))
            
            # Update material map with detection confidence
            if 0 <= grid_x < 5 and 0 <= grid_y < 5:
                material_map[grid_y, grid_x] = confidence
        
        # Encode material map to spikes
        spike_map = self.encode_to_spikes(material_map)
        self.spike_history.append(spike_map)
        
        # Keep history limited
        if len(self.spike_history) > self.config.temporal_window:
            self.spike_history = self.spike_history[-self.config.temporal_window:]
        
        # Process spectral data (simplified)
        processed_spectra = []
        for spectrum in spectral_data:
            # Encode spectrum to spikes
            spectrum_values = np.array(spectrum.get('values', []))
            if len(spectrum_values) > 0:
                spectrum_spikes = self.encode_to_spikes(spectrum_values / np.max(spectrum_values))
                processed_spectra.append({
                    'position': spectrum.get('position', [0, 0, 0]),
                    'spike_pattern': spectrum_spikes.tolist()
                })
        
        # Increment time step
        self.time_step += 1
        
        return {
            'processed_spectra': processed_spectra,
            'material_map': material_map.tolist(),
            'spike_map': spike_map.tolist(),
            'detection_count': len(material_detections),
            'time_step': self.time_step
        }


class QuantumRadarAlgorithm(BaseSensorAlgorithm):
    """Neuromorphic processing algorithm for Quantum Radar sensors."""
    
    def process(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Quantum Radar sensor data."""
        # Extract quantum radar detections
        quantum_detections = sensor_data.get('quantum_detections', [])
        entanglement_data = sensor_data.get('entanglement_data', {})
        
        # If no detections, return empty result
        if not quantum_detections:
            return {'processed_detections': [], 'entanglement_map': np.zeros((8, 8))}
        
        # Create a simplified entanglement map
        entanglement_map = np.zeros((8, 8))
        
        # Process each detection
        processed_detections = []
        for detection in quantum_detections:
            # Extract detection data
            confidence = detection.get('detection_confidence', 0.0)
            distance = detection.get('distance', 0.0)
            azimuth = detection.get('azimuth', 0.0)
            
            # Simple mapping to grid coordinates
            x = min(7, int(4 + 3 * np.sin(np.radians(azimuth))))
            y = min(7, int(7 - 7 * (distance / sensor_data.get('max_range', 1000.0))))
            
            # Update entanglement map with confidence
            if 0 <= x < 8 and 0 <= y < 8:
                entanglement_map[y, x] = max(entanglement_map[y, x], confidence)
            
            # Create processed detection with neuromorphic features
            processed_detection = detection.copy()
            processed_detection['spike_strength'] = self.encode_to_spikes(np.array([confidence]))[0]
            processed_detections.append(processed_detection)
        
        # Encode entanglement map to spikes
        spike_map = self.encode_to_spikes(entanglement_map)
        self.spike_history.append(spike_map)
        
        # Keep history limited
        if len(self.spike_history) > self.config.temporal_window:
            self.spike_history = self.spike_history[-self.config.temporal_window:]
        
        # Process entanglement data (simplified)
        entanglement_quality = entanglement_data.get('quality', 0.0)
        entanglement_duration = entanglement_data.get('duration', 0.0)
        
        # Increment time step
        self.time_step += 1
        
        return {
            'processed_detections': processed_detections,
            'entanglement_map': entanglement_map.tolist(),
            'spike_map': spike_map.tolist(),
            'entanglement_quality': entanglement_quality,
            'detection_count': len(processed_detections),
            'time_step': self.time_step
        }


class MultimodalFusionAlgorithm(BaseSensorAlgorithm):
    """Neuromorphic algorithm for fusing data from multiple sensors."""
    
    def __init__(self, config: NeuromorphicAlgorithmConfig = NeuromorphicAlgorithmConfig()):
        """Initialize the multimodal fusion algorithm."""
        super().__init__(config)
        self.sensor_weights = {}
        self.fusion_history = []
        
    def set_sensor_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for different sensor types in fusion."""
        self.sensor_weights = weights
        
    def process(self, sensor_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process and fuse data from multiple sensors."""
        # If no data, return empty result
        if not sensor_data:
            return {'fused_map': np.zeros((10, 10)), 'confidence': 0.0}
        
        # Create a fusion grid
        fusion_grid = np.zeros((10, 10))
        confidence_grid = np.zeros((10, 10))
        
        # Process each sensor's data
        for sensor_type, data in sensor_data.items():
            # Get weight for this sensor type
            weight = self.sensor_weights.get(sensor_type, 1.0)
            
            # Extract detection map based on sensor type
            detection_map = None
            if sensor_type == 'vision':
                # Vision sensors provide feature maps
                detection_map = np.array(data.get('feature_map', []))
            elif sensor_type == 'sar':
                # SAR provides detection maps
                detection_map = np.array(data.get('detection_map', []))
            elif sensor_type == 'lidar':
                # LiDAR provides obstacle maps
                detection_map = np.array(data.get('obstacle_map', []))
            elif sensor_type == 'terahertz':
                # Terahertz provides material maps
                detection_map = np.array(data.get('material_map', []))
            elif sensor_type == 'quantum':
                # Quantum radar provides entanglement maps
                detection_map = np.array(data.get('entanglement_map', []))
            
            # Skip if no detection map or wrong shape
            if detection_map is None or len(detection_map) == 0:
                continue
                
            # Resize to fusion grid size if needed
            if detection_map.shape != fusion_grid.shape:
                # Simple resize by repeating or sampling
                h, w = detection_map.shape
                h_ratio = fusion_grid.shape[0] / h
                w_ratio = fusion_grid.shape[1] / w
                
                resized_map = np.zeros(fusion_grid.shape)
                for i in range(fusion_grid.shape[0]):
                    for j in range(fusion_grid.shape[1]):
                        src_i = min(h-1, int(i / h_ratio))
                        src_j = min(w-1, int(j / w_ratio))
                        resized_map[i, j] = detection_map[src_i, src_j]
                
                detection_map = resized_map
            
            # Add to fusion grid with weight
            fusion_grid += weight * detection_map
            
            # Update confidence based on sensor reliability
            confidence = data.get('detection_confidence', 0.8)  # Default confidence
            confidence_grid += weight * (detection_map > 0) * confidence
        
        # Normalize fusion grid
        total_weight = sum(self.sensor_weights.values()) if self.sensor_weights else len(sensor_data)
        if total_weight > 0:
            fusion_grid /= total_weight
            confidence_grid /= total_weight
        
        # Encode fusion grid to spikes
        spike_grid = self.encode_to_spikes(fusion_grid)
        self.spike_history.append(spike_grid)
        
        # Keep history limited
        if len(self.spike_history) > self.config.temporal_window:
            self.spike_history = self.spike_history[-self.config.temporal_window:]
        
        # Temporal integration
        if len(self.spike_history) > 1:
            integrated_grid = np.mean(self.spike_history, axis=0)
        else:
            integrated_grid = spike_grid
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_grid[fusion_grid > 0]) if np.any(fusion_grid > 0) else 0.0
        
        # Store fusion result in history
        self.fusion_history.append({
            'grid': fusion_grid.copy(),
            'confidence': overall_confidence,
            'time_step': self.time_step
        })
        
        # Keep fusion history limited
        if len(self.fusion_history) > 10:
            self.fusion_history = self.fusion_history[-10:]
        
        # Increment time step
        self.time_step += 1
        
        return {
            'fused_map': fusion_grid.tolist(),
            'confidence_map': confidence_grid.tolist(),
            'spike_map': spike_grid.tolist(),
            'integrated_map': integrated_grid.tolist(),
            'overall_confidence': float(overall_confidence),
            'time_step': self.time_step
        }


# Factory function to create appropriate algorithm for sensor type
def create_algorithm_for_sensor(sensor_type: SensorType, 
                              config: NeuromorphicAlgorithmConfig = NeuromorphicAlgorithmConfig()) -> BaseSensorAlgorithm:
    """Create appropriate neuromorphic algorithm for the given sensor type."""
    if sensor_type == SensorType.NEUROMORPHIC_VISION:
        return NeuromorphicVisionAlgorithm(config)
    elif sensor_type == SensorType.SYNTHETIC_APERTURE_RADAR:
        return SyntheticApertureRadarAlgorithm(config)
    elif sensor_type == SensorType.LIDAR:
        return LidarAlgorithm(config)
    elif sensor_type == SensorType.TERAHERTZ:
        return TerahertzAlgorithm(config)
    elif sensor_type == SensorType.QUANTUM_RADAR:
        return QuantumRadarAlgorithm(config)
    else:
        # Default algorithm
        return BaseSensorAlgorithm(config)


# Create a fusion algorithm with default weights
def create_fusion_algorithm(config: NeuromorphicAlgorithmConfig = NeuromorphicAlgorithmConfig()) -> MultimodalFusionAlgorithm:
    """Create a multimodal fusion algorithm with default weights."""
    fusion_algo = MultimodalFusionAlgorithm(config)
    
    # Set default weights for different sensor types
    fusion_algo.set_sensor_weights({
        SensorType.NEUROMORPHIC_VISION.name: 1.0,
        SensorType.SYNTHETIC_APERTURE_RADAR.name: 1.2,
        SensorType.LIDAR.name: 1.0,
        SensorType.TERAHERTZ.name: 0.8,
        SensorType.QUANTUM_RADAR.name: 1.5
    })
    
    return fusion_algo
#!/usr/bin/env python3
"""
Sensor Fusion Algorithm

Implements Extended Kalman Filter (EKF) based sensor fusion for robust
state estimation by combining multiple sensor inputs.
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

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from src.core.utils.logging_framework import get_logger

logger = get_logger("sensor_fusion")


class SensorPriority(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class SensorWeight:
    """Sensor weighting configuration."""
    
    base_weight: float
    priority: SensorPriority
    reliability: float = 1.0
    
    def compute_weight(self) -> float:
        """Compute effective sensor weight."""
        return self.base_weight * self.priority.value * self.reliability


class ExtendedKalmanFilter:
    """EKF implementation for sensor fusion."""
    
    def __init__(self, state_dim: int = 12):
        """Initialize EKF."""
        self.state_dim = state_dim
        
        # State vector: [position(3), velocity(3), attitude(3), angular_rates(3)]
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)
        
        # Process noise
        self.Q = np.eye(state_dim) * 0.1
        
        # Measurement noise (updated dynamically)
        self.R = np.eye(state_dim)
        
        logger.info("Initialized Extended Kalman Filter")
    
    def predict(self, dt: float) -> None:
        """Prediction step."""
        # State transition matrix
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * dt  # Position update from velocity
        F[6:9, 9:12] = np.eye(3) * dt  # Attitude update from angular rates
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q * dt
    
    def update(self, measurement: np.ndarray, 
               H: np.ndarray, R: np.ndarray) -> None:
        """Update step."""
        # Innovation
        y = measurement - H @ self.state
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance


class SensorFusion:
    """Main sensor fusion implementation."""
    
    def __init__(self):
        """Initialize sensor fusion system."""
        self.ekf = ExtendedKalmanFilter()
        
        # Sensor weights configuration
        self.sensor_weights = {
            "radar": SensorWeight(0.8, SensorPriority.HIGH),
            "gps": SensorWeight(0.7, SensorPriority.MEDIUM),
            "imu": SensorWeight(0.9, SensorPriority.HIGH),
            "altimeter": SensorWeight(0.8, SensorPriority.MEDIUM)
        }
        
        # Sensor health monitoring
        self.sensor_health = {}
        self.health_threshold = 0.5
        
        logger.info("Initialized sensor fusion system")
    
    def _update_sensor_health(self, sensor_data: Dict[str, Dict[str, Any]]) -> None:
        """Update sensor health metrics."""
        for sensor_name, data in sensor_data.items():
            if sensor_name not in self.sensor_health:
                self.sensor_health[sensor_name] = 1.0
            
            # Check for sensor failure
            if not data:
                self.sensor_health[sensor_name] *= 0.5
                continue
            
            # Check data validity
            signal_strength = data.get('signal_strength', 1.0)
            self.sensor_health[sensor_name] = min(1.0, 
                self.sensor_health[sensor_name] * 1.1 * signal_strength)
            
            # Update sensor reliability
            if sensor_name in self.sensor_weights:
                self.sensor_weights[sensor_name].reliability = self.sensor_health[sensor_name]
    
    def _prepare_measurement(self, sensor_data: Dict[str, Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare measurement data for EKF update."""
        measurement = np.zeros(self.ekf.state_dim)
        H = np.zeros((self.ekf.state_dim, self.ekf.state_dim))
        R = np.eye(self.ekf.state_dim)
        
        measurement_count = np.zeros(self.ekf.state_dim)
        
        for sensor_name, data in sensor_data.items():
            if self.sensor_health[sensor_name] < self.health_threshold:
                continue
            
            weight = self.sensor_weights[sensor_name].compute_weight()
            
            if sensor_name == "radar":
                if 'targets' in data and data['targets']:
                    target = data['targets'][0]  # Consider closest target
                    measurement[0:3] += weight * np.array([
                        target['distance'],
                        target['azimuth'],
                        target['elevation']
                    ])
                    measurement_count[0:3] += weight
                    H[0:3, 0:3] = np.eye(3)
                    R[0:3, 0:3] *= 1.0 / weight
            
            elif sensor_name == "gps":
                if 'position' in data:
                    measurement[0:3] += weight * data['position']
                    measurement_count[0:3] += weight
                    H[0:3, 0:3] = np.eye(3)
                    R[0:3, 0:3] *= 1.0 / weight
            
            elif sensor_name == "imu":
                if 'angular_rates' in data:
                    measurement[9:12] += weight * data['angular_rates']
                    measurement_count[9:12] += weight
                    H[9:12, 9:12] = np.eye(3)
                    R[9:12, 9:12] *= 1.0 / weight
            
            elif sensor_name == "altimeter":
                if 'altitude_msl' in data:
                    measurement[2] += weight * data['altitude_msl']
                    measurement_count[2] += weight
                    H[2, 2] = 1.0
                    R[2, 2] *= 1.0 / weight
        
        # Normalize measurements
        valid_measurements = measurement_count > 0
        measurement[valid_measurements] /= measurement_count[valid_measurements]
        
        return measurement, H, R
    
    def update(self, sensor_data: Dict[str, Dict[str, Any]], 
               dt: float) -> np.ndarray:
        """Update state estimation using sensor fusion."""
        # Update sensor health
        self._update_sensor_health(sensor_data)
        
        # Prediction step
        self.ekf.predict(dt)
        
        # Prepare measurement data
        measurement, H, R = self._prepare_measurement(sensor_data)
        
        # Update step
        self.ekf.update(measurement, H, R)
        
        # Log fusion status
        logger.debug(f"Sensor health: {self.sensor_health}")
        logger.debug(f"State estimate: {self.ekf.state}")
        
        return self.ekf.state
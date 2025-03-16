"""
UKF-based Sensor Fusion Algorithm

Implements Unscented Kalman Filter for robust state estimation with
improved handling of non-linear dynamics.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

from src.core.utils.logging_framework import get_logger

logger = get_logger("ukf_sensor_fusion")


@dataclass
class UKFParameters:
    """UKF tuning parameters."""
    
    alpha: float = 0.3    # Spread of sigma points
    beta: float = 2.0     # Prior knowledge of state distribution
    kappa: float = 0.1    # Secondary scaling parameter
    
    def compute_weights(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute UKF weights."""
        lambda_param = self.alpha**2 * (n + self.kappa) - n
        
        # Weights for mean
        weights_m = np.full(2*n + 1, 1.0/(2*(n + lambda_param)))
        weights_m[0] = lambda_param/(n + lambda_param)
        
        # Weights for covariance
        weights_c = weights_m.copy()
        weights_c[0] += (1 - self.alpha**2 + self.beta)
        
        return weights_m, weights_c


class UnscentedKalmanFilter:
    """UKF implementation for sensor fusion."""
    
    def __init__(self, state_dim: int = 12):
        """Initialize UKF."""
        self.state_dim = state_dim
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)
        
        # UKF parameters
        self.params = UKFParameters()
        self.weights_m, self.weights_c = self.params.compute_weights(state_dim)
        
        # Process and measurement noise
        self.Q = np.eye(state_dim) * 0.1
        self.R = np.eye(state_dim)
        
        logger.info("Initialized Unscented Kalman Filter")
    
    def generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points using unscented transform."""
        n = self.state_dim
        lambda_param = self.params.alpha**2 * (n + self.params.kappa) - n
        
        # Matrix square root using Cholesky decomposition
        L = np.linalg.cholesky((n + lambda_param) * self.covariance)
        
        # Generate sigma points
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = self.state
        
        for i in range(n):
            sigma_points[i+1] = self.state + L[i]
            sigma_points[i+1+n] = self.state - L[i]
        
        return sigma_points
    
    def predict_dynamics(self, sigma_point: np.ndarray, dt: float) -> np.ndarray:
        """Non-linear state prediction."""
        # Extract states
        pos = sigma_point[0:3]
        vel = sigma_point[3:6]
        att = sigma_point[6:9]
        rates = sigma_point[9:12]
        
        # Simple non-linear dynamics
        pos_new = pos + vel * dt
        vel_new = vel + np.cross(rates, vel) * dt
        att_new = att + rates * dt
        rates_new = rates
        
        return np.concatenate([pos_new, vel_new, att_new, rates_new])
    
    def predict(self, dt: float) -> None:
        """Prediction step using unscented transform."""
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        
        # Propagate sigma points
        predicted_sigma_points = np.array([
            self.predict_dynamics(sigma, dt) for sigma in sigma_points
        ])
        
        # Compute predicted mean
        self.state = np.sum(self.weights_m.reshape(-1, 1) * predicted_sigma_points, axis=0)
        
        # Compute predicted covariance
        self.covariance = np.zeros_like(self.covariance)
        for i, sigma in enumerate(predicted_sigma_points):
            diff = (sigma - self.state).reshape(-1, 1)
            self.covariance += self.weights_c[i] * diff @ diff.T
        
        self.covariance += self.Q * dt
    
    def update(self, measurement: np.ndarray, 
               H: np.ndarray, R: np.ndarray) -> None:
        """Update step using unscented transform."""
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        
        # Predict measurements
        predicted_measurements = H @ sigma_points.T
        
        # Mean predicted measurement
        y_pred = np.sum(self.weights_m.reshape(-1, 1) * predicted_measurements.T, axis=0)
        
        # Innovation covariance
        Pyy = np.zeros((self.state_dim, self.state_dim))
        Pxy = np.zeros((self.state_dim, self.state_dim))
        
        for i in range(len(sigma_points)):
            diff_y = (predicted_measurements.T[i] - y_pred).reshape(-1, 1)
            diff_x = (sigma_points[i] - self.state).reshape(-1, 1)
            Pyy += self.weights_c[i] * diff_y @ diff_y.T
            Pxy += self.weights_c[i] * diff_x @ diff_y.T
        
        Pyy += R
        
        # Kalman gain
        K = Pxy @ np.linalg.inv(Pyy)
        
        # Update state and covariance
        self.state = self.state + K @ (measurement - y_pred)
        self.covariance = self.covariance - K @ Pyy @ K.T


class UKFSensorFusion:
    """UKF-based sensor fusion implementation."""
    
    def __init__(self):
        """Initialize UKF sensor fusion."""
        self.ukf = UnscentedKalmanFilter()
        self.sensor_weights = {
            "radar": 0.8,
            "gps": 0.7,
            "imu": 0.9,
            "altimeter": 0.8
        }
        self.sensor_health = {}
        
        logger.info("Initialized UKF sensor fusion")
    
    def _process_measurements(self, sensor_data: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process and combine sensor measurements."""
        measurement = np.zeros(self.ukf.state_dim)
        H = np.zeros((self.ukf.state_dim, self.ukf.state_dim))
        R = np.eye(self.ukf.state_dim)
        
        for sensor_name, data in sensor_data.items():
            if not data:
                continue
                
            weight = self.sensor_weights[sensor_name]
            
            if sensor_name == "imu":
                if 'angular_rates' in data:
                    measurement[9:12] = data['angular_rates']
                    H[9:12, 9:12] = np.eye(3)
                    R[9:12, 9:12] *= 1.0 / weight
                if 'attitude' in data:
                    measurement[6:9] = data['attitude']
                    H[6:9, 6:9] = np.eye(3)
                    R[6:9, 6:9] *= 1.0 / weight
            
            elif sensor_name == "gps":
                if 'position' in data and 'velocity' in data:
                    measurement[0:3] = data['position']
                    measurement[3:6] = data['velocity']
                    H[0:6, 0:6] = np.eye(6)
                    R[0:6, 0:6] *= 1.0 / weight
        
        return measurement, H, R
    
    def update(self, sensor_data: Dict[str, Dict[str, Any]], 
               dt: float) -> np.ndarray:
        """Update state estimation."""
        # Prediction
        self.ukf.predict(dt)
        
        # Process measurements
        measurement, H, R = self._process_measurements(sensor_data)
        
        # Update
        self.ukf.update(measurement, H, R)
        
        logger.debug(f"State estimate: {self.ukf.state}")
        return self.ukf.state
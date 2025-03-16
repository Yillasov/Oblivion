"""
Particle Filter Sensor Fusion Algorithm

Implements a particle filter for robust state estimation in highly non-linear
systems with non-Gaussian noise distributions.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

from src.core.utils.logging_framework import get_logger

logger = get_logger("particle_filter_fusion")


@dataclass
class ParticleFilterConfig:
    """Configuration parameters for particle filter."""
    
    num_particles: int = 1000
    resample_threshold: float = 0.5
    process_noise: float = 0.1
    measurement_noise: float = 0.1


class ParticleFilter:
    """Particle filter implementation for state estimation."""
    
    def __init__(self, state_dim: int = 12):
        """Initialize particle filter."""
        self.config = ParticleFilterConfig()
        self.state_dim = state_dim
        
        # Initialize particles and weights
        self.particles = np.zeros((self.config.num_particles, state_dim))
        self.weights = np.ones(self.config.num_particles) / self.config.num_particles
        
        # State estimate
        self.state_estimate = np.zeros(state_dim)
        self.state_covariance = np.eye(state_dim)
        
        logger.info("Initialized Particle Filter")
    
    def predict(self, dt: float) -> None:
        """Predict step: propagate particles through system dynamics."""
        for i in range(self.config.num_particles):
            # Extract states
            pos = self.particles[i, 0:3]
            vel = self.particles[i, 3:6]
            att = self.particles[i, 6:9]
            rates = self.particles[i, 9:12]
            
            # Non-linear dynamics
            pos_new = pos + vel * dt
            vel_new = vel + np.cross(rates, vel) * dt
            att_new = att + rates * dt
            rates_new = rates
            
            # Add process noise
            noise = np.random.normal(0, self.config.process_noise, self.state_dim)
            self.particles[i] = np.concatenate([pos_new, vel_new, att_new, rates_new]) + noise
    
    def update_weights(self, measurement: np.ndarray, 
                      noise_cov: np.ndarray) -> None:
        """Update particle weights based on measurements."""
        for i in range(self.config.num_particles):
            # Compute likelihood
            error = measurement - self.particles[i]
            likelihood = np.exp(-0.5 * error.T @ np.linalg.inv(noise_cov) @ error)
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
    
    def estimate_state(self) -> None:
        """Compute state estimate from particles."""
        self.state_estimate = np.average(self.particles, weights=self.weights, axis=0)
        
        # Compute weighted covariance
        self.state_covariance = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.config.num_particles):
            diff = (self.particles[i] - self.state_estimate).reshape(-1, 1)
            self.state_covariance += self.weights[i] * diff @ diff.T
    
    def resample(self) -> None:
        """Resample particles based on their weights."""
        effective_particles = 1.0 / np.sum(self.weights**2)
        
        if effective_particles < self.config.num_particles * self.config.resample_threshold:
            # Systematic resampling
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            
            # Generate random starting point
            positions = (np.random.random() + np.arange(self.config.num_particles)) / self.config.num_particles
            
            # Resample particles
            new_particles = np.zeros_like(self.particles)
            i, j = 0, 0
            while i < self.config.num_particles:
                if positions[i] < cumsum[j]:
                    new_particles[i] = self.particles[j]
                    i += 1
                else:
                    j += 1
            
            self.particles = new_particles
            self.weights.fill(1.0 / self.config.num_particles)


class ParticleFilterFusion:
    """Particle filter-based sensor fusion implementation."""
    
    def __init__(self):
        """Initialize particle filter fusion."""
        self.particle_filter = ParticleFilter()
        
        # Sensor configuration
        self.sensor_noise = {
            "gps": np.diag([5.0, 5.0, 10.0, 0.1, 0.1, 0.1]),
            "imu": np.diag([0.01, 0.01, 0.01, 0.05, 0.05, 0.05]),
            "altimeter": np.array([1.0])
        }
        
        logger.info("Initialized particle filter fusion")
    
    def process_sensor_data(self, sensor_data: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Process and combine sensor measurements."""
        measurement = np.zeros(self.particle_filter.state_dim)
        noise_cov = np.eye(self.particle_filter.state_dim)
        
        for sensor_name, data in sensor_data.items():
            if not data:
                continue
            
            if sensor_name == "gps" and 'position' in data and 'velocity' in data:
                measurement[0:3] = data['position']
                measurement[3:6] = data['velocity']
                noise_cov[0:6, 0:6] = self.sensor_noise["gps"]
            
            elif sensor_name == "imu" and 'attitude' in data and 'angular_rates' in data:
                measurement[6:9] = data['attitude']
                measurement[9:12] = data['angular_rates']
                noise_cov[6:12, 6:12] = self.sensor_noise["imu"]
        
        return measurement, noise_cov
    
    def update(self, sensor_data: Dict[str, Dict[str, Any]], 
               dt: float) -> np.ndarray:
        """Update state estimation using particle filter."""
        # Prediction step
        self.particle_filter.predict(dt)
        
        # Process measurements
        measurement, noise_cov = self.process_sensor_data(sensor_data)
        
        # Update weights
        self.particle_filter.update_weights(measurement, noise_cov)
        
        # Resample if needed
        self.particle_filter.resample()
        
        # Compute state estimate
        self.particle_filter.estimate_state()
        
        logger.debug(f"State estimate: {self.particle_filter.state_estimate}")
        return self.particle_filter.state_estimate
"""
Attitude Determination module for UCAV platforms.

Provides advanced algorithms for determining spacecraft attitude
using star tracker measurements and sensor fusion techniques.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

from src.navigation.star_tracker import StarTracker
from src.navigation.error_handling import safe_navigation_operation, UncertaintyQuantifier

# Configure logger
logger = logging.getLogger(__name__)


class AttitudeMethod(Enum):
    """Attitude determination methods."""
    TRIAD = "triad"
    QUEST = "quest"
    OLAE = "olae"  # Optimal Linear Attitude Estimator
    EKF = "extended_kalman_filter"


@dataclass
class AttitudeConfig:
    """Configuration for attitude determination."""
    method: AttitudeMethod = AttitudeMethod.QUEST
    use_gyro: bool = True
    fusion_weight: float = 0.7  # Weight for star tracker vs gyro
    smoothing_factor: float = 0.3
    max_uncertainty: float = 0.05  # radians


class AttitudeDetermination:
    """
    Attitude determination system using star tracker data.
    
    Implements multiple algorithms for determining spacecraft attitude
    from star measurements with optional sensor fusion.
    """
    
    def __init__(self, 
                star_tracker: StarTracker,
                config: AttitudeConfig = AttitudeConfig(),
                gyro_interface = None):
        """
        Initialize attitude determination system.
        
        Args:
            star_tracker: Star tracker system
            config: Attitude determination configuration
            gyro_interface: Optional gyroscope interface
        """
        self.star_tracker = star_tracker
        self.config = config
        self.gyro_interface = gyro_interface
        
        # Current attitude state
        self.quaternion = [0.0, 0.0, 0.0, 1.0]  # w, x, y, z
        self.euler_angles = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.angular_velocity = [0.0, 0.0, 0.0]
        
        # Uncertainty estimation
        self.uncertainty = UncertaintyQuantifier()
        self.attitude_covariance = np.eye(3) * 0.01
        
        # Previous measurements for filtering
        self.prev_quaternion = self.quaternion.copy()
        
        logger.info(f"Initialized attitude determination with {config.method.value} method")
    
    @safe_navigation_operation
    def update(self, delta_time: float) -> Dict[str, Any]:
        """
        Update attitude estimate.
        
        Args:
            delta_time: Time since last update in seconds
            
        Returns:
            Current attitude state
        """
        # Get star tracker quaternion
        star_quaternion = self.star_tracker.get_quaternion()
        
        # Get gyro data if available
        gyro_data = None
        if self.config.use_gyro and self.gyro_interface:
            gyro_data = self.gyro_interface.get_angular_velocity()
            self.angular_velocity = gyro_data
        
        # Determine attitude based on selected method
        if self.config.method == AttitudeMethod.TRIAD:
            self._triad_method(star_quaternion)
        elif self.config.method == AttitudeMethod.QUEST:
            self._quest_method(star_quaternion)
        elif self.config.method == AttitudeMethod.OLAE:
            self._olae_method(star_quaternion)
        elif self.config.method == AttitudeMethod.EKF:
            self._ekf_update(star_quaternion, gyro_data, delta_time)
        
        # Apply smoothing
        self._apply_smoothing()
        
        # Update Euler angles
        self.euler_angles = self._quaternion_to_euler()
        
        # Store current quaternion for next update
        self.prev_quaternion = self.quaternion.copy()
        
        # Return current attitude state
        return {
            "quaternion": self.quaternion,
            "euler_angles": self.euler_angles,
            "angular_velocity": self.angular_velocity,
            "uncertainty": self.attitude_covariance.diagonal().tolist(),
            "method": self.config.method.value
        }
    
    def _triad_method(self, star_quaternion: List[float]) -> None:
        """
        TRIAD method for attitude determination.
        
        Args:
            star_quaternion: Quaternion from star tracker
        """
        # Simple implementation - in a real system this would use vector observations
        self.quaternion = star_quaternion
        self.attitude_covariance = np.eye(3) * 0.02
    
    def _quest_method(self, star_quaternion: List[float]) -> None:
        """
        QUEST (QUaternion ESTimator) method.
        
        Args:
            star_quaternion: Quaternion from star tracker
        """
        # Simple implementation - in a real system this would use multiple vector observations
        self.quaternion = star_quaternion
        self.attitude_covariance = np.eye(3) * 0.015
    
    def _olae_method(self, star_quaternion: List[float]) -> None:
        """
        Optimal Linear Attitude Estimator method.
        
        Args:
            star_quaternion: Quaternion from star tracker
        """
        # Simple implementation - in a real system this would use multiple vector observations
        self.quaternion = star_quaternion
        self.attitude_covariance = np.eye(3) * 0.01
    
    def _ekf_update(self, star_quaternion: List[float], 
                   gyro_data: Optional[List[float]], 
                   delta_time: float) -> None:
        """
        Extended Kalman Filter update for attitude.
        
        Args:
            star_quaternion: Quaternion from star tracker
            gyro_data: Angular velocity from gyroscope
            delta_time: Time since last update
        """
        # Simple implementation - in a real system this would be a full EKF
        if gyro_data is not None:
            # Propagate quaternion with gyro data
            w, x, y, z = self.quaternion
            p, q, r = gyro_data
            
            # Simplified quaternion integration
            q_dot = [
                -0.5 * (x * p + y * q + z * r),
                0.5 * (w * p + z * q - y * r),
                0.5 * (w * q - z * p + x * r),
                0.5 * (w * r + y * p - x * q)
            ]
            
            # Update quaternion
            self.quaternion = [
                w + q_dot[0] * delta_time,
                x + q_dot[1] * delta_time,
                y + q_dot[2] * delta_time,
                z + q_dot[3] * delta_time
            ]
            
            # Normalize quaternion
            norm = np.sqrt(sum(q**2 for q in self.quaternion))
            self.quaternion = [q/norm for q in self.quaternion]
            
            # Fuse with star tracker measurement
            w_st = self.config.fusion_weight
            w_gyro = 1.0 - w_st
            
            self.quaternion = [
                w_gyro * self.quaternion[0] + w_st * star_quaternion[0],
                w_gyro * self.quaternion[1] + w_st * star_quaternion[1],
                w_gyro * self.quaternion[2] + w_st * star_quaternion[2],
                w_gyro * self.quaternion[3] + w_st * star_quaternion[3]
            ]
            
            # Normalize again
            norm = np.sqrt(sum(q**2 for q in self.quaternion))
            self.quaternion = [q/norm for q in self.quaternion]
            
            # Update covariance (simplified)
            self.attitude_covariance = np.eye(3) * 0.005
        else:
            # No gyro data, use star tracker only
            self.quaternion = star_quaternion
            self.attitude_covariance = np.eye(3) * 0.02
    
    def _apply_smoothing(self) -> None:
        """Apply smoothing to quaternion."""
        alpha = self.config.smoothing_factor
        
        # Interpolate between previous and current quaternion
        self.quaternion = [
            (1-alpha) * self.prev_quaternion[0] + alpha * self.quaternion[0],
            (1-alpha) * self.prev_quaternion[1] + alpha * self.quaternion[1],
            (1-alpha) * self.prev_quaternion[2] + alpha * self.quaternion[2],
            (1-alpha) * self.prev_quaternion[3] + alpha * self.quaternion[3]
        ]
        
        # Normalize quaternion
        norm = np.sqrt(sum(q**2 for q in self.quaternion))
        self.quaternion = [q/norm for q in self.quaternion]
    
    def _quaternion_to_euler(self) -> Dict[str, float]:
        """
        Convert quaternion to Euler angles.
        
        Returns:
            Dictionary with roll, pitch, yaw in radians
        """
        w, x, y, z = self.quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return {"roll": roll, "pitch": pitch, "yaw": yaw}
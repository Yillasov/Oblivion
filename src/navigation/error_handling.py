"""
Error handling and uncertainty quantification for navigation systems.

This module provides specialized error handling and uncertainty quantification
for navigation systems in UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Type, Tuple

from src.core.utils.error_handling import (
    OblivionError, ErrorHandler, ErrorContext, ErrorSeverity, 
    handle_errors, global_error_handler
)

# Configure logger
logger = logging.getLogger(__name__)


class NavigationError(OblivionError):
    """Base class for navigation-specific errors."""
    pass


class SensorDataError(NavigationError):
    """Error related to sensor data processing."""
    pass


class PositionEstimationError(NavigationError):
    """Error during position estimation."""
    pass


class OrientationEstimationError(NavigationError):
    """Error during orientation estimation."""
    pass


class NavigationSystemFailure(NavigationError):
    """Complete failure of a navigation system."""
    pass


class NavigationErrorHandler(ErrorHandler):
    """Specialized error handler for navigation systems."""
    
    def __init__(self):
        """Initialize the navigation error handler."""
        super().__init__()
        self.register_default_strategies()
        
    def register_default_strategies(self):
        """Register default recovery strategies for navigation errors."""
        self.register_recovery_strategy(
            SensorDataError, 
            lambda error: self._recover_from_sensor_data_error(error) if isinstance(error, SensorDataError) else False
        )
        self.register_recovery_strategy(
            PositionEstimationError,
            lambda error: self._recover_from_position_error(error) if isinstance(error, PositionEstimationError) else False
        )
        self.register_recovery_strategy(
            OrientationEstimationError,
            lambda error: self._recover_from_orientation_error(error) if isinstance(error, OrientationEstimationError) else False
        )
    
    def _recover_from_sensor_data_error(self, error: SensorDataError) -> bool:
        """Attempt to recover from sensor data errors."""
        logger.info(f"Attempting to recover from sensor data error: {error.message}")
        # Recovery logic: use last valid data or alternative sensor
        return True
    
    def _recover_from_position_error(self, error: PositionEstimationError) -> bool:
        """Attempt to recover from position estimation errors."""
        logger.info(f"Attempting to recover from position error: {error.message}")
        # Recovery logic: switch to backup position system
        return True
    
    def _recover_from_orientation_error(self, error: OrientationEstimationError) -> bool:
        """Attempt to recover from orientation estimation errors."""
        logger.info(f"Attempting to recover from orientation error: {error.message}")
        # Recovery logic: use alternative orientation source
        return True


# Create a global navigation error handler
navigation_error_handler = NavigationErrorHandler()


class UncertaintyQuantifier:
    """
    Quantifies uncertainty in navigation measurements.
    
    This class provides methods to estimate uncertainty in position,
    orientation, and velocity measurements.
    """
    
    def __init__(self):
        """Initialize the uncertainty quantifier."""
        self.position_uncertainty = np.zeros(3)  # x, y, z uncertainty
        self.orientation_uncertainty = np.zeros(3)  # roll, pitch, yaw uncertainty
        self.velocity_uncertainty = np.zeros(3)  # vx, vy, vz uncertainty
        self.sensor_noise_models = {}
        self.confidence_threshold = 0.7
    
    def estimate_position_uncertainty(self, 
                                     measurements: List[Dict[str, float]], 
                                     weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Estimate uncertainty in position measurements.
        
        Args:
            measurements: List of position measurements from different sources
            weights: Optional weights for each measurement source
            
        Returns:
            Numpy array of position uncertainties [x, y, z]
        """
        if not measurements:
            return np.ones(3) * float('inf')
            
        if weights is None:
            weights = [1.0] * len(measurements)
            
        # Extract x, y, z values
        x_vals = [m.get('x', 0.0) for m in measurements]
        y_vals = [m.get('y', 0.0) for m in measurements]
        z_vals = [m.get('z', 0.0) for m in measurements]
        
        # Calculate weighted standard deviation
        x_uncertainty = self._weighted_std(x_vals, weights)
        y_uncertainty = self._weighted_std(y_vals, weights)
        z_uncertainty = self._weighted_std(z_vals, weights)
        
        self.position_uncertainty = np.array([x_uncertainty, y_uncertainty, z_uncertainty])
        return self.position_uncertainty
    
    def estimate_orientation_uncertainty(self, 
                                        measurements: List[Dict[str, float]], 
                                        weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Estimate uncertainty in orientation measurements.
        
        Args:
            measurements: List of orientation measurements from different sources
            weights: Optional weights for each measurement source
            
        Returns:
            Numpy array of orientation uncertainties [roll, pitch, yaw]
        """
        if not measurements:
            return np.ones(3) * float('inf')
            
        if weights is None:
            weights = [1.0] * len(measurements)
            
        # Extract roll, pitch, yaw values
        roll_vals = [m.get('roll', 0.0) for m in measurements]
        pitch_vals = [m.get('pitch', 0.0) for m in measurements]
        yaw_vals = [m.get('yaw', 0.0) for m in measurements]
        
        # Calculate weighted standard deviation
        roll_uncertainty = self._weighted_std(roll_vals, weights)
        pitch_uncertainty = self._weighted_std(pitch_vals, weights)
        yaw_uncertainty = self._weighted_std(yaw_vals, weights)
        
        self.orientation_uncertainty = np.array([roll_uncertainty, pitch_uncertainty, yaw_uncertainty])
        return self.orientation_uncertainty
    
    def calculate_confidence(self, uncertainties: np.ndarray, 
                           thresholds: np.ndarray) -> float:
        """
        Calculate confidence level based on uncertainties.
        
        Args:
            uncertainties: Array of uncertainty values
            thresholds: Array of threshold values for each uncertainty
            
        Returns:
            Confidence level between 0 and 1
        """
        # Calculate normalized confidence for each dimension
        confidence_values = np.exp(-uncertainties / thresholds)
        
        # Overall confidence is the product of individual confidences
        overall_confidence = np.prod(confidence_values)
        
        return float(overall_confidence)
    
    def _weighted_std(self, values: List[float], weights: List[float]) -> float:
        """
        Calculate weighted standard deviation.
        
        Args:
            values: List of values
            weights: List of weights
            
        Returns:
            Weighted standard deviation
        """
        if not values:
            return float('inf')
            
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Calculate weighted mean
        weighted_mean = np.sum(np.array(values) * weights)
        
        # Calculate weighted variance
        variance = np.sum(weights * (np.array(values) - weighted_mean)**2)
        
        return np.sqrt(variance)


@handle_errors(error_handler=navigation_error_handler, 
              context={"module": "navigation"})
def safe_navigation_operation(func: Callable) -> Callable:
    """
    Decorator for safe navigation operations with error handling.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Navigation operation failed: {str(e)}")
            raise NavigationError(
                message=f"Navigation operation failed: {str(e)}",
                severity=ErrorSeverity.ERROR
            )
    return wrapper
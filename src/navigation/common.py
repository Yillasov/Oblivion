"""
Common navigation interfaces for UCAV platforms.

This module provides standardized interfaces for accessing navigation data
across different navigation systems.
"""

from typing import Dict, Protocol, Any, Optional
import numpy as np


class PositionProvider(Protocol):
    """Interface for systems that provide position data."""
    
    def get_position(self) -> Dict[str, float]:
        """
        Get current position data.
        
        Returns:
            Dictionary with x, y, z coordinates in meters
        """
        ...
    
    def get_position_accuracy(self) -> float:
        """
        Get current position accuracy in meters.
        
        Returns:
            Position accuracy (lower is better)
        """
        ...


class OrientationProvider(Protocol):
    """Interface for systems that provide orientation data."""
    
    def get_orientation(self) -> Dict[str, float]:
        """
        Get current orientation data.
        
        Returns:
            Dictionary with roll, pitch, yaw in radians
        """
        ...
    
    def get_orientation_accuracy(self) -> float:
        """
        Get current orientation accuracy in radians.
        
        Returns:
            Orientation accuracy (lower is better)
        """
        ...


class VelocityProvider(Protocol):
    """Interface for systems that provide velocity data."""
    
    def get_velocity(self) -> Dict[str, float]:
        """
        Get current velocity data.
        
        Returns:
            Dictionary with vx, vy, vz in meters per second
        """
        ...
    
    def get_velocity_accuracy(self) -> float:
        """
        Get current velocity accuracy in meters per second.
        
        Returns:
            Velocity accuracy (lower is better)
        """
        ...


class NavigationDataProvider(PositionProvider, OrientationProvider, VelocityProvider, Protocol):
    """Combined interface for systems that provide complete navigation data."""
    
    def get_navigation_state(self) -> Dict[str, Any]:
        """
        Get complete navigation state.
        
        Returns:
            Dictionary with position, orientation, velocity and metadata
        """
        ...
"""
Navigation interfaces for UCAV platforms.

This module provides interfaces for connecting navigation systems
with other UCAV subsystems.
"""

import logging
from typing import Dict, List, Any, Optional, Protocol, Tuple
from dataclasses import dataclass
import numpy as np

from src.navigation.base import NavigationSpecs, NavigationType

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class NavigationState:
    """Current navigation state data."""
    position: Dict[str, float]  # x, y, z coordinates
    orientation: Dict[str, float]  # roll, pitch, yaw
    velocity: Dict[str, float]  # vx, vy, vz
    acceleration: Dict[str, float]  # ax, ay, az
    timestamp: float  # timestamp of the state
    confidence: float  # confidence level (0-1)
    system_id: str  # ID of the navigation system providing this state
    mode: str  # current navigation mode


class PropulsionInterface(Protocol):
    """Interface for propulsion system integration."""
    
    def get_thrust_vector(self) -> Tuple[float, float, float]:
        """Get current thrust vector."""
        ...
    
    def get_propulsion_state(self) -> Dict[str, Any]:
        """Get current propulsion system state."""
        ...
    
    def set_thrust_parameters(self, params: Dict[str, Any]) -> bool:
        """Set thrust parameters based on navigation needs."""
        ...
    
    def get_performance_envelope(self) -> Dict[str, Any]:
        """Get current performance envelope of the propulsion system."""
        ...


class SensorInterface(Protocol):
    """Interface for sensor system integration."""
    
    def get_sensor_data(self, sensor_type: str) -> Dict[str, Any]:
        """Get data from specific sensor type."""
        ...
    
    def get_terrain_data(self, position: Dict[str, float], radius: float) -> Dict[str, Any]:
        """Get terrain data around a position."""
        ...
    
    def get_obstacle_data(self) -> List[Dict[str, Any]]:
        """Get detected obstacle data."""
        ...
    
    def calibrate_sensor(self, sensor_id: str, navigation_data: NavigationState) -> bool:
        """Calibrate sensor using navigation data."""
        ...


class CommunicationInterface(Protocol):
    """Interface for communication system integration."""
    
    def send_navigation_update(self, nav_state: NavigationState) -> bool:
        """Send navigation state update to other platforms."""
        ...
    
    def receive_navigation_data(self) -> List[NavigationState]:
        """Receive navigation data from other platforms."""
        ...
    
    def get_datalink_quality(self) -> float:
        """Get current datalink quality (0-1)."""
        ...
    
    def request_external_navigation_fix(self) -> Optional[Dict[str, float]]:
        """Request navigation fix from external source."""
        ...


class MissionInterface(Protocol):
    """Interface for mission system integration."""
    
    def get_waypoints(self) -> List[Dict[str, float]]:
        """Get mission waypoints."""
        ...
    
    def get_current_objective(self) -> Dict[str, Any]:
        """Get current mission objective."""
        ...
    
    def update_navigation_status(self, nav_state: NavigationState) -> None:
        """Update mission with current navigation status."""
        ...
    
    def get_navigation_constraints(self) -> Dict[str, Any]:
        """Get navigation constraints from mission parameters."""
        ...


class NavigationServiceProvider:
    """
    Service provider for navigation capabilities.
    
    This class exposes navigation services to other UCAV systems.
    """
    
    def __init__(self, navigation_integrator):
        """
        Initialize navigation service provider.
        
        Args:
            navigation_integrator: The navigation integrator instance
        """
        self.integrator = navigation_integrator
        self.connected_systems: Dict[str, Any] = {}
        self.last_state: Optional[NavigationState] = None
        self.update_counter = 0
    
    def register_consumer(self, system_id: str, system_type: str, callback=None) -> bool:
        """
        Register a system that consumes navigation data.
        
        Args:
            system_id: Unique identifier for the system
            system_type: Type of system (propulsion, sensor, etc.)
            callback: Optional callback function for navigation updates
            
        Returns:
            Success status
        """
        if system_id in self.connected_systems:
            logger.warning(f"System {system_id} already registered")
            return False
            
        self.connected_systems[system_id] = {
            "type": system_type,
            "callback": callback,
            "last_update": 0,
            "active": True
        }
        
        logger.info(f"Registered {system_type} system {system_id} with navigation services")
        return True
    
    def get_current_state(self) -> NavigationState:
        """
        Get current navigation state.
        
        Returns:
            Current navigation state
        """
        # Get position from best available system
        position = self.integrator.get_position()
        orientation = self.integrator.get_orientation() if hasattr(self.integrator, "get_orientation") else {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        velocity = self.integrator.get_velocity() if hasattr(self.integrator, "get_velocity") else {"vx": 0.0, "vy": 0.0, "vz": 0.0}
        
        # Get best system ID
        best_system = self._get_best_system()
        
        # Create navigation state
        state = NavigationState(
            position=position,
            orientation=orientation,
            velocity=velocity,
            acceleration={"ax": 0.0, "ay": 0.0, "az": 0.0},
            timestamp=np.datetime64('now').astype(float),
            confidence=self._calculate_confidence(),
            system_id=best_system or "unknown",
            mode="normal"
        )
        
        self.last_state = state
        self.update_counter += 1
        
        return state
    
    def _get_best_system(self) -> Optional[str]:
        """Get ID of best available navigation system."""
        best_system = None
        best_accuracy = -1.0
        
        for sys_id, state in self.integrator.system_states.items():
            if state.get("active", False):
                accuracy = state.get("accuracy", 0.0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_system = sys_id
        
        return best_system
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence level in current navigation solution."""
        # Count active systems
        active_systems = sum(1 for state in self.integrator.system_states.values() 
                           if state.get("active", False))
        
        # Base confidence on number of active systems and their health
        if active_systems == 0:
            return 0.0
            
        # Calculate average health of active systems
        total_health = sum(state.get("health", 0.0) for state in self.integrator.system_states.values() 
                         if state.get("active", False))
        
        # Normalize confidence between 0.3 (single system) and 1.0 (multiple healthy systems)
        base_confidence = min(0.3 + (active_systems - 1) * 0.2, 0.9)
        health_factor = total_health / active_systems if active_systems > 0 else 0
        
        return base_confidence * health_factor
    
    def notify_consumers(self) -> None:
        """Notify all registered consumers of navigation updates."""
        if not self.last_state:
            self.last_state = self.get_current_state()
            
        for system_id, system_info in self.connected_systems.items():
            if not system_info["active"]:
                continue
                
            callback = system_info["callback"]
            if callback and callable(callback):
                try:
                    callback(self.last_state)
                    system_info["last_update"] = self.update_counter
                except Exception as e:
                    logger.error(f"Error notifying system {system_id}: {str(e)}")
                    system_info["active"] = False
    
    def get_navigation_capabilities(self) -> Dict[str, Any]:
        """
        Get navigation capabilities for other systems.
        
        Returns:
            Dictionary of navigation capabilities
        """
        capabilities = {
            "available_systems": list(self.integrator.navigation_systems.keys()),
            "active_systems": [sys_id for sys_id, state in self.integrator.system_states.items() 
                             if state.get("active", False)],
            "position_accuracy": self._get_best_accuracy("position"),
            "orientation_accuracy": self._get_best_accuracy("orientation"),
            "update_rate": self._calculate_update_rate(),
            "modes": ["normal", "precision", "low_power", "stealth"]
        }
        
        return capabilities
    
    def _get_best_accuracy(self, accuracy_type: str) -> float:
        """Get best accuracy value of specified type from active systems."""
        best_accuracy = 0.0
        
        for system_id, system in self.integrator.navigation_systems.items():
            if not self.integrator.system_states[system_id].get("active", False):
                continue
                
            metrics = system.calculate_performance_metrics()
            accuracy = metrics.get(f"{accuracy_type}_accuracy", 0.0)
            best_accuracy = max(best_accuracy, accuracy)
            
        return best_accuracy
    
    def _calculate_update_rate(self) -> float:
        """Calculate current navigation update rate."""
        # In a real system, this would measure actual update frequency
        # For now, return a reasonable value based on active systems
        active_systems = sum(1 for state in self.integrator.system_states.values() 
                           if state.get("active", False))
        
        # Base rate of 10 Hz, higher with more systems
        return 10.0 + active_systems * 5.0
"""
Sensor Simulation Framework

A lightweight framework for simulating various sensors in the UCAV simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time

from src.core.utils.logging_framework import get_logger

logger = get_logger("sensors")


class SensorType(Enum):
    """Types of sensors that can be simulated."""
    RADAR = 0
    INFRARED = 1
    OPTICAL = 2
    LIDAR = 3
    GPS = 4
    IMU = 5
    ALTIMETER = 6


@dataclass
class SensorConfig:
    """Configuration for a sensor."""
    
    # Basic properties
    type: SensorType
    name: str
    update_rate: float  # Hz
    
    # Field of view (degrees)
    fov_horizontal: float = 60.0
    fov_vertical: float = 45.0
    
    # Range properties
    min_range: float = 0.0  # meters
    max_range: float = 10000.0  # meters
    
    # Accuracy and noise
    accuracy: float = 0.95  # 0-1 scale
    noise_factor: float = 0.01  # 0-1 scale
    
    # Power requirements
    power_consumption: float = 10.0  # watts
    
    # Failure probability per hour
    failure_rate: float = 0.0001


class Sensor:
    """Base class for all sensors."""
    
    def __init__(self, config: SensorConfig):
        """Initialize the sensor."""
        self.config = config
        self.is_active = True
        self.is_failed = False
        self.last_update_time = 0.0
        self.data = {}
        self.rng = np.random.RandomState(int(time.time()))
        
        logger.info(f"Initialized {config.name} sensor of type {config.type.name}")
    
    def update(self, time_now: float, platform_state: Dict[str, Any], 
              environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update sensor readings based on platform state and environment.
        
        Args:
            time_now: Current simulation time (s)
            platform_state: State of the platform (position, orientation, etc.)
            environment: Environmental conditions
            
        Returns:
            Dict[str, Any]: Sensor data
        """
        # Check if it's time to update
        dt = time_now - self.last_update_time
        if dt < 1.0 / self.config.update_rate:
            return self.data
        
        # Check for random failure
        if not self.is_failed and self.rng.random() < self.config.failure_rate * dt / 3600.0:
            self.is_failed = True
            logger.warning(f"Sensor {self.config.name} has failed")
            return self.data
        
        # Skip if inactive or failed
        if not self.is_active or self.is_failed:
            return self.data
        
        # Update sensor data (to be implemented by subclasses)
        self._update_sensor_data(platform_state, environment)
        
        # Add noise to data
        self._add_noise()
        
        self.last_update_time = time_now
        return self.data
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """
        Update sensor data based on platform state and environment.
        To be implemented by subclasses.
        """
        pass
    
    def _add_noise(self) -> None:
        """Add noise to sensor data based on noise factor."""
        pass
    
    def set_active(self, active: bool) -> None:
        """Set sensor active state."""
        self.is_active = active
        logger.debug(f"Sensor {self.config.name} {'activated' if active else 'deactivated'}")
    
    def reset(self) -> None:
        """Reset sensor to initial state."""
        self.is_failed = False
        self.last_update_time = 0.0
        self.data = {}
        logger.debug(f"Sensor {self.config.name} reset")


class Radar(Sensor):
    """Radar sensor simulation."""
    
    def __init__(self, config: SensorConfig):
        """Initialize radar sensor."""
        super().__init__(config)
        self.data = {
            'targets': [],
            'range': 0.0,
            'azimuth': 0.0,
            'elevation': 0.0,
            'velocity': 0.0
        }
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update radar sensor data."""
        # Get platform position and orientation
        position = platform_state.get('position', np.zeros(3))
        orientation = platform_state.get('orientation', np.zeros(3))
        
        # Get targets from environment
        targets = environment.get('targets', [])
        detected_targets = []
        
        for target in targets:
            target_pos = target.get('position', np.zeros(3))
            
            # Calculate relative position
            rel_pos = target_pos - position
            distance = np.linalg.norm(rel_pos)
            
            # Check if in range
            if distance < self.config.min_range or distance > self.config.max_range:
                continue
            
            # Calculate angles
            azimuth = np.arctan2(rel_pos[1], rel_pos[0])
            elevation = np.arcsin(rel_pos[2] / distance)
            
            # Convert to degrees
            azimuth_deg = np.degrees(azimuth)
            elevation_deg = np.degrees(elevation)
            
            # Check if in field of view
            if (abs(azimuth_deg) > self.config.fov_horizontal / 2 or
                abs(elevation_deg) > self.config.fov_vertical / 2):
                continue
            
            # Calculate detection probability based on distance
            detection_prob = self.config.accuracy * (1.0 - distance / self.config.max_range)
            
            # Apply weather effects
            if 'weather' in environment:
                weather = environment['weather']
                if weather in ['RAIN', 'SNOW', 'FOG']:
                    detection_prob *= 0.7  # Reduced effectiveness in bad weather
            
            # Random detection based on probability
            if self.rng.random() < detection_prob:
                # Calculate relative velocity (simplified)
                target_vel = target.get('velocity', np.zeros(3))
                platform_vel = platform_state.get('velocity', np.zeros(3))
                rel_vel = target_vel - platform_vel
                
                # Radial velocity component
                radial_vel = np.dot(rel_vel, rel_pos) / distance
                
                detected_targets.append({
                    'id': target.get('id', 0),
                    'distance': distance,
                    'azimuth': azimuth_deg,
                    'elevation': elevation_deg,
                    'radial_velocity': radial_vel,
                    'signal_strength': detection_prob
                })
        
        # Update radar data
        self.data['targets'] = detected_targets
        
        # Update terrain data if available
        if 'terrain' in environment:
            terrain = environment['terrain']
            terrain_height = terrain.get_height(position[0], position[1])
            self.data['terrain_height'] = terrain_height
            self.data['altitude_agl'] = position[2] - terrain_height
    
    def _add_noise(self) -> None:
        """Add noise to radar data."""
        noise_factor = self.config.noise_factor
        
        for target in self.data['targets']:
            # Add noise to measurements
            target['distance'] += self.rng.normal(0, noise_factor * target['distance'])
            target['azimuth'] += self.rng.normal(0, noise_factor * 5.0)  # degrees
            target['elevation'] += self.rng.normal(0, noise_factor * 3.0)  # degrees
            target['radial_velocity'] += self.rng.normal(0, noise_factor * 5.0)  # m/s


class Altimeter(Sensor):
    """Altimeter sensor simulation."""
    
    def __init__(self, config: SensorConfig):
        """Initialize altimeter sensor."""
        super().__init__(config)
        self.data = {
            'altitude_agl': 0.0,  # Above ground level
            'altitude_msl': 0.0,  # Mean sea level
            'vertical_speed': 0.0
        }
        self._last_altitude = 0.0
        self._last_time = 0.0
    
    def _update_sensor_data(self, platform_state: Dict[str, Any], 
                           environment: Dict[str, Any]) -> None:
        """Update altimeter sensor data."""
        # Get current time from the last update time
        time_now = self.last_update_time
        
        # Get platform position
        position = platform_state.get('position', np.zeros(3))
        
        # Mean sea level altitude
        altitude_msl = position[2]
        
        # Above ground level altitude
        altitude_agl = altitude_msl
        if 'terrain' in environment:
            terrain = environment['terrain']
            terrain_height = terrain.get_height(position[0], position[1])
            altitude_agl = altitude_msl - terrain_height
        
        # Calculate vertical speed
        dt = time_now - self._last_time
        if dt > 0:
            self.data['vertical_speed'] = (altitude_msl - self._last_altitude) / dt
        
        # Update data
        self.data['altitude_agl'] = altitude_agl
        self.data['altitude_msl'] = altitude_msl
        
        # Store for next update
        self._last_altitude = altitude_msl
        self._last_time = time_now
    
    def _add_noise(self) -> None:
        """Add noise to altimeter data."""
        noise_factor = self.config.noise_factor
        
        # Add noise to measurements
        self.data['altitude_agl'] += self.rng.normal(0, noise_factor * max(10.0, self.data['altitude_agl'] * 0.01))
        self.data['altitude_msl'] += self.rng.normal(0, noise_factor * max(10.0, self.data['altitude_msl'] * 0.01))
        self.data['vertical_speed'] += self.rng.normal(0, noise_factor * max(1.0, abs(self.data['vertical_speed']) * 0.05))


class SensorManager:
    """Manages multiple sensors and provides a unified interface."""
    
    def __init__(self):
        """Initialize the sensor manager."""
        self.sensors = {}
        logger.info("Sensor manager initialized")
    
    def add_sensor(self, sensor: Sensor) -> None:
        """Add a sensor to the manager."""
        self.sensors[sensor.config.name] = sensor
        logger.info(f"Added sensor {sensor.config.name} to manager")
    
    def remove_sensor(self, name: str) -> None:
        """Remove a sensor from the manager."""
        if name in self.sensors:
            del self.sensors[name]
            logger.info(f"Removed sensor {name} from manager")
    
    def update_all(self, time_now: float, platform_state: Dict[str, Any], 
                  environment: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Update all sensors.
        
        Args:
            time_now: Current simulation time (s)
            platform_state: State of the platform
            environment: Environmental conditions
            
        Returns:
            Dict[str, Dict[str, Any]]: All sensor data
        """
        sensor_data = {}
        
        for name, sensor in self.sensors.items():
            sensor_data[name] = sensor.update(time_now, platform_state, environment)
        
        return sensor_data
    
    def get_sensor(self, name: str) -> Optional[Sensor]:
        """Get a sensor by name."""
        return self.sensors.get(name)
    
    def reset_all(self) -> None:
        """Reset all sensors."""
        for sensor in self.sensors.values():
            sensor.reset()
        logger.info("All sensors reset")


def create_default_sensors() -> SensorManager:
    """
    Create a default set of sensors.
    
    Returns:
        SensorManager: Sensor manager with default sensors
    """
    manager = SensorManager()
    
    # Add radar
    radar_config = SensorConfig(
        type=SensorType.RADAR,
        name="primary_radar",
        update_rate=10.0,
        fov_horizontal=120.0,
        fov_vertical=60.0,
        max_range=50000.0,
        accuracy=0.9,
        noise_factor=0.02
    )
    manager.add_sensor(Radar(radar_config))
    
    # Add altimeter
    alt_config = SensorConfig(
        type=SensorType.ALTIMETER,
        name="radar_altimeter",
        update_rate=20.0,
        max_range=5000.0,
        accuracy=0.98,
        noise_factor=0.01
    )
    manager.add_sensor(Altimeter(alt_config))
    
    return manager
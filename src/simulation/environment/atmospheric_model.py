"""
Atmospheric and Environmental Factors Simulation

Provides realistic environmental conditions for flight simulation including
wind, temperature, pressure, and weather effects.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import time
from enum import Enum

from src.core.utils.logging_framework import get_logger

logger = get_logger("environment")


class WeatherCondition(Enum):
    """Weather condition types."""
    CLEAR = 0
    PARTLY_CLOUDY = 1
    CLOUDY = 2
    RAIN = 3
    THUNDERSTORM = 4
    SNOW = 5
    FOG = 6


@dataclass
class WindProfile:
    """Wind profile parameters."""
    
    # Base wind vector (m/s) in Earth frame [east, north, up]
    base_velocity: np.ndarray
    
    # Wind shear parameters (change in wind per meter of altitude)
    shear_factor: float = 0.001
    
    # Turbulence intensity (0-1 scale)
    turbulence_intensity: float = 0.0
    
    # Turbulence scale length (m)
    turbulence_scale: float = 100.0
    
    # Gust parameters
    gust_probability: float = 0.0
    max_gust_magnitude: float = 0.0
    max_gust_duration: float = 0.0


class AtmosphericModel:
    """
    Atmospheric and environmental factors simulation model.
    """
    
    def __init__(self):
        """Initialize the atmospheric model."""
        # Standard atmospheric parameters
        self.sea_level_pressure = 101325.0  # Pa
        self.sea_level_temperature = 288.15  # K
        self.sea_level_density = 1.225  # kg/m^3
        self.gas_constant = 287.05  # J/(kg*K)
        self.gravity = 9.81  # m/s^2
        self.lapse_rate = 0.0065  # K/m
        
        # Wind model
        self.wind_profile = WindProfile(base_velocity=np.zeros(3))
        
        # Current gust state
        self.current_gust = np.zeros(3)
        self.gust_remaining_time = 0.0
        
        # Weather conditions
        self.weather = WeatherCondition.CLEAR
        self.cloud_base = 2000.0  # m
        self.cloud_tops = 3000.0  # m
        self.visibility = 10000.0  # m
        
        # Terrain effects
        self.terrain_roughness = 0.0  # 0-1 scale
        
        # Time of day effects (0-24 hours)
        self.time_of_day = 12.0
        
        # Random seed for reproducibility
        self.seed = int(time.time())
        self.rng = np.random.RandomState(self.seed)
        
        logger.info("Atmospheric model initialized with standard conditions")
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible simulations."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def set_wind_profile(self, wind_profile: WindProfile):
        """Set the wind profile."""
        self.wind_profile = wind_profile
        logger.info(f"Wind profile set with base velocity {wind_profile.base_velocity} m/s")
    
    def set_weather_condition(self, condition: WeatherCondition, 
                             cloud_base: float = 2000.0,
                             cloud_tops: float = 3000.0,
                             visibility: float = 10000.0):
        """Set weather conditions."""
        self.weather = condition
        self.cloud_base = cloud_base
        self.cloud_tops = cloud_tops
        self.visibility = visibility
        
        # Adjust turbulence based on weather
        if condition == WeatherCondition.CLEAR:
            self.wind_profile.turbulence_intensity = 0.05
        elif condition == WeatherCondition.PARTLY_CLOUDY:
            self.wind_profile.turbulence_intensity = 0.1
        elif condition == WeatherCondition.CLOUDY:
            self.wind_profile.turbulence_intensity = 0.15
        elif condition == WeatherCondition.RAIN:
            self.wind_profile.turbulence_intensity = 0.2
            self.visibility = min(visibility, 5000.0)
        elif condition == WeatherCondition.THUNDERSTORM:
            self.wind_profile.turbulence_intensity = 0.4
            self.wind_profile.gust_probability = 0.1
            self.wind_profile.max_gust_magnitude = 15.0
            self.visibility = min(visibility, 2000.0)
        elif condition == WeatherCondition.SNOW:
            self.wind_profile.turbulence_intensity = 0.15
            self.visibility = min(visibility, 1000.0)
        elif condition == WeatherCondition.FOG:
            self.wind_profile.turbulence_intensity = 0.05
            self.visibility = min(visibility, 500.0)
        
        logger.info(f"Weather set to {condition.name} with visibility {self.visibility}m")
    
    def get_temperature(self, altitude: float) -> float:
        """
        Get air temperature at the specified altitude.
        
        Args:
            altitude: Altitude above sea level (m)
            
        Returns:
            float: Temperature (K)
        """
        # International Standard Atmosphere model
        if altitude < 11000:  # Troposphere
            return self.sea_level_temperature - self.lapse_rate * altitude
        else:  # Stratosphere (simplified)
            return 216.65
    
    def get_pressure(self, altitude: float) -> float:
        """
        Get air pressure at the specified altitude.
        
        Args:
            altitude: Altitude above sea level (m)
            
        Returns:
            float: Pressure (Pa)
        """
        # International Standard Atmosphere model
        if altitude < 11000:  # Troposphere
            temperature = self.get_temperature(altitude)
            return self.sea_level_pressure * (temperature / self.sea_level_temperature) ** 5.255
        else:  # Stratosphere (simplified)
            return 22632 * np.exp(-0.00015769 * (altitude - 11000))
    
    def get_density(self, altitude: float) -> float:
        """
        Get air density at the specified altitude.
        
        Args:
            altitude: Altitude above sea level (m)
            
        Returns:
            float: Density (kg/m^3)
        """
        # Calculate from pressure and temperature using ideal gas law
        pressure = self.get_pressure(altitude)
        temperature = self.get_temperature(altitude)
        return pressure / (self.gas_constant * temperature)
    
    def get_speed_of_sound(self, altitude: float) -> float:
        """
        Get speed of sound at the specified altitude.
        
        Args:
            altitude: Altitude above sea level (m)
            
        Returns:
            float: Speed of sound (m/s)
        """
        # a = sqrt(gamma * R * T)
        temperature = self.get_temperature(altitude)
        return np.sqrt(1.4 * self.gas_constant * temperature)
    
    def _generate_turbulence(self, altitude: float) -> np.ndarray:
        """
        Generate turbulence components based on current conditions.
        
        Args:
            altitude: Altitude above sea level (m)
            
        Returns:
            np.ndarray: Turbulence velocity components [east, north, up]
        """
        # Scale turbulence with intensity and altitude
        base_intensity = self.wind_profile.turbulence_intensity
        
        # Increase turbulence near clouds
        if (self.weather in [WeatherCondition.PARTLY_CLOUDY, WeatherCondition.CLOUDY, 
                            WeatherCondition.RAIN, WeatherCondition.THUNDERSTORM] and
            altitude > self.cloud_base - 200 and altitude < self.cloud_tops + 200):
            # Enhanced turbulence at cloud boundaries
            distance_from_boundary = min(
                abs(altitude - self.cloud_base),
                abs(altitude - self.cloud_tops)
            )
            if distance_from_boundary < 200:
                base_intensity *= 1.5
        
        # Terrain-induced turbulence at low altitudes
        if altitude < 500:
            terrain_factor = 1.0 + self.terrain_roughness * (1.0 - altitude / 500)
            base_intensity *= terrain_factor
        
        # Generate random turbulence components
        # Using simplified Dryden model approach
        sigma = base_intensity * 3.0  # Standard deviation in m/s
        turbulence = self.rng.normal(0, sigma, 3)
        
        return turbulence
    
    def _update_gusts(self, dt: float) -> np.ndarray:
        """
        Update and return current gust components.
        
        Args:
            dt: Time step (s)
            
        Returns:
            np.ndarray: Gust velocity components [east, north, up]
        """
        # Update existing gust
        if self.gust_remaining_time > 0:
            self.gust_remaining_time -= dt
            if self.gust_remaining_time <= 0:
                self.current_gust = np.zeros(3)
        
        # Possibly generate new gust
        if (self.gust_remaining_time <= 0 and 
            self.wind_profile.gust_probability > 0 and
            self.rng.random() < self.wind_profile.gust_probability * dt):
            
            # Generate new gust
            magnitude = self.rng.uniform(0, self.wind_profile.max_gust_magnitude)
            direction = self.rng.uniform(0, 2 * np.pi)
            
            # Horizontal gust
            self.current_gust[0] = magnitude * np.cos(direction)
            self.current_gust[1] = magnitude * np.sin(direction)
            
            # Vertical component (usually smaller)
            self.current_gust[2] = self.rng.uniform(-0.3, 0.3) * magnitude
            
            # Set duration
            self.gust_remaining_time = self.rng.uniform(1.0, self.wind_profile.max_gust_duration)
            
            logger.debug(f"Generated gust with magnitude {magnitude:.1f} m/s, duration {self.gust_remaining_time:.1f}s")
        
        return self.current_gust
    
    def get_wind_at_altitude(self, altitude: float, dt: float = 0.0) -> np.ndarray:
        """
        Get wind velocity at the specified altitude.
        
        Args:
            altitude: Altitude above sea level (m)
            dt: Time step for gust updates (s)
            
        Returns:
            np.ndarray: Wind velocity [east, north, up] (m/s)
        """
        # Base wind with altitude-based shear
        wind = self.wind_profile.base_velocity.copy()
        
        # Apply wind shear (wind typically increases with altitude)
        shear_factor = self.wind_profile.shear_factor
        altitude_factor = min(1.0, altitude / 1000.0)  # Cap effect at 1000m
        wind[0] *= (1.0 + shear_factor * altitude)
        wind[1] *= (1.0 + shear_factor * altitude)
        
        # Add turbulence
        if self.wind_profile.turbulence_intensity > 0:
            wind += self._generate_turbulence(altitude)
        
        # Add gusts
        if dt > 0 and self.wind_profile.gust_probability > 0:
            wind += self._update_gusts(dt)
        
        return wind
    
    def get_atmospheric_conditions(self, altitude: float, dt: float = 0.0) -> Dict[str, Any]:
        """
        Get complete atmospheric conditions at the specified altitude.
        
        Args:
            altitude: Altitude above sea level (m)
            dt: Time step for gust updates (s)
            
        Returns:
            Dict[str, Any]: Atmospheric conditions
        """
        temperature = self.get_temperature(altitude)
        pressure = self.get_pressure(altitude)
        density = self.get_density(altitude)
        speed_of_sound = self.get_speed_of_sound(altitude)
        wind = self.get_wind_at_altitude(altitude, dt)
        
        # Determine if in clouds
        in_clouds = (self.weather in [WeatherCondition.CLOUDY, WeatherCondition.RAIN, 
                                     WeatherCondition.THUNDERSTORM, WeatherCondition.SNOW] and
                    altitude >= self.cloud_base and altitude <= self.cloud_tops)
        
        # Determine visibility at altitude
        actual_visibility = self.visibility
        if in_clouds:
            if self.weather == WeatherCondition.RAIN:
                actual_visibility = min(actual_visibility, 500.0)
            elif self.weather == WeatherCondition.THUNDERSTORM:
                actual_visibility = min(actual_visibility, 200.0)
            elif self.weather == WeatherCondition.SNOW:
                actual_visibility = min(actual_visibility, 100.0)
            else:  # CLOUDY
                actual_visibility = min(actual_visibility, 1000.0)
        elif self.weather == WeatherCondition.FOG and altitude < self.cloud_tops:
            actual_visibility = min(actual_visibility, 200.0)
        
        return {
            'temperature': temperature,
            'pressure': pressure,
            'density': density,
            'speed_of_sound': speed_of_sound,
            'wind': wind,
            'turbulence_intensity': self.wind_profile.turbulence_intensity,
            'weather': self.weather,
            'in_clouds': in_clouds,
            'visibility': actual_visibility
        }


def create_default_atmosphere() -> AtmosphericModel:
    """
    Create a default atmospheric model with mild conditions.
    
    Returns:
        AtmosphericModel: Default atmospheric model
    """
    model = AtmosphericModel()
    
    # Set mild wind conditions
    wind_profile = WindProfile(
        base_velocity=np.array([5.0, 2.0, 0.0]),  # 5 m/s from west, 2 m/s from south
        turbulence_intensity=0.1,
        gust_probability=0.01,
        max_gust_magnitude=8.0,
        max_gust_duration=5.0
    )
    model.set_wind_profile(wind_profile)
    
    # Set partly cloudy weather
    model.set_weather_condition(
        WeatherCondition.PARTLY_CLOUDY,
        cloud_base=1800.0,
        cloud_tops=2500.0,
        visibility=8000.0
    )
    
    return model
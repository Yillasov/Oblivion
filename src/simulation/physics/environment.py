from typing import Dict, Any, List, Optional
import numpy as np

class EnvironmentModel:
    """Physics-based environment model for airframe simulation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wind = config.get("wind", np.zeros(3))
        self.turbulence_intensity = config.get("turbulence_intensity", 0.0)
        self.temperature = config.get("temperature", 288.15)  # K (15°C)
        self.pressure = config.get("pressure", 101325)  # Pa (sea level)
        
    def get_air_density(self, altitude: float) -> float:
        """Calculate air density based on altitude."""
        # Simple exponential atmosphere model
        return 1.225 * np.exp(-altitude / 8500.0)
    
    def get_wind_vector(self, position: np.ndarray) -> np.ndarray:
        """Get wind vector at a given position."""
        # Base wind vector
        wind_vector = self.wind.copy()
        
        # Add turbulence if enabled
        if self.turbulence_intensity > 0:
            turbulence = np.random.normal(0, self.turbulence_intensity, 3)
            wind_vector += turbulence
        
        return wind_vector
    
    def get_atmospheric_conditions(self, position: np.ndarray) -> Dict[str, float]:
        """Get atmospheric conditions at a given position."""
        altitude = position[2]
        
        # Calculate temperature using standard lapse rate
        temperature = self.temperature - 0.0065 * altitude
        
        # Calculate pressure using barometric formula
        pressure = self.pressure * (temperature / self.temperature) ** 5.2561
        
        # Calculate density
        density = self.get_air_density(altitude)
        
        # Calculate speed of sound
        speed_of_sound = 20.05 * np.sqrt(temperature)
        
        return {
            "temperature": temperature,
            "pressure": pressure,
            "density": density,
            "speed_of_sound": speed_of_sound
        }
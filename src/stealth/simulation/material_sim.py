"""
Material properties simulation for stealth systems.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.stealth.base.config import StealthMaterialConfig
from src.stealth.materials.ram.ram_system import RAMMaterial


class MaterialPropertiesSimulator:
    """Simulator for stealth material properties under various conditions."""
    
    def __init__(self):
        """Initialize the material properties simulator."""
        self.temperature_effects = {
            "permittivity": 0.002,  # % change per degree C
            "conductivity": 0.003,  # % change per degree C
            "thickness": 0.0001,    # % change per degree C
        }
        
        self.humidity_effects = {
            "permittivity": 0.005,  # % change per % humidity
            "conductivity": 0.008,  # % change per % humidity
            "absorption": 0.003,    # % change per % humidity
        }
        
    def simulate_material(self, 
                         material_config: StealthMaterialConfig, 
                         environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate material properties under given environmental conditions.
        
        Args:
            material_config: Material configuration
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of simulated material properties
        """
        # Extract environmental conditions
        temperature = environmental_conditions.get("temperature", 20.0)  # °C
        humidity = environmental_conditions.get("humidity", 50.0)  # %
        altitude = environmental_conditions.get("altitude", 0.0)  # m
        
        # Base properties
        base_thickness = material_config.thickness_mm
        base_permittivity = material_config.permittivity or 4.0  # Default value if None
        base_conductivity = material_config.conductivity or 0.01  # Default value if None
        
        # Calculate temperature effects
        temp_diff = temperature - 20.0  # Difference from standard temperature
        thickness_factor = 1.0 + (temp_diff * self.temperature_effects["thickness"])
        permittivity_factor = 1.0 + (temp_diff * self.temperature_effects["permittivity"])
        conductivity_factor = 1.0 + (temp_diff * self.temperature_effects["conductivity"])
        
        # Calculate humidity effects
        humidity_diff = humidity - 50.0  # Difference from standard humidity
        permittivity_humidity_factor = 1.0 + (humidity_diff * self.humidity_effects["permittivity"])
        conductivity_humidity_factor = 1.0 + (humidity_diff * self.humidity_effects["conductivity"])
        
        # Calculate final properties
        simulated_thickness = base_thickness * thickness_factor
        simulated_permittivity = base_permittivity * permittivity_factor * permittivity_humidity_factor
        simulated_conductivity = base_conductivity * conductivity_factor * conductivity_humidity_factor
        
        # Calculate radar absorption effectiveness
        base_effectiveness = 0.8  # Base effectiveness (80%)
        temperature_effectiveness = 1.0 - abs(temp_diff) * 0.005  # Reduce by 0.5% per degree from optimal
        humidity_effectiveness = 1.0 - abs(humidity_diff) * 0.002  # Reduce by 0.2% per % from optimal
        
        # Altitude effects (simplified)
        altitude_factor = 1.0
        if altitude > 10000:
            altitude_factor = 0.9  # 10% reduction at high altitude
        
        # Calculate overall effectiveness
        overall_effectiveness = base_effectiveness * temperature_effectiveness * humidity_effectiveness * altitude_factor
        
        return {
            "simulated_thickness_mm": simulated_thickness,
            "simulated_permittivity": simulated_permittivity,
            "simulated_conductivity": simulated_conductivity,
            "absorption_effectiveness": overall_effectiveness,
            "environmental_factors": {
                "temperature_factor": temperature_effectiveness,
                "humidity_factor": humidity_effectiveness,
                "altitude_factor": altitude_factor
            }
        }
    
    def simulate_ram_material(self, 
                             ram_material: RAMMaterial,
                             environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate RAM material properties under given environmental conditions.
        
        Args:
            ram_material: RAM material
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of simulated material properties
        """
        # Extract environmental conditions
        temperature = environmental_conditions.get("temperature", 20.0)  # °C
        
        # Check if temperature is within operating range
        temp_range = ram_material.temperature_range
        if temperature < temp_range[0] or temperature > temp_range[1]:
            effectiveness_factor = 0.7  # Reduced effectiveness outside operating range
        else:
            # Calculate how optimal the temperature is (1.0 at middle of range, lower at extremes)
            optimal_temp = (temp_range[0] + temp_range[1]) / 2
            temp_deviation = abs(temperature - optimal_temp) / (temp_range[1] - temp_range[0])
            effectiveness_factor = 1.0 - (temp_deviation * 0.3)  # At most 30% reduction
        
        # Simulate frequency response changes
        modified_frequency_response = {}
        for freq, attenuation in ram_material.frequency_response.items():
            modified_frequency_response[freq] = attenuation * effectiveness_factor
        
        return {
            "name": ram_material.name,
            "modified_frequency_response": modified_frequency_response,
            "effectiveness_factor": effectiveness_factor,
            "simulated_density": ram_material.density * (1.0 + (temperature - 20) * 0.0001),
            "weather_performance": ram_material.weather_resistance * effectiveness_factor
        }
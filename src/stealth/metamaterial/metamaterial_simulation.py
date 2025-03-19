"""
Metamaterial property simulation for electromagnetic cloaking.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from src.stealth.metamaterial.electromagnetic_cloaking import ElectromagneticProperties, MetamaterialCloaking


class MetamaterialSimulator:
    """
    Simulator for metamaterial properties under various environmental conditions.
    Provides models for how temperature, humidity, and other factors affect cloaking.
    """
    
    def __init__(self):
        """Initialize metamaterial simulator."""
        # Environmental effect coefficients
        self.temperature_effects = {
            "permittivity": 0.003,  # % change per degree C
            "permeability": 0.002,  # % change per degree C
            "conductivity": 0.005,  # % change per degree C
            "resonance": 0.001,     # % change per degree C
        }
        
        self.humidity_effects = {
            "permittivity": 0.004,  # % change per % humidity
            "permeability": 0.001,  # % change per % humidity
            "conductivity": 0.007,  # % change per % humidity
        }
        
        self.altitude_effects = {
            "permittivity": 0.0001,  # % change per meter
            "permeability": 0.0001,  # % change per meter
        }
    
    def simulate_properties(self, 
                           base_properties: ElectromagneticProperties,
                           environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate metamaterial properties under given environmental conditions.
        
        Args:
            base_properties: Base electromagnetic properties
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of simulated properties
        """
        # Extract environmental conditions
        temperature = environmental_conditions.get("temperature", 20.0)  # °C
        humidity = environmental_conditions.get("humidity", 50.0)  # %
        altitude = environmental_conditions.get("altitude", 0.0)  # m
        
        # Calculate temperature effects
        temp_diff = temperature - 20.0  # Difference from standard temperature
        permittivity_temp_factor = 1.0 + (temp_diff * self.temperature_effects["permittivity"])
        permeability_temp_factor = 1.0 + (temp_diff * self.temperature_effects["permeability"])
        conductivity_temp_factor = 1.0 + (temp_diff * self.temperature_effects["conductivity"])
        resonance_temp_factor = 1.0 + (temp_diff * self.temperature_effects["resonance"])
        
        # Calculate humidity effects
        humidity_diff = humidity - 50.0  # Difference from standard humidity
        permittivity_humidity_factor = 1.0 + (humidity_diff * self.humidity_effects["permittivity"])
        permeability_humidity_factor = 1.0 + (humidity_diff * self.humidity_effects["permeability"])
        conductivity_humidity_factor = 1.0 + (humidity_diff * self.humidity_effects["conductivity"])
        
        # Calculate altitude effects
        permittivity_altitude_factor = 1.0 + (altitude * self.altitude_effects["permittivity"])
        permeability_altitude_factor = 1.0 + (altitude * self.altitude_effects["permeability"])
        
        # Calculate final properties
        simulated_permittivity = base_properties.permittivity * permittivity_temp_factor * permittivity_humidity_factor * permittivity_altitude_factor
        simulated_permeability = base_properties.permeability * permeability_temp_factor * permeability_humidity_factor * permeability_altitude_factor
        simulated_conductivity = base_properties.conductivity * conductivity_temp_factor * conductivity_humidity_factor
        simulated_resonant_frequency = base_properties.resonant_frequency * resonance_temp_factor
        
        # Calculate overall effectiveness factor
        effectiveness_factor = self._calculate_effectiveness_factor(temperature, humidity, altitude)
        
        return {
            "simulated_properties": ElectromagneticProperties(
                permittivity=simulated_permittivity,
                permeability=simulated_permeability,
                conductivity=simulated_conductivity,
                resonant_frequency=simulated_resonant_frequency,
                phase_shift=base_properties.phase_shift,
                bandwidth=base_properties.bandwidth
            ),
            "effectiveness_factor": effectiveness_factor,
            "environmental_impact": {
                "temperature_impact": abs(1.0 - permittivity_temp_factor),
                "humidity_impact": abs(1.0 - permittivity_humidity_factor),
                "altitude_impact": abs(1.0 - permittivity_altitude_factor)
            }
        }
    
    def _calculate_effectiveness_factor(self, temperature: float, humidity: float, altitude: float) -> float:
        """Calculate overall effectiveness factor based on environmental conditions."""
        # Base effectiveness
        base_effectiveness = 1.0
        
        # Temperature effect (optimal at 20°C)
        temp_effect = 1.0 - (abs(temperature - 20.0) * 0.01)  # 1% reduction per degree from optimal
        
        # Humidity effect (optimal at 40-60%)
        if 40.0 <= humidity <= 60.0:
            humidity_effect = 1.0
        else:
            humidity_effect = 1.0 - (min(abs(humidity - 50.0), 50.0) * 0.005)  # 0.5% per % from optimal
        
        # Altitude effect (reduced effectiveness at high altitudes)
        altitude_effect = 1.0
        if altitude > 5000.0:
            altitude_effect = 1.0 - ((altitude - 5000.0) / 15000.0)  # Linear reduction above 5000m
            altitude_effect = max(0.7, altitude_effect)  # Maximum 30% reduction
        
        return base_effectiveness * temp_effect * humidity_effect * altitude_effect
    
    def simulate_cloaking_performance(self, 
                                    cloaking_system: MetamaterialCloaking,
                                    environmental_conditions: Dict[str, float],
                                    radar_frequencies: List[float]) -> Dict[str, Any]:
        """
        Simulate cloaking performance across different radar frequencies.
        
        Args:
            cloaking_system: Metamaterial cloaking system
            environmental_conditions: Environmental conditions
            radar_frequencies: List of radar frequencies to test (GHz)
            
        Returns:
            Dictionary of performance metrics
        """
        # Simulate properties
        simulated_props = self.simulate_properties(
            cloaking_system.properties, 
            environmental_conditions
        )
        
        # Calculate RCS reduction at each frequency
        rcs_reductions = []
        for freq in radar_frequencies:
            # Test at different incident angles
            angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
            angle_results = []
            
            for angle in angles:
                # Use the cloaking system's calculation with simulated properties
                # Temporarily replace properties for calculation
                original_props = cloaking_system.properties
                cloaking_system.properties = simulated_props["simulated_properties"]
                
                reduction = cloaking_system.calculate_radar_cross_section(freq, angle)
                angle_results.append(reduction)
                
                # Restore original properties
                cloaking_system.properties = original_props
            
            rcs_reductions.append({
                "frequency": freq,
                "average_reduction": np.mean(angle_results),
                "min_reduction": np.min(angle_results),
                "max_reduction": np.max(angle_results),
                "angle_results": dict(zip([str(int(a * 180/np.pi)) + "°" for a in angles], angle_results))
            })
        
        return {
            "environmental_conditions": environmental_conditions,
            "simulated_properties": simulated_props,
            "rcs_reductions": rcs_reductions,
            "overall_effectiveness": simulated_props["effectiveness_factor"]
        }
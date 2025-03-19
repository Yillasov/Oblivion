"""
Simulation model for Radar-Absorbent Material (RAM) effectiveness.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from src.stealth.materials.ram.ram_system import RAMSystem, RAMMaterial


class RAMSimulator:
    """Simple simulator for RAM effectiveness under various conditions."""
    
    def __init__(self, ram_system: RAMSystem):
        """
        Initialize RAM simulator.
        
        Args:
            ram_system: RAM system to simulate
        """
        self.ram_system = ram_system
        
    def simulate_frequency_response(self, 
                                   frequency_range: Tuple[float, float],
                                   frequency_steps: int = 20,
                                   environmental_conditions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Simulate RAM frequency response across a range of frequencies.
        
        Args:
            frequency_range: (min_freq, max_freq) in GHz
            frequency_steps: Number of frequency points to simulate
            environmental_conditions: Environmental conditions for simulation
            
        Returns:
            Simulation results
        """
        if not self.ram_system.active_material:
            return {"error": "No active material selected"}
            
        # Default environmental conditions
        env_conditions = environmental_conditions or {
            "temperature": 20.0,
            "humidity": 50.0
        }
        
        # Generate frequency points
        min_freq, max_freq = frequency_range
        frequencies = np.linspace(min_freq, max_freq, frequency_steps)
        
        # Simulate response at each frequency
        results = []
        for freq in frequencies:
            threat_data = {"radar_frequency_ghz": freq}
            effectiveness = self.ram_system.calculate_effectiveness(threat_data, env_conditions)
            results.append({
                "frequency_ghz": freq,
                "radar_reduction": effectiveness["radar_reduction"],
                "attenuation_db": effectiveness["attenuation_db"]
            })
            
        return {
            "material": self.ram_system.active_material.name,
            "environmental_conditions": env_conditions,
            "frequency_range": frequency_range,
            "results": results
        }
        
    def simulate_environmental_impact(self,
                                     threat_frequency: float,
                                     temperature_range: Tuple[float, float] = (-20.0, 80.0),
                                     humidity_range: Tuple[float, float] = (0.0, 100.0),
                                     steps: int = 10) -> Dict[str, Any]:
        """
        Simulate environmental impact on RAM effectiveness.
        
        Args:
            threat_frequency: Radar frequency in GHz
            temperature_range: Temperature range to simulate (°C)
            humidity_range: Humidity range to simulate (%)
            steps: Number of steps for each parameter
            
        Returns:
            Simulation results
        """
        if not self.ram_system.active_material:
            return {"error": "No active material selected"}
            
        # Generate parameter ranges
        temperatures = np.linspace(temperature_range[0], temperature_range[1], steps)
        humidities = np.linspace(humidity_range[0], humidity_range[1], steps)
        
        # Threat data
        threat_data = {"radar_frequency_ghz": threat_frequency}
        
        # Simulate temperature impact
        temp_results = []
        for temp in temperatures:
            env_conditions = {"temperature": temp, "humidity": 50.0}
            effectiveness = self.ram_system.calculate_effectiveness(threat_data, env_conditions)
            temp_results.append({
                "temperature": temp,
                "radar_reduction": effectiveness["radar_reduction"],
                "temperature_factor": effectiveness["temperature_factor"]
            })
            
        # Simulate humidity impact
        humidity_results = []
        for humidity in humidities:
            env_conditions = {"temperature": 20.0, "humidity": humidity}
            effectiveness = self.ram_system.calculate_effectiveness(threat_data, env_conditions)
            humidity_results.append({
                "humidity": humidity,
                "radar_reduction": effectiveness["radar_reduction"],
                "humidity_factor": effectiveness["humidity_factor"]
            })
            
        return {
            "material": self.ram_system.active_material.name,
            "threat_frequency": threat_frequency,
            "temperature_results": temp_results,
            "humidity_results": humidity_results
        }
        
    def plot_frequency_response(self, simulation_result: Dict[str, Any]) -> None:
        """
        Plot frequency response simulation results.
        
        Args:
            simulation_result: Results from simulate_frequency_response
        """
        if "error" in simulation_result:
            print(f"Error: {simulation_result['error']}")
            return
            
        frequencies = [r["frequency_ghz"] for r in simulation_result["results"]]
        reductions = [r["radar_reduction"] for r in simulation_result["results"]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, reductions, 'b-', linewidth=2)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Radar Reduction (0-1)')
        plt.title(f'RAM Frequency Response: {simulation_result["material"]}')
        plt.grid(True)
        plt.show()
"""
Simulation model for plasma-radar interactions.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from src.stealth.plasma.plasma_system import PlasmaStealthSystem, PlasmaParameters
from src.stealth.plasma.plasma_generator import PlasmaGenerator, PlasmaControlSystem


class PlasmaRadarSimulator:
    """Simple simulator for plasma-radar interactions."""
    
    def __init__(self, plasma_system: PlasmaStealthSystem):
        """
        Initialize plasma-radar simulator.
        
        Args:
            plasma_system: Plasma stealth system to simulate
        """
        self.plasma_system = plasma_system
        
    def simulate_radar_reflection(self, 
                                 radar_frequency: float,
                                 radar_power: float,
                                 plasma_power_levels: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
                                 altitude: float = 5000.0) -> Dict[str, Any]:
        """
        Simulate radar reflection at different plasma power levels.
        
        Args:
            radar_frequency: Radar frequency in GHz
            radar_power: Radar power in W
            plasma_power_levels: List of plasma power levels to simulate
            altitude: Altitude in meters
            
        Returns:
            Simulation results
        """
        results = []
        
        # Environmental conditions
        env_conditions = {
            "temperature": 20.0,
            "altitude": altitude
        }
        
        # Threat data
        threat_data = {
            "frequency": radar_frequency,
            "power": radar_power
        }
        
        # Simulate for each power level
        for power_level in plasma_power_levels:
            # Update plasma system power level (simulation only)
            self.plasma_system.status["power_level"] = power_level
            self.plasma_system.status["active"] = True if power_level > 0 else False
            
            # Calculate effectiveness
            effectiveness = self.plasma_system.calculate_effectiveness(threat_data, env_conditions)
            
            # Calculate plasma frequency
            plasma_density = self.plasma_system.plasma_params.density * power_level
            plasma_frequency = 8.98 * np.sqrt(plasma_density) / 1.0e9  # GHz
            
            results.append({
                "power_level": power_level,
                "rcs_reduction": effectiveness["rcs_reduction"],
                "detection_probability": effectiveness["detection_probability"],
                "plasma_frequency": plasma_frequency
            })
            
        return {
            "radar_frequency": radar_frequency,
            "radar_power": radar_power,
            "altitude": altitude,
            "results": results
        }
        
    def simulate_frequency_sweep(self,
                               min_freq: float = 1.0,
                               max_freq: float = 20.0,
                               steps: int = 20,
                               plasma_power: float = 0.8) -> Dict[str, Any]:
        """
        Simulate radar effectiveness across frequency range.
        
        Args:
            min_freq: Minimum radar frequency in GHz
            max_freq: Maximum radar frequency in GHz
            steps: Number of frequency steps
            plasma_power: Plasma power level
            
        Returns:
            Simulation results
        """
        frequencies = np.linspace(min_freq, max_freq, steps)
        results = []
        
        # Set plasma power level
        self.plasma_system.status["power_level"] = plasma_power
        self.plasma_system.status["active"] = True if plasma_power > 0 else False
        
        # Environmental conditions
        env_conditions = {
            "temperature": 20.0,
            "altitude": 5000.0
        }
        
        # Calculate plasma frequency
        plasma_density = self.plasma_system.plasma_params.density * plasma_power
        plasma_frequency = 8.98 * np.sqrt(plasma_density) / 1.0e9  # GHz
        
        # Simulate for each frequency
        for freq in frequencies:
            threat_data = {
                "frequency": freq,
                "power": 1000.0  # Fixed power
            }
            
            effectiveness = self.plasma_system.calculate_effectiveness(threat_data, env_conditions)
            
            results.append({
                "frequency": freq,
                "rcs_reduction": effectiveness["rcs_reduction"],
                "detection_probability": effectiveness["detection_probability"]
            })
            
        return {
            "plasma_power": plasma_power,
            "plasma_frequency": plasma_frequency,
            "frequency_range": (min_freq, max_freq),
            "results": results
        }
        
    def plot_frequency_response(self, simulation_result: Dict[str, Any]) -> None:
        """
        Plot frequency response simulation results.
        
        Args:
            simulation_result: Results from simulate_frequency_sweep
        """
        frequencies = [r["frequency"] for r in simulation_result["results"]]
        rcs_reduction = [r["rcs_reduction"] for r in simulation_result["results"]]
        detection_prob = [r["detection_probability"] for r in simulation_result["results"]]
        
        plasma_freq = simulation_result["plasma_frequency"]
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(frequencies, rcs_reduction, 'b-', linewidth=2)
        plt.axvline(x=plasma_freq, color='r', linestyle='--', label=f'Plasma Freq: {plasma_freq:.2f} GHz')
        plt.xlabel('Radar Frequency (GHz)')
        plt.ylabel('RCS Reduction (0-1)')
        plt.title(f'Plasma Stealth Effectiveness (Power: {simulation_result["plasma_power"]:.1f})')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(frequencies, detection_prob, 'g-', linewidth=2)
        plt.axvline(x=plasma_freq, color='r', linestyle='--', label=f'Plasma Freq: {plasma_freq:.2f} GHz')
        plt.xlabel('Radar Frequency (GHz)')
        plt.ylabel('Detection Probability')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
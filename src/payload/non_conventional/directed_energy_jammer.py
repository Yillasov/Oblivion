"""
Directed Energy Jammer implementation for UCAV platforms.
"""

from typing import Dict, List, Any, Optional, Set
import numpy as np

from src.payload.non_conventional.countermeasures import AdaptiveCountermeasure, CountermeasureSpecs
from src.payload.types import CountermeasureType, JammingFrequencyBand


class DirectedEnergyJammer(AdaptiveCountermeasure):
    """
    Directed energy jamming system that uses neuromorphic processing
    for adaptive jamming pattern generation and frequency targeting.
    """
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "DEJ-100":
            specs = CountermeasureSpecs(
                weight=120.0,
                volume={"length": 1.2, "width": 0.5, "height": 0.4},
                power_requirements=2500.0,  # 2.5 kW
                mounting_points=["fuselage", "wing"],
                countermeasure_type=CountermeasureType.DIRECTED_ENERGY_JAMMER,
                response_time=0.05,  # Very fast response time
                effectiveness_rating=0.92,
                capacity=50,  # Can be used many times
                coverage_angle=120.0,
                energy_consumption=2200.0,
                thermal_signature=0.6,
                stealth_impact=0.3,
                cooldown_time=0.5,
                frequency_bands={
                    JammingFrequencyBand.X_BAND, 
                    JammingFrequencyBand.KU_BAND,
                    JammingFrequencyBand.K_BAND
                },
                neuromorphic_processing_requirements={
                    "snn_neurons": 5000,
                    "learning_enabled": True,
                    "adaptation_rate": 0.8
                }
            )
        elif model == "DEJ-200":
            specs = CountermeasureSpecs(
                weight=180.0,
                volume={"length": 1.5, "width": 0.6, "height": 0.5},
                power_requirements=4000.0,  # 4 kW
                mounting_points=["fuselage", "internal_bay"],
                countermeasure_type=CountermeasureType.DIRECTED_ENERGY_JAMMER,
                response_time=0.03,
                effectiveness_rating=0.95,
                capacity=75,
                coverage_angle=180.0,
                energy_consumption=3800.0,
                thermal_signature=0.7,
                stealth_impact=0.4,
                cooldown_time=0.3,
                frequency_bands={
                    JammingFrequencyBand.X_BAND, 
                    JammingFrequencyBand.KU_BAND,
                    JammingFrequencyBand.K_BAND,
                    JammingFrequencyBand.KA_BAND,
                    JammingFrequencyBand.MILLIMETER
                },
                neuromorphic_processing_requirements={
                    "snn_neurons": 8000,
                    "learning_enabled": True,
                    "adaptation_rate": 0.9
                }
            )
        else:
            raise ValueError(f"Unknown jammer model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        self.active_frequencies = set()
        self.jamming_pattern = "adaptive"
        self.power_level = 100.0  # Percentage
        self.thermal_status = {
            "current_temperature": 25.0,
            "max_temperature": 95.0,
            "cooling_efficiency": 0.85
        }
    
    def target_frequency(self, frequency_band: JammingFrequencyBand) -> bool:
        """
        Target a specific frequency band for jamming.
        
        Args:
            frequency_band: Frequency band to target
            
        Returns:
            Success status
        """
        if frequency_band in self.specs.frequency_bands:
            self.active_frequencies.add(frequency_band)
            return True
        return False
    
    def set_jamming_pattern(self, pattern: str) -> bool:
        """
        Set the jamming pattern.
        
        Args:
            pattern: Jamming pattern (adaptive, pulse, continuous, barrage)
            
        Returns:
            Success status
        """
        valid_patterns = ["adaptive", "pulse", "continuous", "barrage"]
        if pattern in valid_patterns:
            self.jamming_pattern = pattern
            return True
        return False
    
    def set_power_level(self, power_level: float) -> bool:
        """
        Set the power level for the jammer.
        
        Args:
            power_level: Power level as a percentage (0-100)
            
        Returns:
            Success status
        """
        if 0 <= power_level <= 100:
            self.power_level = power_level
            return True
        return False
    
    def set_power(self, power_ratio: float) -> bool:
        """
        Set the power ratio for the jammer.
        
        Args:
            power_ratio: Power ratio (0.0-1.0)
            
        Returns:
            Success status
        """
        # Convert ratio to percentage
        return self.set_power_level(power_ratio * 100.0)

    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy jamming against a target threat.
        
        Args:
            target_data: Data about the target threat
            
        Returns:
            Success status
        """
        # First check if base deployment is successful
        if not super().deploy(target_data):
            return False
        
        # Use neuromorphic processing to optimize jamming
        jamming_result = self.process_data({
            "threat": target_data,
            "computation": "jamming_optimization",
            "active_frequencies": list(self.active_frequencies),
            "jamming_pattern": self.jamming_pattern,
            "power_level": self.power_level
        })
        
        # Update thermal status
        self._manage_thermal_load(jamming_result.get("duration", 5.0))
        
        return True
    
    def _manage_thermal_load(self, duration: float) -> None:
        """
        Manage thermal load during jamming.
        
        Args:
            duration: Duration of jamming in seconds
        """
        # Use neuromorphic processing for thermal management
        thermal_result = self.process_data({
            "power_level": self.power_level,
            "duration": duration,
            "computation": "thermal_management"
        })
        
        self.thermal_status["current_temperature"] = thermal_result.get("temperature", 25.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current jammer status."""
        status = super().get_status()
        status.update({
            "active_frequencies": [freq.name for freq in self.active_frequencies],
            "jamming_pattern": self.jamming_pattern,
            "power_level": self.power_level,
            "thermal_status": self.thermal_status
        })
        return status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using neuromorphic computing.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processing results
        """
        base_result = super().process_data(input_data)
        
        computation_type = input_data.get("computation", "")
        
        if computation_type == "jamming_optimization":
            # Neuromorphic jamming optimization
            threat_type = input_data.get("threat", {}).get("type", "unknown")
            
            # Adapt jamming based on threat
            if threat_type == "radar_guided":
                optimal_frequencies = [JammingFrequencyBand.X_BAND, JammingFrequencyBand.KU_BAND]
                optimal_pattern = "barrage"
            elif threat_type == "active_radar":
                optimal_frequencies = [JammingFrequencyBand.X_BAND]
                optimal_pattern = "adaptive"
            else:
                optimal_frequencies = list(self.specs.frequency_bands)[:2]
                optimal_pattern = "pulse"
            
            # Update jammer settings
            self.active_frequencies = set(optimal_frequencies) & self.specs.frequency_bands
            self.jamming_pattern = optimal_pattern
            
            # Calculate effectiveness
            effectiveness = base_result.get("effectiveness", 0.0)
            if self.active_frequencies and threat_type != "unknown":
                effectiveness += 0.15
            
            base_result["effectiveness"] = min(1.0, effectiveness)
            base_result["optimal_frequencies"] = [f.name for f in optimal_frequencies]
            base_result["optimal_pattern"] = optimal_pattern
            base_result["duration"] = input_data.get("threat", {}).get("duration", 5.0)
            
        elif computation_type == "thermal_management":
            # Neuromorphic thermal management
            power_level = input_data.get("power_level", 100.0)
            duration = input_data.get("duration", 5.0)
            
            # Calculate temperature increase based on power and duration
            base_temp = 25.0
            temp_increase = (power_level / 100.0) * duration * 2.0
            cooling_factor = self.thermal_status["cooling_efficiency"]
            
            new_temp = base_temp + (temp_increase * (1.0 - cooling_factor))
            
            base_result["temperature"] = min(new_temp, self.thermal_status["max_temperature"])
            base_result["cooling_efficiency"] = cooling_factor
            base_result["max_continuous_operation"] = (
                (self.thermal_status["max_temperature"] - base_temp) / 
                (temp_increase * (1.0 - cooling_factor))
            )
        
        return base_result
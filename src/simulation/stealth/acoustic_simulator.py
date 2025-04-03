"""
Acoustic Signature Simulation Module

Provides capabilities to simulate and analyze acoustic signatures
of UCAV platforms, including propulsion noise and airframe effects.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.stealth.acoustic.acoustic_reduction import AcousticReductionSystem


class FrequencyRange(Enum):
    """Acoustic frequency ranges."""
    LOW = "low"        # 20-500 Hz
    MID = "mid"        # 500-2000 Hz
    HIGH = "high"      # 2000-20000 Hz
    FULL = "full"      # 20-20000 Hz (full audible range)


@dataclass
class AcousticSimConfig:
    """Configuration for acoustic signature simulation."""
    frequency_range: FrequencyRange = FrequencyRange.FULL
    distance: float = 100.0  # Distance in meters for measurement
    include_propulsion_noise: bool = True
    include_airframe_noise: bool = True
    include_atmospheric_effects: bool = True
    ambient_noise: float = 30.0  # Ambient noise level in dB


class AcousticSignatureSimulator:
    """Simulates acoustic signatures of UCAV platforms."""
    
    def __init__(self, config: AcousticSimConfig):
        """Initialize acoustic signature simulator."""
        self.config = config
        self.platform_data: Dict[str, Any] = {}
        self.acoustic_systems: Dict[str, AcousticReductionSystem] = {}
        
    def register_platform(self, platform_data: Dict[str, Any]) -> None:
        """Register platform data for acoustic simulation."""
        self.platform_data = platform_data
        
    def register_acoustic_system(self, system_id: str, system: AcousticReductionSystem) -> None:
        """Register an acoustic reduction system."""
        self.acoustic_systems[system_id] = system
        
    def calculate_signature(self, 
                           platform_state: Dict[str, Any],
                           environmental_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate acoustic signature for given platform state and conditions.
        
        Args:
            platform_state: Current platform state including propulsion
            environmental_conditions: Current environmental conditions
            
        Returns:
            Dictionary with acoustic signature results
        """
        # Extract relevant data
        speed = platform_state.get("speed", 0.0)
        altitude = platform_state.get("altitude", 0.0)
        propulsion_state = platform_state.get("propulsion", {})
        
        # Calculate propulsion noise
        propulsion_noise = self._calculate_propulsion_noise(propulsion_state)
        
        # Calculate airframe noise
        airframe_noise = self._calculate_airframe_noise(speed)
        
        # Calculate total base noise
        total_base_noise = self._combine_noise_levels([propulsion_noise, airframe_noise])
        
        # Apply acoustic reduction systems
        reduction_factor = self._calculate_reduction_factor()
        
        # Apply atmospheric attenuation
        atmospheric_factor = self._calculate_atmospheric_factor(altitude, environmental_conditions)
        
        # Calculate final noise level
        final_noise = self.config.ambient_noise + (total_base_noise - self.config.ambient_noise) * reduction_factor * atmospheric_factor
        
        # Calculate detection ranges
        detection_ranges = self._calculate_detection_ranges(final_noise)
        
        return {
            "total_signature_db": final_noise,
            "components": {
                "propulsion_noise_db": propulsion_noise,
                "airframe_noise_db": airframe_noise,
                "ambient_noise_db": self.config.ambient_noise
            },
            "factors": {
                "reduction": reduction_factor,
                "atmospheric": atmospheric_factor
            },
            "frequency_range": self.config.frequency_range.value,
            "detection_ranges": detection_ranges
        }
        
    def _calculate_propulsion_noise(self, propulsion_state: Dict[str, Any]) -> float:
        """Calculate noise from propulsion systems."""
        if not self.config.include_propulsion_noise or not propulsion_state:
            return 0.0
            
        total_noise = 0.0
        
        # Process each propulsion system
        for system_id, state in propulsion_state.items():
            # Get power level
            power_level = state.get("power_level", 0.0)
            
            # Base noise level depends on power
            # Simplified model: 60dB at idle, up to 120dB at max power
            system_noise = 60.0 + (power_level * 60.0)
            
            # Add to total (logarithmic addition)
            if total_noise == 0.0:
                total_noise = system_noise
            else:
                total_noise = 10 * np.log10(10**(total_noise/10) + 10**(system_noise/10))
            
        return total_noise
        
    def _calculate_airframe_noise(self, speed: float) -> float:
        """Calculate noise from airframe."""
        if not self.config.include_airframe_noise:
            return 0.0
            
        # Simplified model: airframe noise increases with speed
        # At low speeds (<100 m/s), minimal noise
        # At high speeds (>300 m/s), significant noise
        if speed < 100.0:
            return 40.0  # Minimal noise
        elif speed > 300.0:
            return 90.0  # Maximum airframe noise
        else:
            # Linear interpolation between 40dB and 90dB
            return 40.0 + ((speed - 100.0) / 200.0) * 50.0
        
    def _calculate_reduction_factor(self) -> float:
        """Calculate acoustic reduction factor from active systems."""
        if not self.acoustic_systems:
            return 1.0
            
        # Start with no reduction
        reduction_factor = 1.0
        
        # Apply each reduction system
        for system_id, system in self.acoustic_systems.items():
            # Check if system is active
            if system.status.get("active", False):
                # Get current attenuation in dB
                attenuation_db = system.status.get("current_attenuation", 0.0)
                
                # Convert dB to linear factor
                system_factor = 10**(-attenuation_db/20)
                
                # Apply the strongest reduction
                reduction_factor = min(reduction_factor, system_factor)
                
        return reduction_factor
        
    def _calculate_atmospheric_factor(self, altitude: float, environmental_conditions: Dict[str, Any]) -> float:
        """Calculate atmospheric effects on sound propagation."""
        if not self.config.include_atmospheric_effects:
            return 1.0
            
        # Extract relevant conditions
        humidity = environmental_conditions.get("humidity", 0.5)
        temperature = environmental_conditions.get("temperature", 15.0)
        wind_speed = np.linalg.norm(environmental_conditions.get("wind", np.array([0.0, 0.0, 0.0])))
        
        # Base attenuation factor (higher = more attenuation)
        # Distance is already factored into the base calculations
        attenuation_factor = 1.0
        
        # Humidity effects (higher humidity = more attenuation at high frequencies)
        if self.config.frequency_range in [FrequencyRange.HIGH, FrequencyRange.FULL]:
            attenuation_factor += humidity * 0.2
            
        # Temperature effects (sound travels differently in different temperatures)
        temp_factor = 1.0 + ((temperature - 15.0) / 100.0)  # Small adjustment based on temperature
        attenuation_factor *= temp_factor
        
        # Wind effects (simplified)
        if wind_speed > 5.0:
            attenuation_factor *= (1.0 + (wind_speed - 5.0) / 20.0)
            
        # Altitude effects (thinner air = less sound propagation)
        if altitude > 1000.0:
            altitude_factor = 1.0 + ((altitude - 1000.0) / 10000.0)
            attenuation_factor *= altitude_factor
            
        return attenuation_factor
        
    def _calculate_detection_ranges(self, noise_level: float) -> Dict[str, float]:
        """Calculate detection ranges for different listener types."""
        # Base detection range depends on noise level
        # Using inverse square law: sound drops by 6dB when distance doubles
        
        # Reference: 60dB at 100m is barely audible to humans
        base_range = self.config.distance * 10**((noise_level - 60.0) / 20.0)
        
        return {
            "human_hearing": base_range * 1.0,
            "basic_microphone": base_range * 2.0,
            "advanced_acoustic_sensor": base_range * 4.0,
            "military_acoustic_array": base_range * 8.0
        }
        
    def _combine_noise_levels(self, noise_levels: List[float]) -> float:
        """Combine multiple noise levels (logarithmic addition)."""
        if not noise_levels:
            return 0.0
            
        # Filter out zero values
        valid_levels = [level for level in noise_levels if level > 0.0]
        
        if not valid_levels:
            return 0.0
            
        # Convert to linear, sum, then back to logarithmic
        linear_sum = sum(10**(level/10) for level in valid_levels)
        return 10 * np.log10(linear_sum)
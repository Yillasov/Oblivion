#!/usr/bin/env python3
"""
Active noise cancellation system for acoustic signature reduction.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass

from src.stealth.acoustic.acoustic_reduction import AcousticReductionSystem, AcousticParameters
from src.simulation.stealth.acoustic_simulator import FrequencyRange


class NoiseCancellationMode(Enum):
    """Noise cancellation operational modes."""
    OFF = "off"
    PASSIVE = "passive"
    REACTIVE = "reactive"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"


@dataclass
class FrequencyBand:
    """Frequency band for targeted noise cancellation."""
    min_freq: float  # Minimum frequency in Hz
    max_freq: float  # Maximum frequency in Hz
    priority: float  # Priority level (0.0-1.0)
    attenuation: float  # Target attenuation in dB


class NoiseCancellationSystem:
    """
    Active noise cancellation system for acoustic signature reduction.
    Integrates with existing acoustic reduction systems to provide enhanced
    noise suppression capabilities.
    """
    
    def __init__(self, acoustic_system: Optional[AcousticReductionSystem] = None):
        """
        Initialize noise cancellation system.
        
        Args:
            acoustic_system: Optional acoustic reduction system to integrate with
        """
        self.acoustic_system = acoustic_system
        self.mode = NoiseCancellationMode.OFF
        self.active = False
        self.power_level = 0.0
        self.target_bands: List[FrequencyBand] = []
        self.current_attenuation = 0.0
        self.response_time = 0.02  # 20ms response time
        self.effectiveness = 0.0
        self.power_consumption = 0.0
        
        # Initialize default frequency bands
        self._initialize_default_bands()
        
    def _initialize_default_bands(self) -> None:
        """Initialize default frequency bands for noise cancellation."""
        self.target_bands = [
            FrequencyBand(20.0, 100.0, 0.9, 25.0),    # Low frequency (engine rumble)
            FrequencyBand(100.0, 500.0, 0.8, 20.0),   # Low-mid frequency
            FrequencyBand(500.0, 2000.0, 0.7, 15.0),  # Mid frequency
            FrequencyBand(2000.0, 8000.0, 0.5, 10.0), # High-mid frequency
            FrequencyBand(8000.0, 20000.0, 0.3, 5.0)  # High frequency
        ]
    
    def activate(self, mode: NoiseCancellationMode = NoiseCancellationMode.ADAPTIVE) -> Dict[str, Any]:
        """
        Activate noise cancellation system.
        
        Args:
            mode: Operational mode
            
        Returns:
            System status
        """
        self.mode = mode
        self.active = True
        
        # Configure system based on mode
        if mode == NoiseCancellationMode.PASSIVE:
            self.power_level = 0.2
            self.effectiveness = 0.3
            self.power_consumption = 0.5
        elif mode == NoiseCancellationMode.REACTIVE:
            self.power_level = 0.5
            self.effectiveness = 0.6
            self.power_consumption = 1.5
        elif mode == NoiseCancellationMode.ADAPTIVE:
            self.power_level = 0.7
            self.effectiveness = 0.8
            self.power_consumption = 2.5
        elif mode == NoiseCancellationMode.PREDICTIVE:
            self.power_level = 0.9
            self.effectiveness = 0.9
            self.power_consumption = 3.5
        elif mode == NoiseCancellationMode.EMERGENCY:
            self.power_level = 1.0
            self.effectiveness = 0.95
            self.power_consumption = 5.0
        else:  # OFF
            self.power_level = 0.0
            self.effectiveness = 0.0
            self.power_consumption = 0.0
            self.active = False
        
        # Integrate with acoustic system if available
        if self.acoustic_system and self.active:
            self.acoustic_system.status["active_damping_enabled"] = True
            self.acoustic_system.status["power_level"] = max(
                self.acoustic_system.status["power_level"],
                self.power_level * 0.8
            )
        
        return self.get_status()
    
    def deactivate(self) -> Dict[str, Any]:
        """
        Deactivate noise cancellation system.
        
        Returns:
            System status
        """
        self.active = False
        self.mode = NoiseCancellationMode.OFF
        self.power_level = 0.0
        self.effectiveness = 0.0
        self.power_consumption = 0.0
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            System status information
        """
        return {
            "active": self.active,
            "mode": self.mode.value,
            "power_level": self.power_level,
            "effectiveness": self.effectiveness,
            "power_consumption": self.power_consumption,
            "target_bands": len(self.target_bands),
            "response_time_ms": self.response_time * 1000,
            "integrated_with_acoustic_system": self.acoustic_system is not None
        }
    
    def add_frequency_band(self, band: FrequencyBand) -> bool:
        """
        Add a frequency band for targeted noise cancellation.
        
        Args:
            band: Frequency band to add
            
        Returns:
            Success status
        """
        # Validate frequency range
        if band.min_freq < 20.0 or band.max_freq > 20000.0 or band.min_freq >= band.max_freq:
            return False
            
        self.target_bands.append(band)
        return True
    
    def calculate_noise_reduction(self, 
                                noise_profile: Dict[str, float],
                                platform_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate noise reduction for a given noise profile.
        
        Args:
            noise_profile: Noise profile with frequency bands as keys and dB levels as values
            platform_state: Current platform state
            
        Returns:
            Noise reduction results
        """
        if not self.active:
            return {
                "total_reduction_db": 0.0,
                "effectiveness": 0.0,
                "bands_affected": 0,
                "status": "inactive"
            }
        
        # Extract relevant platform data
        speed = platform_state.get("speed", 0.0)
        altitude = platform_state.get("altitude", 0.0)
        
        # Calculate effectiveness adjustment based on conditions
        speed_factor = max(0.5, min(1.0, 1.0 - (speed / 300.0)))
        altitude_factor = min(1.2, 1.0 + (altitude / 10000.0))
        
        # Adjust effectiveness
        adjusted_effectiveness = self.effectiveness * speed_factor * altitude_factor
        
        # Calculate reduction for each frequency band
        total_energy_before = 0.0
        total_energy_after = 0.0
        bands_affected = 0
        
        for band in self.target_bands:
            # Find noise in this band
            band_noise = 0.0
            for freq_str, level in noise_profile.items():
                try:
                    freq = float(freq_str)
                    if band.min_freq <= freq <= band.max_freq:
                        band_noise = max(band_noise, level)
                except ValueError:
                    continue
            
            if band_noise > 0:
                bands_affected += 1
                
                # Calculate attenuation for this band
                band_attenuation = band.attenuation * adjusted_effectiveness * band.priority
                
                # Convert to energy domain
                energy_before = 10**(band_noise/10)
                energy_after = 10**((band_noise - band_attenuation)/10)
                
                total_energy_before += energy_before
                total_energy_after += energy_after
        
        # Calculate total reduction in dB
        if total_energy_before > 0:
            total_level_before = 10 * np.log10(total_energy_before)
            total_level_after = 10 * np.log10(max(1e-10, total_energy_after))
            total_reduction = total_level_before - total_level_after
        else:
            total_reduction = 0.0
        
        return {
            "total_reduction_db": total_reduction,
            "effectiveness": adjusted_effectiveness,
            "bands_affected": bands_affected,
            "status": "active",
            "mode": self.mode.value,
            "power_consumption": self.power_consumption
        }
    
    def optimize_for_signature(self, 
                             signature_profile: Dict[str, Any],
                             available_power: float) -> Dict[str, Any]:
        """
        Optimize noise cancellation for a specific acoustic signature.
        
        Args:
            signature_profile: Acoustic signature profile
            available_power: Available power in kW
            
        Returns:
            Optimization results
        """
        if not self.active or available_power <= 0:
            return {"success": False, "reason": "System inactive or no power available"}
        
        # Extract dominant frequencies from signature
        dominant_freqs = signature_profile.get("dominant_frequencies", {})
        
        # Sort frequencies by amplitude
        sorted_freqs = sorted(
            [(float(freq), level) for freq, level in dominant_freqs.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Adjust band priorities based on dominant frequencies
        for band in self.target_bands:
            band.priority = 0.3  # Reset to base priority
            
        # Increase priority for bands containing dominant frequencies
        for freq, level in sorted_freqs[:3]:  # Focus on top 3 dominant frequencies
            for band in self.target_bands:
                if band.min_freq <= freq <= band.max_freq:
                    band.priority = min(1.0, band.priority + 0.2)
        
        # Adjust power level based on available power
        max_power = 5.0  # Maximum power consumption in kW
        power_ratio = min(1.0, available_power / max_power)
        
        self.power_level = power_ratio
        self.power_consumption = max_power * power_ratio
        self.effectiveness = min(0.95, 0.3 + (0.65 * power_ratio))
        
        # Select appropriate mode based on power level
        if power_ratio >= 0.8:
            self.mode = NoiseCancellationMode.PREDICTIVE
        elif power_ratio >= 0.6:
            self.mode = NoiseCancellationMode.ADAPTIVE
        elif power_ratio >= 0.4:
            self.mode = NoiseCancellationMode.REACTIVE
        else:
            self.mode = NoiseCancellationMode.PASSIVE
        
        return {
            "success": True,
            "optimized_bands": len(self.target_bands),
            "power_level": self.power_level,
            "effectiveness": self.effectiveness,
            "mode": self.mode.value,
            "power_consumption": self.power_consumption
        }
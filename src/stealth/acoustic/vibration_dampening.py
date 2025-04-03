#!/usr/bin/env python3
"""
Vibration dampening technologies for acoustic signature reduction.
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

from src.stealth.acoustic.noise_cancellation import NoiseCancellationSystem, FrequencyBand
from src.stealth.acoustic.acoustic_reduction import AcousticReductionSystem


class DampeningMaterial(Enum):
    """Materials used for vibration dampening."""
    VISCOELASTIC = "viscoelastic"
    COMPOSITE = "composite"
    METAMATERIAL = "metamaterial"
    AEROGEL = "aerogel"
    CERAMIC = "ceramic"


@dataclass
class DampeningConfig:
    """Configuration for vibration dampening system."""
    material: DampeningMaterial
    thickness_mm: float
    coverage_percent: float
    weight_kg: float
    resonant_frequency: float


class VibrationDampeningSystem:
    """
    Vibration dampening system for reducing mechanical vibrations
    that contribute to acoustic signatures.
    """
    
    def __init__(self, 
                noise_cancellation: Optional[NoiseCancellationSystem] = None,
                acoustic_system: Optional[AcousticReductionSystem] = None):
        """
        Initialize vibration dampening system.
        
        Args:
            noise_cancellation: Optional noise cancellation system
            acoustic_system: Optional acoustic reduction system
        """
        self.noise_cancellation = noise_cancellation
        self.acoustic_system = acoustic_system
        self.active = False
        self.dampening_configs: Dict[str, DampeningConfig] = {}
        self.effectiveness = 0.0
        self.total_weight = 0.0
        
    def add_dampening_zone(self, zone_id: str, config: DampeningConfig) -> bool:
        """
        Add a dampening zone to the system.
        
        Args:
            zone_id: Zone identifier (e.g., "engine_mount", "wing_root")
            config: Dampening configuration
            
        Returns:
            Success status
        """
        if zone_id in self.dampening_configs:
            return False
            
        self.dampening_configs[zone_id] = config
        self.total_weight += config.weight_kg
        
        # Recalculate overall effectiveness
        self._calculate_effectiveness()
        
        return True
    
    def _calculate_effectiveness(self) -> None:
        """Calculate overall dampening effectiveness."""
        if not self.dampening_configs:
            self.effectiveness = 0.0
            return
            
        # Calculate weighted effectiveness based on coverage and material
        total_coverage = sum(config.coverage_percent for config in self.dampening_configs.values())
        weighted_effectiveness = 0.0
        
        for config in self.dampening_configs.values():
            # Material effectiveness factors
            material_factor = {
                DampeningMaterial.VISCOELASTIC: 0.7,
                DampeningMaterial.COMPOSITE: 0.8,
                DampeningMaterial.METAMATERIAL: 0.9,
                DampeningMaterial.AEROGEL: 0.75,
                DampeningMaterial.CERAMIC: 0.65
            }.get(config.material, 0.5)
            
            # Thickness factor (thicker is better, up to a point)
            thickness_factor = min(1.0, config.thickness_mm / 10.0)
            
            # Weight factor (diminishing returns)
            weight_factor = min(1.0, 0.5 + (config.weight_kg / 20.0))
            
            # Combined effectiveness for this zone
            zone_effectiveness = material_factor * thickness_factor * weight_factor
            
            # Add to weighted total
            weighted_effectiveness += zone_effectiveness * (config.coverage_percent / 100.0)
        
        # Normalize by total coverage
        if total_coverage > 0:
            self.effectiveness = weighted_effectiveness / (total_coverage / 100.0)
        else:
            self.effectiveness = 0.0
    
    def calculate_vibration_reduction(self, 
                                    vibration_profile: Dict[str, float],
                                    platform_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate vibration reduction for a given vibration profile.
        
        Args:
            vibration_profile: Vibration profile with frequency bands as keys and amplitudes as values
            platform_state: Current platform state
            
        Returns:
            Vibration reduction results
        """
        if not self.active or not self.dampening_configs:
            return {
                "reduction_factor": 0.0,
                "effectiveness": 0.0,
                "zones_active": 0
            }
        
        # Extract relevant platform data
        speed = platform_state.get("speed", 0.0)
        altitude = platform_state.get("altitude", 0.0)
        
        # Calculate reduction for each frequency
        total_reduction = 0.0
        frequencies_affected = 0
        
        for freq_str, amplitude in vibration_profile.items():
            try:
                freq = float(freq_str)
                
                # Calculate reduction for this frequency
                freq_reduction = self._calculate_frequency_reduction(freq, amplitude)
                
                if freq_reduction > 0:
                    frequencies_affected += 1
                    total_reduction += freq_reduction
                    
            except ValueError:
                continue
        
        # Calculate average reduction
        avg_reduction = total_reduction / max(1, frequencies_affected)
        
        # Integrate with noise cancellation if available
        if self.noise_cancellation and self.noise_cancellation.active:
            # Create frequency bands for the noise cancellation system
            for freq_str, amplitude in vibration_profile.items():
                try:
                    freq = float(freq_str)
                    # Only add high-amplitude vibrations
                    if amplitude > 0.5:
                        band = FrequencyBand(
                            min_freq=freq * 0.9,
                            max_freq=freq * 1.1,
                            priority=min(1.0, amplitude),
                            attenuation=15.0
                        )
                        self.noise_cancellation.add_frequency_band(band)
                except ValueError:
                    continue
        
        return {
            "reduction_factor": avg_reduction,
            "effectiveness": self.effectiveness,
            "zones_active": len(self.dampening_configs),
            "total_weight_kg": self.total_weight,
            "integrated_with_noise_cancellation": self.noise_cancellation is not None and self.noise_cancellation.active
        }
    
    def _calculate_frequency_reduction(self, frequency: float, amplitude: float) -> float:
        """
        Calculate reduction factor for a specific frequency.
        
        Args:
            frequency: Vibration frequency in Hz
            amplitude: Vibration amplitude (0.0-1.0)
            
        Returns:
            Reduction factor (0.0-1.0)
        """
        total_reduction = 0.0
        
        for config in self.dampening_configs.values():
            # Calculate resonance factor (effectiveness drops near resonant frequency)
            resonance_factor = max(0.3, 1.0 - (0.7 / (1.0 + abs(frequency - config.resonant_frequency) / 10.0)))
            
            # Material effectiveness at this frequency
            material_factor = {
                DampeningMaterial.VISCOELASTIC: 0.8 if frequency < 500 else 0.6,
                DampeningMaterial.COMPOSITE: 0.7,
                DampeningMaterial.METAMATERIAL: 0.9 if 100 <= frequency <= 2000 else 0.7,
                DampeningMaterial.AEROGEL: 0.75 if frequency > 1000 else 0.6,
                DampeningMaterial.CERAMIC: 0.6 if frequency > 500 else 0.7
            }.get(config.material, 0.5)
            
            # Amplitude factor (more effective at higher amplitudes)
            amplitude_factor = 0.5 + (0.5 * amplitude)
            
            # Zone reduction
            zone_reduction = material_factor * resonance_factor * amplitude_factor * (config.coverage_percent / 100.0)
            total_reduction += zone_reduction
        
        # Normalize and cap at 0.95 (95% reduction)
        return min(0.95, total_reduction / len(self.dampening_configs))
    
    def activate(self) -> Dict[str, Any]:
        """
        Activate vibration dampening system.
        
        Returns:
            System status
        """
        self.active = True
        return self.get_status()
    
    def deactivate(self) -> Dict[str, Any]:
        """
        Deactivate vibration dampening system.
        
        Returns:
            System status
        """
        self.active = False
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            System status information
        """
        return {
            "active": self.active,
            "zones": len(self.dampening_configs),
            "effectiveness": self.effectiveness,
            "total_weight_kg": self.total_weight,
            "integrated_systems": {
                "noise_cancellation": self.noise_cancellation is not None,
                "acoustic_reduction": self.acoustic_system is not None
            }
        }
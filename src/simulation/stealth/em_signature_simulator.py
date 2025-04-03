"""
Electromagnetic Signature Simulation Module

Provides capabilities to simulate and analyze electromagnetic signatures
of UCAV platforms across various frequency bands.
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

from src.stealth.electromagnetic.emp_shielding import EMPShieldingSystem


class EMBand(Enum):
    """Electromagnetic frequency bands."""
    ELF = "elf"      # Extremely Low Frequency (3-30 Hz)
    VLF = "vlf"      # Very Low Frequency (3-30 kHz)
    LF = "lf"        # Low Frequency (30-300 kHz)
    MF = "mf"        # Medium Frequency (300 kHz-3 MHz)
    HF = "hf"        # High Frequency (3-30 MHz)
    VHF = "vhf"      # Very High Frequency (30-300 MHz)
    UHF = "uhf"      # Ultra High Frequency (300 MHz-3 GHz)
    SHF = "shf"      # Super High Frequency (3-30 GHz)
    EHF = "ehf"      # Extremely High Frequency (30-300 GHz)


@dataclass
class EMSignatureConfig:
    """Configuration for electromagnetic signature simulation."""
    em_band: EMBand = EMBand.UHF
    include_propulsion_effects: bool = True
    include_avionics_effects: bool = True
    include_communications_effects: bool = True
    include_atmospheric_effects: bool = True
    ambient_em_level: float = -90.0  # dBm


class EMSignatureSimulator:
    """Simulates electromagnetic signatures of UCAV platforms."""
    
    def __init__(self, config: EMSignatureConfig):
        """Initialize EM signature simulator."""
        self.config = config
        self.platform_data: Dict[str, Any] = {}
        self.em_systems: Dict[str, EMPShieldingSystem] = {}
        
    def register_platform(self, platform_data: Dict[str, Any]) -> None:
        """Register platform data for EM simulation."""
        self.platform_data = platform_data
        
    def register_em_system(self, system_id: str, system: EMPShieldingSystem) -> None:
        """Register an EM shielding system."""
        self.em_systems[system_id] = system
        
    def calculate_signature(self, 
                           platform_state: Dict[str, Any],
                           environmental_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate EM signature for given platform state and conditions.
        
        Args:
            platform_state: Current platform state including propulsion and avionics
            environmental_conditions: Current environmental conditions
            
        Returns:
            Dictionary with EM signature results
        """
        # Extract relevant data
        altitude = platform_state.get("altitude", 0.0)
        propulsion_state = platform_state.get("propulsion", {})
        avionics_state = platform_state.get("avionics", {})
        comms_state = platform_state.get("communications", {})
        
        # Calculate propulsion EM emissions
        propulsion_emissions = self._calculate_propulsion_emissions(propulsion_state)
        
        # Calculate avionics EM emissions
        avionics_emissions = self._calculate_avionics_emissions(avionics_state)
        
        # Calculate communications EM emissions
        comms_emissions = self._calculate_comms_emissions(comms_state)
        
        # Calculate total base emissions
        total_base_emissions = self._combine_emissions([propulsion_emissions, avionics_emissions, comms_emissions])
        
        # Apply EM shielding systems
        shielding_factor = self._calculate_shielding_factor()
        
        # Apply atmospheric attenuation
        atmospheric_factor = self._calculate_atmospheric_factor(altitude, environmental_conditions)
        
        # Calculate final emission level
        final_emissions = self.config.ambient_em_level + (total_base_emissions - self.config.ambient_em_level) * shielding_factor * atmospheric_factor
        
        # Calculate detection ranges
        detection_ranges = self._calculate_detection_ranges(final_emissions)
        
        return {
            "total_signature_dbm": final_emissions,
            "components": {
                "propulsion_emissions_dbm": propulsion_emissions,
                "avionics_emissions_dbm": avionics_emissions,
                "comms_emissions_dbm": comms_emissions,
                "ambient_em_dbm": self.config.ambient_em_level
            },
            "factors": {
                "shielding": shielding_factor,
                "atmospheric": atmospheric_factor
            },
            "em_band": self.config.em_band.value,
            "detection_ranges": detection_ranges
        }
        
    def _calculate_propulsion_emissions(self, propulsion_state: Dict[str, Any]) -> float:
        """Calculate EM emissions from propulsion systems."""
        if not self.config.include_propulsion_effects or not propulsion_state:
            return self.config.ambient_em_level
            
        total_emissions = self.config.ambient_em_level
        
        # Process each propulsion system
        for system_id, state in propulsion_state.items():
            # Get power level
            power_level = state.get("power_level", 0.0)
            
            # Base emission level depends on power
            # Simplified model: -80dBm at idle, up to -40dBm at max power
            system_emissions = -80.0 + (power_level * 40.0)
            
            # Add to total (logarithmic addition)
            total_emissions = 10 * np.log10(10**(total_emissions/10) + 10**(system_emissions/10))
            
        return total_emissions
        
    def _calculate_avionics_emissions(self, avionics_state: Dict[str, Any]) -> float:
        """Calculate EM emissions from avionics systems."""
        if not self.config.include_avionics_effects or not avionics_state:
            return self.config.ambient_em_level
            
        total_emissions = self.config.ambient_em_level
        
        # Process each avionics system
        for system_id, state in avionics_state.items():
            # Get active status
            active = state.get("active", False)
            
            if active:
                # Different systems have different emission levels
                if "radar" in system_id.lower():
                    system_emissions = -30.0  # Radar has high emissions
                elif "computer" in system_id.lower():
                    system_emissions = -70.0  # Computers have moderate emissions
                else:
                    system_emissions = -80.0  # Other systems have lower emissions
                
                # Add to total (logarithmic addition)
                total_emissions = 10 * np.log10(10**(total_emissions/10) + 10**(system_emissions/10))
            
        return total_emissions
        
    def _calculate_comms_emissions(self, comms_state: Dict[str, Any]) -> float:
        """Calculate EM emissions from communication systems."""
        if not self.config.include_communications_effects or not comms_state:
            return self.config.ambient_em_level
            
        total_emissions = self.config.ambient_em_level
        
        # Process each communication system
        for system_id, state in comms_state.items():
            # Get active status and power
            active = state.get("active", False)
            power = state.get("power_level", 0.0)
            
            if active:
                # Communications have high emissions when active
                system_emissions = -60.0 + (power * 30.0)  # Up to -30dBm at max power
                
                # Add to total (logarithmic addition)
                total_emissions = 10 * np.log10(10**(total_emissions/10) + 10**(system_emissions/10))
            
        return total_emissions
        
    def _calculate_shielding_factor(self) -> float:
        """Calculate EM shielding factor from active systems."""
        if not self.em_systems:
            return 1.0
            
        # Start with no shielding
        shielding_factor = 1.0
        
        # Apply each shielding system
        for system_id, system in self.em_systems.items():
            # Check if system is active
            if system.status.get("active", False):
                # Get shielding integrity
                integrity = system.status.get("shielding_integrity", 1.0)
                
                # Calculate shielding (higher integrity = more shielding)
                system_shielding = 1.0 - (integrity * 0.9)  # Up to 90% reduction
                
                # Apply the strongest shielding
                shielding_factor = min(shielding_factor, system_shielding)
                
        return shielding_factor
        
    def _calculate_atmospheric_factor(self, altitude: float, environmental_conditions: Dict[str, Any]) -> float:
        """Calculate atmospheric effects on EM propagation."""
        if not self.config.include_atmospheric_effects:
            return 1.0
            
        # Extract relevant conditions
        humidity = environmental_conditions.get("humidity", 0.5)
        precipitation = environmental_conditions.get("precipitation", 0.0)
        
        # Base attenuation increases with humidity and precipitation
        attenuation = 1.0 - (humidity * 0.1) - (precipitation * 0.3)
        
        # Altitude effects (thinner air = less attenuation)
        if altitude > 1000.0:
            altitude_factor = 1.0 + ((altitude - 1000.0) / 30000.0)  # Increase by up to 33% at high altitude
            attenuation *= altitude_factor
            
        # Frequency band effects
        if self.config.em_band in [EMBand.SHF, EMBand.EHF]:
            # Higher frequencies are more affected by atmosphere
            attenuation *= 0.8
        elif self.config.em_band in [EMBand.ELF, EMBand.VLF, EMBand.LF]:
            # Lower frequencies are less affected
            attenuation = min(1.0, attenuation * 1.2)
            
        return max(0.1, min(attenuation, 1.0))  # Clamp between 0.1 and 1.0
        
    def _calculate_detection_ranges(self, emission_level: float) -> Dict[str, float]:
        """Calculate detection ranges for different sensor types."""
        # Base detection range depends on emission level
        # Using simplified model: range proportional to emission level
        base_range = 10.0 * 10**((emission_level + 90.0) / 20.0)  # -90dBm reference
        
        return {
            "basic_em_detector": base_range * 1.0,
            "military_esm": base_range * 3.0,
            "advanced_sigint": base_range * 5.0,
            "satellite_collection": base_range * 10.0
        }
        
    def _combine_emissions(self, emission_levels: List[float]) -> float:
        """Combine multiple emission levels (logarithmic addition)."""
        if not emission_levels:
            return self.config.ambient_em_level
            
        # Filter out ambient values
        valid_levels = [level for level in emission_levels if level > self.config.ambient_em_level]
        
        if not valid_levels:
            return self.config.ambient_em_level
            
        # Convert to linear, sum, then back to logarithmic
        linear_sum = sum(10**(level/10) for level in valid_levels)
        return 10 * np.log10(linear_sum)
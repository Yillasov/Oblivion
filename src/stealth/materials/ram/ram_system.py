"""
Radar-Absorbent Materials (RAM) implementation for stealth systems.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.stealth.base.interfaces import StealthInterface, StealthSpecs, StealthType, NeuromorphicStealth
from src.stealth.base.config import StealthMaterialConfig, StealthSystemConfig


@dataclass
class RAMMaterial:
    """Properties of radar-absorbent materials."""
    name: str
    density: float  # kg/m³
    thickness: float  # mm
    frequency_response: Dict[str, float]  # Attenuation in dB by frequency (GHz)
    temperature_range: Tuple[float, float]  # Operating temperature range (°C)
    weather_resistance: float  # 0-1 scale
    durability: float  # 0-1 scale
    cost_factor: float  # Relative cost


class RAMSystem(NeuromorphicStealth):
    """Radar-Absorbent Material stealth system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """
        Initialize RAM stealth system.
        
        Args:
            config: System configuration
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.config = config
        self.materials_library = self._initialize_materials_library()
        self.active_material = None
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.RADAR_ABSORBENT_MATERIAL,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=self._calculate_base_rcs(),
            infrared_signature=1.0,  # RAM has minimal effect on IR
            acoustic_signature=1.0,  # RAM has minimal effect on acoustics
            activation_time=0.0,  # Passive system, instant activation
            operational_duration=float('inf'),  # Passive system, unlimited duration
            cooldown_period=0.0,  # No cooldown needed
            material_composition=self._get_material_composition()
        )
        
        # Select initial material based on config
        if config.material_config:
            self.select_material(config.material_config.material_type)
    
    def _initialize_materials_library(self) -> Dict[str, RAMMaterial]:
        """Initialize the library of available RAM materials."""
        return {
            "ferrite_composite": RAMMaterial(
                name="Ferrite Composite",
                density=2200.0,
                thickness=3.0,
                frequency_response={
                    "1.0": 10.0,  # 1 GHz: 10 dB attenuation
                    "5.0": 15.0,  # 5 GHz: 15 dB attenuation
                    "10.0": 12.0,  # 10 GHz: 12 dB attenuation
                    "15.0": 8.0,   # 15 GHz: 8 dB attenuation
                },
                temperature_range=(-40.0, 120.0),
                weather_resistance=0.7,
                durability=0.8,
                cost_factor=1.0
            ),
            "advanced_composite": RAMMaterial(
                name="Advanced Composite",
                density=1800.0,
                thickness=5.0,
                frequency_response={
                    "1.0": 15.0,   # 1 GHz: 15 dB attenuation
                    "5.0": 22.0,   # 5 GHz: 22 dB attenuation
                    "10.0": 20.0,  # 10 GHz: 20 dB attenuation
                    "15.0": 18.0,  # 15 GHz: 18 dB attenuation
                    "20.0": 15.0,  # 20 GHz: 15 dB attenuation
                },
                temperature_range=(-60.0, 150.0),
                weather_resistance=0.9,
                durability=0.85,
                cost_factor=2.5
            ),
            "carbon_nanotube": RAMMaterial(
                name="Carbon Nanotube Composite",
                density=1500.0,
                thickness=2.0,
                frequency_response={
                    "1.0": 18.0,   # 1 GHz: 18 dB attenuation
                    "5.0": 25.0,   # 5 GHz: 25 dB attenuation
                    "10.0": 30.0,  # 10 GHz: 30 dB attenuation
                    "15.0": 28.0,  # 15 GHz: 28 dB attenuation
                    "20.0": 22.0,  # 20 GHz: 22 dB attenuation
                },
                temperature_range=(-80.0, 200.0),
                weather_resistance=0.95,
                durability=0.9,
                cost_factor=5.0
            )
        }
    
    def _calculate_base_rcs(self) -> float:
        """Calculate base radar cross-section based on configuration."""
        if not self.config.material_config:
            return 1.0
            
        # Calculate RCS reduction based on material and coverage
        base_reduction = 0.0
        if self.config.signature_config:
            base_reduction = self.config.signature_config.radar_cross_section_reduction / 100.0
            
        # Adjust for coverage
        coverage = self.config.material_config.coverage_percentage / 100.0
        
        # Final RCS (lower is better)
        return 1.0 - (base_reduction * coverage)
    
    def _get_material_composition(self) -> Dict[str, float]:
        """Get material composition percentages."""
        if not self.config.material_config:
            return {"unknown": 100.0}
            
        return {self.config.material_config.material_type: 100.0}
    
    def select_material(self, material_name: str) -> bool:
        """
        Select a specific RAM material.
        
        Args:
            material_name: Name of the material to select
            
        Returns:
            Success status
        """
        if material_name in self.materials_library:
            self.active_material = self.materials_library[material_name]
            return True
        return False
    
    def initialize(self) -> bool:
        """Initialize the RAM system."""
        self.initialized = True
        self.status["active"] = True
        self.status["mode"] = "active"  # RAM is always active once installed
        return True
    
    def get_specifications(self) -> StealthSpecs:
        """Get the physical specifications of the stealth system."""
        return self.specs
    
    def calculate_effectiveness(self, 
                              threat_data: Dict[str, Any],
                              environmental_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate RAM effectiveness against specific threats under given conditions.
        
        Args:
            threat_data: Information about the threat (radar type, frequency, etc.)
            environmental_conditions: Environmental conditions (temperature, humidity, etc.)
            
        Returns:
            Dictionary of effectiveness metrics
        """
        # Extract threat information
        radar_frequency = threat_data.get("frequency", 10.0)  # Default to 10 GHz
        radar_power = threat_data.get("power", 1000.0)  # Default to 1000W
        
        # Extract environmental conditions
        temperature = environmental_conditions.get("temperature", 20.0)  # °C
        humidity = environmental_conditions.get("humidity", 50.0)  # %
        precipitation = environmental_conditions.get("precipitation", 0.0)  # mm/h
        
        # Base effectiveness from material properties
        if not self.active_material:
            return {"rcs_reduction": 0.0, "detection_probability": 1.0}
        
        # Get attenuation for the closest frequency
        frequency_str = str(float(int(radar_frequency)))
        attenuation = 0.0
        
        # Find closest frequency in material specs
        available_freqs = [float(f) for f in self.active_material.frequency_response.keys()]
        closest_freq = min(available_freqs, key=lambda x: abs(x - radar_frequency))
        attenuation = self.active_material.frequency_response[str(closest_freq)]
        
        # Adjust for environmental factors
        if temperature < self.active_material.temperature_range[0] or temperature > self.active_material.temperature_range[1]:
            # Outside temperature range reduces effectiveness
            temp_factor = 0.7
        else:
            temp_factor = 1.0
            
        # Precipitation reduces effectiveness
        if precipitation > 0:
            precip_factor = 1.0 - (0.2 * min(precipitation / 10.0, 1.0))
        else:
            precip_factor = 1.0
            
        # Calculate final effectiveness
        effectiveness = attenuation * temp_factor * precip_factor
        
        # Convert to RCS reduction (0-1 scale, lower is better)
        rcs_reduction = min(effectiveness / 30.0, 0.95)  # Cap at 95% reduction
        
        # Calculate detection probability - Fix for None material_config
        coverage_percentage = 100.0  # Default value
        if self.config.material_config is not None:
            coverage_percentage = self.config.material_config.coverage_percentage
        
        detection_probability = 1.0 - (rcs_reduction * coverage_percentage / 100.0)
        
        return {
            "rcs_reduction": rcs_reduction,
            "detection_probability": detection_probability,
            "effective_attenuation_db": effectiveness
        }
    
    def activate(self, activation_params: Dict[str, Any] = {}) -> bool:
        """
        Activate the RAM system (passive, always active).
        
        Args:
            activation_params: Not used for RAM systems
            
        Returns:
            Always True for RAM systems
        """
        self.status["active"] = True
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the RAM system (not applicable, always active).
        
        Returns:
            Always False for RAM systems
        """
        # RAM systems cannot be deactivated once installed
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Adjust operational parameters of the stealth system.
        
        Args:
            parameters: New parameters to set
            
        Returns:
            Success status
        """
        # RAM systems have limited adjustable parameters
        if "material" in parameters:
            return self.select_material(parameters["material"])
        return False
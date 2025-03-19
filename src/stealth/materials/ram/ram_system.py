"""
Radar-Absorbent Materials (RAM) implementation for stealth systems.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.stealth.base.interfaces import StealthInterface, StealthSpecs, StealthType, NeuromorphicStealth
from src.stealth.base.config import StealthMaterialConfig, StealthSystemConfig
from src.stealth.materials.ram.material_database import RAMMaterialDatabase


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
        self.material_database = RAMMaterialDatabase()
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
        # Now we use the material database instead of hardcoding materials
        return self.material_database.materials
        
    def _calculate_base_rcs(self) -> float:
        """Calculate base radar cross-section reduction."""
        if not self.active_material:
            return 1.0
            
        # Calculate average attenuation across frequencies
        attenuations = list(self.active_material.frequency_response.values())
        avg_attenuation = sum(attenuations) / len(attenuations) if attenuations else 0
        
        # Convert dB to linear scale (0-1 where 0 is perfect absorption)
        return 10 ** (-avg_attenuation / 20)
        
    def _get_material_composition(self) -> Dict[str, Any]:
        """Get material composition for specifications."""
        if not self.active_material:
            return {}
            
        return {
            "material": self.active_material.name,
            "thickness_mm": self.active_material.thickness,
            "density_kg_m3": self.active_material.density
        }
        
    def select_material(self, material_name: str) -> bool:
        """
        Select a specific RAM material.
        
        Args:
            material_name: Name of the material to select
            
        Returns:
            Success status
        """
        material = self.material_database.get_material(material_name)
        if material:
            self.active_material = material
            return True
        return False
        
    def get_available_materials(self) -> List[str]:
        """
        Get list of available RAM materials.
        
        Returns:
            List of material IDs
        """
        return self.material_database.list_materials()
        
    def get_material_properties(self, material_id: str) -> Dict[str, Any]:
        """
        Get properties of a specific material.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Dictionary of material properties
        """
        return self.material_database.get_material_properties(material_id)
        
    def find_optimal_material(self, 
                            threat_frequency: float,
                            environmental_conditions: Dict[str, float]) -> str:
        """
        Find the optimal material for given threat and conditions.
        
        Args:
            threat_frequency: Threat radar frequency in GHz
            environmental_conditions: Environmental conditions
            
        Returns:
            ID of the optimal material
        """
        return self.material_database.get_optimal_material(
            threat_frequency, environmental_conditions)
            
    def calculate_effectiveness(self, 
                              threat_data: Dict[str, Any],
                              environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate stealth effectiveness against specific threats.
        
        Args:
            threat_data: Threat data
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of effectiveness metrics
        """
        if not self.active_material:
            return {"radar_reduction": 0.0}
            
        # Extract threat frequency
        threat_frequency = threat_data.get("radar_frequency_ghz", 10.0)
        
        # Find closest frequency in the response data
        closest_freq = min(self.active_material.frequency_response.keys(), 
                          key=lambda x: abs(float(x) - threat_frequency))
        
        # Get attenuation at that frequency
        attenuation_db = self.active_material.frequency_response.get(closest_freq, 0.0)
        
        # Apply environmental factors
        temperature = environmental_conditions.get("temperature", 20.0)
        humidity = environmental_conditions.get("humidity", 50.0)
        
        # Check if temperature is within operating range
        temp_range = self.active_material.temperature_range
        if temperature < temp_range[0] or temperature > temp_range[1]:
            temperature_factor = 0.7  # Reduced effectiveness outside operating range
        else:
            # Calculate how optimal the temperature is
            optimal_temp = (temp_range[0] + temp_range[1]) / 2
            temp_deviation = abs(temperature - optimal_temp) / (temp_range[1] - temp_range[0])
            temperature_factor = 1.0 - (temp_deviation * 0.3)  # At most 30% reduction
            
        # Apply humidity factor
        humidity_factor = 1.0
        if humidity > 80.0:
            humidity_factor = self.active_material.weather_resistance
            
        # Calculate final effectiveness
        modified_attenuation = attenuation_db * temperature_factor * humidity_factor
        
        # Convert dB to linear scale (0-1 where 0 is perfect absorption)
        radar_reduction = 1.0 - (10 ** (-modified_attenuation / 20))
        
        return {
            "radar_reduction": radar_reduction,
            "temperature_factor": temperature_factor,
            "humidity_factor": humidity_factor,
            "attenuation_db": modified_attenuation
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
        Deactivate the RAM system (passive, always active).
        
        Returns:
            Always True for RAM systems
        """
        # RAM is passive, so it's always "active" in a sense
        self.status["active"] = True
        return True
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get system status.
        
        Returns:
            Dictionary of system status
        """
        status = {
            "active": True,  # RAM is always active
            "system_type": "radar_absorbent_material",
            "material": self.active_material.name if self.active_material else "None",
            "thickness_mm": self.active_material.thickness if self.active_material else 0.0,
            "temperature_range": self.active_material.temperature_range if self.active_material else (0.0, 0.0),
            "average_attenuation_db": 0.0
        }
        
        # Calculate average attenuation if material is selected
        if self.active_material:
            attenuations = list(self.active_material.frequency_response.values())
            status["average_attenuation_db"] = sum(attenuations) / len(attenuations) if attenuations else 0
            
        return status
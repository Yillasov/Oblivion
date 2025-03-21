"""
Nano-Chaff Cloud System implementation for UCAV platforms.
"""

from typing import Dict, List, Any, Optional, Set
import numpy as np

from src.payload.non_conventional.countermeasures import AdaptiveCountermeasure, CountermeasureSpecs
from src.payload.types import CountermeasureType, ChaffType


class NanoChaffCloud(AdaptiveCountermeasure):
    """
    Advanced nano-chaff cloud system that uses neuromorphic processing
    for optimal deployment timing and pattern generation.
    """
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "NCC-50":
            specs = CountermeasureSpecs(
                weight=35.0,
                volume={"length": 0.6, "width": 0.3, "height": 0.2},
                power_requirements=150.0,
                mounting_points=["fuselage", "wing"],
                countermeasure_type=CountermeasureType.NANO_CHAFF_CLOUD,
                response_time=0.1,
                effectiveness_rating=0.88,
                capacity=12,  # Number of deployments
                coverage_angle=360.0,
                energy_consumption=120.0,
                thermal_signature=0.2,
                stealth_impact=0.15,
                cooldown_time=3.0,
                chaff_type=ChaffType.NANO
            )
        elif model == "NCC-100":
            specs = CountermeasureSpecs(
                weight=45.0,
                volume={"length": 0.7, "width": 0.35, "height": 0.25},
                power_requirements=200.0,
                mounting_points=["fuselage", "wing", "internal_bay"],
                countermeasure_type=CountermeasureType.NANO_CHAFF_CLOUD,
                response_time=0.08,
                effectiveness_rating=0.92,
                capacity=18,
                coverage_angle=360.0,
                energy_consumption=180.0,
                thermal_signature=0.25,
                stealth_impact=0.2,
                cooldown_time=2.5,
                chaff_type=ChaffType.SMART
            )
        else:
            raise ValueError(f"Unknown nano-chaff model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        self.cloud_properties = {
            "dispersion_pattern": "spherical",
            "particle_density": 1000000,  # Particles per cubic meter
            "persistence_time": 45.0,  # Seconds
            "radar_reflectivity": 0.95,
            "wind_resistance": 0.8,
            "active_duration": 0.0,  # Current active duration
            "current_effectiveness": 0.0
        }
        
        # Nano-specific properties
        if specs.chaff_type == ChaffType.NANO:
            self.nano_properties = {
                "particle_size": 0.0001,  # 100 microns
                "material": "carbon_nanotubes",
                "self_organizing": False
            }
        elif specs.chaff_type == ChaffType.SMART:
            self.nano_properties = {
                "particle_size": 0.0002,  # 200 microns
                "material": "graphene_composite",
                "self_organizing": True,
                "swarm_intelligence": True,
                "adaptive_frequency_response": True
            }
    
    def set_dispersion_pattern(self, pattern: str) -> bool:
        """
        Set the dispersion pattern for the chaff cloud.
        
        Args:
            pattern: Dispersion pattern (spherical, directional, trail)
            
        Returns:
            Success status
        """
        valid_patterns = ["spherical", "directional", "trail", "adaptive"]
        if pattern in valid_patterns:
            self.cloud_properties["dispersion_pattern"] = pattern
            return True
        return False
    
    def set_power_level(self, power_level: float) -> bool:
        """
        Set the power level for the chaff system.
        
        Args:
            power_level: Power level as a percentage (0-100)
            
        Returns:
            Success status
        """
        if 0 <= power_level <= 100:
            # Higher power means more particles and better dispersion
            base_density = 500000
            self.cloud_properties["particle_density"] = base_density * (1 + power_level/100)
            return True
        return False
    
    def set_power(self, power_ratio: float) -> bool:
        """
        Set the power ratio for the chaff system.
        
        Args:
            power_ratio: Power ratio (0.0-1.0)
            
        Returns:
            Success status
        """
        return self.set_power_level(power_ratio * 100.0)
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy nano-chaff cloud against a target threat.
        
        Args:
            target_data: Data about the target threat
            
        Returns:
            Success status
        """
        # First check if base deployment is successful
        if not super().deploy(target_data):
            return False
        
        # Use neuromorphic processing to optimize deployment
        deployment_result = self.process_data({
            "threat": target_data,
            "computation": "chaff_optimization",
            "dispersion_pattern": self.cloud_properties["dispersion_pattern"],
            "particle_density": self.cloud_properties["particle_density"],
            "nano_properties": self.nano_properties
        })
        
        # Update cloud properties based on optimization
        self.cloud_properties["dispersion_pattern"] = deployment_result.get(
            "optimal_pattern", self.cloud_properties["dispersion_pattern"])
        self.cloud_properties["active_duration"] = 0.0
        self.cloud_properties["current_effectiveness"] = deployment_result.get("effectiveness", 0.0)
        
        return True
    
    def update(self, dt: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update chaff cloud state over time.
        
        Args:
            dt: Time step in seconds
            environment_data: Environmental data (wind, etc.)
            
        Returns:
            Updated status
        """
        if not self.status["active"]:
            return self.get_status()
        
        # Update active duration
        self.cloud_properties["active_duration"] += dt
        
        # Calculate remaining effectiveness based on time
        persistence = self.cloud_properties["persistence_time"]
        elapsed = self.cloud_properties["active_duration"]
        
        if elapsed >= persistence:
            # Cloud has dissipated
            self.status["active"] = False
            self.cloud_properties["current_effectiveness"] = 0.0
        else:
            # Effectiveness decreases over time
            decay_factor = 1.0 - (elapsed / persistence) ** 0.5
            initial_effectiveness = self.cloud_properties["current_effectiveness"]
            self.cloud_properties["current_effectiveness"] = initial_effectiveness * decay_factor
            
            # Apply environmental effects if data provided
            if environment_data and "wind_speed" in environment_data:
                wind_effect = min(1.0, environment_data["wind_speed"] / 20.0)  # Normalize to 0-1
                wind_resistance = self.cloud_properties["wind_resistance"]
                
                # Stronger wind reduces effectiveness faster
                wind_decay = wind_effect * (1.0 - wind_resistance)
                self.cloud_properties["current_effectiveness"] *= (1.0 - wind_decay)
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current chaff cloud status."""
        status = super().get_status()
        status.update({
            "cloud_properties": self.cloud_properties,
            "nano_properties": self.nano_properties
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
        
        if computation_type == "chaff_optimization":
            # Neuromorphic chaff deployment optimization
            threat_type = input_data.get("threat", {}).get("type", "unknown")
            threat_direction = input_data.get("threat", {}).get("direction", [0, 0, 0])
            
            # Determine optimal pattern based on threat
            if threat_type == "radar_guided_missile":
                optimal_pattern = "directional"
            elif threat_type == "radar_lock":
                optimal_pattern = "spherical"
            elif threat_type == "pursuit":
                optimal_pattern = "trail"
            else:
                optimal_pattern = "adaptive"
            
            # Calculate effectiveness based on threat and chaff properties
            effectiveness = base_result.get("effectiveness", 0.0)
            
            # Smart chaff is more effective
            if self.specs.chaff_type == ChaffType.SMART:
                effectiveness += 0.15
            
            # Adjust for particle density
            density_factor = min(1.0, self.cloud_properties["particle_density"] / 1000000)
            effectiveness *= (0.7 + 0.3 * density_factor)
            
            base_result["effectiveness"] = min(1.0, effectiveness)
            base_result["optimal_pattern"] = optimal_pattern
            base_result["estimated_persistence"] = self.cloud_properties["persistence_time"]
            
        return base_result
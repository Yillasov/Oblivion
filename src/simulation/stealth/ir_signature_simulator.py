"""
Infrared Signature Simulation Module

Provides capabilities to simulate and analyze infrared signatures
of UCAV platforms, including propulsion heat effects.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.stealth.infrared.infrared_suppression import InfraredSuppressionSystem
from src.propulsion.stealth_integration import PropulsionStealthIntegrator


class IRBand(Enum):
    """Infrared spectrum bands."""
    NEAR_IR = "near"      # 0.75-1.4 μm
    SHORT_WAVE = "swir"   # 1.4-3 μm
    MID_WAVE = "mwir"     # 3-8 μm (most military IR sensors)
    LONG_WAVE = "lwir"    # 8-15 μm
    FAR_IR = "far"        # 15-1000 μm


@dataclass
class IRSignatureConfig:
    """Configuration for IR signature simulation."""
    ir_band: IRBand = IRBand.MID_WAVE
    resolution: float = 0.5  # Meters per pixel
    include_propulsion_effects: bool = True
    include_surface_heating: bool = True
    include_atmospheric_effects: bool = True
    ambient_temperature: float = 20.0  # °C


class IRSignatureSimulator:
    """Simulates infrared signatures of UCAV platforms."""
    
    def __init__(self, config: IRSignatureConfig):
        """Initialize IR signature simulator."""
        self.config = config
        self.platform_geometry: Dict[str, Any] = {}
        self.material_properties: Dict[str, Dict[str, float]] = {}
        self.ir_suppression_systems: Dict[str, InfraredSuppressionSystem] = {}
        self.propulsion_integrator: Optional[PropulsionStealthIntegrator] = None
        
    def register_platform_geometry(self, geometry_data: Dict[str, Any]) -> None:
        """Register platform geometry for IR simulation."""
        self.platform_geometry = geometry_data
        
    def register_material(self, material_id: str, properties: Dict[str, float]) -> None:
        """
        Register material thermal properties.
        
        Args:
            material_id: Material identifier
            properties: Thermal properties including emissivity, conductivity, etc.
        """
        self.material_properties[material_id] = properties
        
    def register_ir_suppression(self, system_id: str, system: InfraredSuppressionSystem) -> None:
        """Register an IR suppression system."""
        self.ir_suppression_systems[system_id] = system
        
    def register_propulsion_integrator(self, integrator: PropulsionStealthIntegrator) -> None:
        """Register propulsion-stealth integrator."""
        self.propulsion_integrator = integrator
        
    def calculate_signature(self, 
                           platform_state: Dict[str, Any],
                           environmental_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate IR signature for given platform state and environmental conditions.
        
        Args:
            platform_state: Current platform state including propulsion
            environmental_conditions: Current environmental conditions
            
        Returns:
            Dictionary with IR signature results
        """
        # Extract relevant data
        ambient_temp = environmental_conditions.get("temperature", self.config.ambient_temperature)
        altitude = platform_state.get("altitude", 0.0)
        speed = platform_state.get("speed", 0.0)
        propulsion_state = platform_state.get("propulsion", {})
        
        # Base surface temperature (ambient + speed heating)
        surface_temp = ambient_temp + (speed * speed / 10000.0)  # Simple model for aerodynamic heating
        
        # Calculate propulsion heat signature
        propulsion_signature = self._calculate_propulsion_signature(propulsion_state, ambient_temp)
        
        # Calculate surface heat signature
        surface_signature = self._calculate_surface_signature(surface_temp, ambient_temp)
        
        # Apply IR suppression if available
        suppression_factor = self._calculate_suppression_factor(platform_state)
        
        # Apply atmospheric effects
        atmospheric_factor = self._calculate_atmospheric_factor(altitude, environmental_conditions)
        
        # Calculate total signature
        total_signature = (propulsion_signature + surface_signature) * suppression_factor * atmospheric_factor
        
        # Calculate detection ranges
        detection_ranges = self._calculate_detection_ranges(total_signature)
        
        return {
            "total_signature": total_signature,
            "components": {
                "propulsion": propulsion_signature,
                "surface": surface_signature
            },
            "factors": {
                "suppression": suppression_factor,
                "atmospheric": atmospheric_factor
            },
            "detection_ranges": detection_ranges,
            "ir_band": self.config.ir_band.value
        }
        
    def generate_ir_image(self, 
                        platform_state: Dict[str, Any],
                        environmental_conditions: Dict[str, Any],
                        view_angle: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """
        Generate a simulated IR image of the platform.
        
        Args:
            platform_state: Current platform state
            environmental_conditions: Current environmental conditions
            view_angle: (azimuth, elevation) in degrees
            
        Returns:
            2D numpy array representing IR intensity
        """
        # Get platform dimensions
        length = self.platform_geometry.get("length", 10.0)
        width = self.platform_geometry.get("width", 8.0)
        
        # Calculate image dimensions based on resolution
        resolution = self.config.resolution
        img_height = int(length / resolution)
        img_width = int(width / resolution)
        
        # Create base image (ambient temperature)
        ambient_temp = environmental_conditions.get("temperature", self.config.ambient_temperature)
        ir_image = np.ones((img_height, img_width)) * ambient_temp
        
        # Calculate signature components
        signature_data = self.calculate_signature(platform_state, environmental_conditions)
        
        # Apply propulsion heat sources
        if self.config.include_propulsion_effects:
            propulsion_state = platform_state.get("propulsion", {})
            self._apply_propulsion_to_image(ir_image, propulsion_state, view_angle)
        
        # Apply surface heating
        if self.config.include_surface_heating:
            speed = platform_state.get("speed", 0.0)
            self._apply_surface_heating_to_image(ir_image, speed, view_angle)
        
        # Apply IR suppression systems
        if self.ir_suppression_systems:
            self._apply_suppression_to_image(ir_image, platform_state)
        
        return ir_image
        
    def _calculate_propulsion_signature(self, propulsion_state: Dict[str, Any], ambient_temp: float) -> float:
        """Calculate IR signature from propulsion systems."""
        if not self.config.include_propulsion_effects or not propulsion_state:
            return 0.0
            
        total_signature = 0.0
        
        # Process each propulsion system
        for system_id, state in propulsion_state.items():
            # Get power level and temperature
            power_level = state.get("power_level", 0.0)
            temperature = state.get("temperature", ambient_temp + 100.0)
            
            # Calculate temperature difference from ambient
            temp_diff = temperature - ambient_temp
            
            # Simple model: signature proportional to temperature difference and power level
            system_signature = temp_diff * power_level * 0.1
            
            # Add to total
            total_signature += system_signature
            
        return total_signature
        
    def _calculate_surface_signature(self, surface_temp: float, ambient_temp: float) -> float:
        """Calculate IR signature from platform surface."""
        if not self.config.include_surface_heating:
            return 0.0
            
        # Calculate average emissivity from materials
        avg_emissivity = 0.9  # Default
        if self.material_properties:
            emissivities = [props.get("emissivity", 0.9) for props in self.material_properties.values()]
            avg_emissivity = sum(emissivities) / len(emissivities)
            
        # Calculate temperature difference
        temp_diff = surface_temp - ambient_temp
        
        # Simple model: signature proportional to temperature difference and emissivity
        return temp_diff * avg_emissivity * 0.05
        
    def _calculate_suppression_factor(self, platform_state: Dict[str, Any]) -> float:
        """Calculate IR suppression factor from active systems."""
        if not self.ir_suppression_systems:
            return 1.0
            
        # Start with no suppression
        suppression_factor = 1.0
        
        # Apply each suppression system
        for system_id, system in self.ir_suppression_systems.items():
            # Check if system is active
            if system.status.get("active", False):
                # Get power level
                power_level = system.status.get("power_level", 0.0)
                
                # Calculate suppression (higher power = more suppression)
                system_suppression = 1.0 - (power_level * 0.7)  # Up to 70% reduction
                
                # Apply the strongest suppression
                suppression_factor = min(suppression_factor, system_suppression)
                
        return suppression_factor
        
    def _calculate_atmospheric_factor(self, altitude: float, environmental_conditions: Dict[str, Any]) -> float:
        """Calculate atmospheric effects on IR propagation."""
        if not self.config.include_atmospheric_effects:
            return 1.0
            
        # Extract relevant conditions
        humidity = environmental_conditions.get("humidity", 0.5)
        precipitation = environmental_conditions.get("precipitation", 0.0)
        
        # Base attenuation increases with humidity and precipitation
        attenuation = 1.0 - (humidity * 0.2) - (precipitation * 0.5)
        
        # Altitude effects (thinner air = less attenuation)
        if altitude > 1000.0:
            altitude_factor = 1.0 + ((altitude - 1000.0) / 20000.0)  # Increase by up to 50% at high altitude
            attenuation *= altitude_factor
            
        return max(0.1, min(attenuation, 1.0))  # Clamp between 0.1 and 1.0
        
    def _calculate_detection_ranges(self, signature: float) -> Dict[str, float]:
        """Calculate detection ranges for different sensor types."""
        # Base detection range depends on signature
        base_range = signature * 10.0  # Simple scaling
        
        return {
            "short_range_ir": base_range * 0.5,  # Short range IR sensors
            "medium_range_ir": base_range * 0.3,  # Medium range IR sensors
            "long_range_ir": base_range * 0.15,   # Long range IR sensors
            "ir_missile_seeker": base_range * 0.4  # Missile seekers
        }
        
    def _apply_propulsion_to_image(self, 
                                 image: np.ndarray, 
                                 propulsion_state: Dict[str, Any],
                                 view_angle: Tuple[float, float]) -> None:
        """Apply propulsion heat sources to IR image."""
        img_height, img_width = image.shape
        
        # Simplified: add engine heat at rear of aircraft
        engine_x = int(img_height * 0.8)  # 80% back from nose
        engine_y = int(img_width * 0.5)  # Center
        
        # Get total power level
        total_power = 0.0
        for system_id, state in propulsion_state.items():
            total_power += state.get("power_level", 0.0)
            
        # Scale for multiple engines
        engine_count = len(propulsion_state)
        if engine_count > 0:
            total_power /= engine_count
            
        # Apply heat source with gaussian distribution
        radius = int(img_width * 0.1)  # 10% of width
        intensity = 100.0 * total_power  # Base temperature increase
        
        y, x = np.ogrid[-engine_y:img_width-engine_y, -engine_x:img_height-engine_x]
        mask = x*x + y*y <= radius*radius
        
        # Add heat to image
        image[mask] += intensity
        
        # Add exhaust plume
        plume_length = int(img_height * 0.3 * total_power)  # 30% of length at max power
        for i in range(plume_length):
            # Decrease intensity with distance
            plume_intensity = intensity * (1.0 - (i / plume_length))
            
            # Position (behind engine)
            plume_x = min(img_height - 1, engine_x + i)
            
            # Apply heat in a cone shape
            cone_width = int(radius * (1.0 + (i / plume_length)))
            for j in range(max(0, engine_y - cone_width), min(img_width, engine_y + cone_width)):
                # Distance from center of plume
                dist = abs(j - engine_y)
                # Decrease intensity with distance from center
                pixel_intensity = plume_intensity * (1.0 - (dist / cone_width))
                # Add heat
                image[plume_x, j] += pixel_intensity
        
    def _apply_surface_heating_to_image(self, 
                                      image: np.ndarray, 
                                      speed: float,
                                      view_angle: Tuple[float, float]) -> None:
        """Apply surface heating to IR image."""
        img_height, img_width = image.shape
        
        # Leading edge heating increases with speed
        leading_edge_temp = speed * speed / 10000.0
        
        # Apply to front 20% of aircraft
        leading_edge_length = int(img_height * 0.2)
        
        for i in range(leading_edge_length):
            # Decrease intensity with distance from nose
            intensity = leading_edge_temp * (1.0 - (i / leading_edge_length))
            
            # Apply across width
            for j in range(img_width):
                # More heating at center, less at edges
                edge_factor = 1.0 - (abs(j - (img_width / 2)) / (img_width / 2)) * 0.5
                image[i, j] += intensity * edge_factor
        
    def _apply_suppression_to_image(self, image: np.ndarray, platform_state: Dict[str, Any]) -> None:
        """Apply IR suppression effects to IR image."""
        # Calculate overall suppression factor
        suppression_factor = self._calculate_suppression_factor(platform_state)
        
        # Apply suppression (reduce temperature difference from ambient)
        ambient_temp = self.config.ambient_temperature
        
        # For each pixel
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Calculate temperature difference from ambient
                temp_diff = image[i, j] - ambient_temp
                
                # Apply suppression to the difference
                suppressed_diff = temp_diff * suppression_factor
                
                # Set new temperature
                image[i, j] = ambient_temp + suppressed_diff
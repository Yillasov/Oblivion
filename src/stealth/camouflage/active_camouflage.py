#!/usr/bin/env python3
"""
Active Camouflage system implementation for Oblivion SDK.
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
from dataclasses import dataclass

from src.stealth.base.interfaces import NeuromorphicStealth, StealthSpecs, StealthType
from src.stealth.base.config import StealthSystemConfig, StealthOperationalMode


@dataclass
class CamouflageParameters:
    """Parameters for active camouflage operation."""
    adaptation_speed: float  # Speed of adaptation (0.0-1.0)
    color_range: Dict[str, Tuple[int, int, int]]  # RGB color ranges
    pattern_complexity: float  # Pattern complexity (0.0-1.0)
    power_level: float  # Power level (0.0-1.0)
    refresh_rate: float  # Refresh rate in Hz


class ActiveCamouflageSystem(NeuromorphicStealth):
    """Active Camouflage system implementation."""
    
    def __init__(self, config: StealthSystemConfig, hardware_interface=None):
        """
        Initialize Active Camouflage system.
        
        Args:
            config: System configuration
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(hardware_interface)
        self.config = config
        
        # Set up specifications
        self.specs = StealthSpecs(
            stealth_type=StealthType.ACTIVE_CAMOUFLAGE,
            weight=config.weight_kg,
            power_requirements=config.power_requirements_kw,
            radar_cross_section=1.0,  # Minimal effect on RCS
            infrared_signature=0.8,  # Some reduction in IR signature
            acoustic_signature=1.0,  # No effect on acoustics
            activation_time=config.activation_time_seconds,
            operational_duration=config.operational_duration_minutes,
            cooldown_period=config.cooldown_time_seconds / 60.0  # Convert to minutes
        )
        
        # Camouflage system specific parameters
        self.camouflage_params = CamouflageParameters(
            adaptation_speed=0.7,
            color_range={
                "desert": (210, 180, 140),
                "forest": (34, 139, 34),
                "urban": (128, 128, 128),
                "ocean": (0, 105, 148),
                "night": (25, 25, 25)
            },
            pattern_complexity=0.5,
            power_level=0.0,
            refresh_rate=30.0
        )
        
        # System status
        self.status = {
            "active": False,
            "mode": "standby",
            "power_level": 0.0,
            "current_environment": None,
            "adaptation_progress": 0.0,
            "remaining_operation_time": config.operational_duration_minutes,
            "cooldown_remaining": 0.0,
            "current_pattern": None
        }
        
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the active camouflage system."""
        self.initialized = True
        self.status["mode"] = "standby"
        return True
    
    def get_specifications(self) -> StealthSpecs:
        """Get the physical specifications of the stealth system."""
        return self.specs
    
    def calculate_effectiveness(self, 
                              threat_data: Dict[str, Any],
                              environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate active camouflage effectiveness against specific threats.
        
        Args:
            threat_data: Information about the threat
            environmental_conditions: Environmental conditions
            
        Returns:
            Dictionary of effectiveness metrics
        """
        if not self.status["active"]:
            return {"visual_reduction": 0.0, "detection_probability": 1.0}
        
        # Extract threat information
        threat_type = threat_data.get("type", "visual")
        threat_distance = threat_data.get("distance", 1000.0)  # meters
        
        # Extract environmental conditions
        light_level = environmental_conditions.get("light_level", 0.5)  # 0.0-1.0
        weather = environmental_conditions.get("weather", "clear")  # clear, rain, fog, etc.
        
        # Base effectiveness depends on the environment match
        if self.status["current_environment"] is None:
            base_effectiveness = 0.3  # Default effectiveness if no environment set
        else:
            # Higher effectiveness when properly adapted to environment
            base_effectiveness = 0.8 * self.status["adaptation_progress"]
        
        # Adjust for power level
        power_factor = self.status["power_level"]
        
        # Adjust for light level (more effective in low light)
        light_factor = 1.0 + (0.5 * (1.0 - light_level))
        
        # Adjust for weather (more effective in poor visibility)
        weather_factor = 1.0
        if weather in ["fog", "heavy_rain", "snow"]:
            weather_factor = 1.3
        elif weather in ["light_rain", "cloudy"]:
            weather_factor = 1.1
        
        # Adjust for distance (more effective at greater distances)
        distance_factor = min(1.0 + (threat_distance / 5000.0), 1.5)
        
        # Calculate final effectiveness
        effectiveness = min(base_effectiveness * power_factor * light_factor * weather_factor * distance_factor, 0.95)
        
        # Calculate detection probability
        detection_probability = 1.0 - (effectiveness * 0.9)  # Even perfect camouflage has some detection probability
        
        return {
            "visual_reduction": effectiveness,
            "detection_probability": detection_probability,
            "adaptation_level": self.status["adaptation_progress"],
            "effectiveness_factors": {
                "base": base_effectiveness,
                "power": power_factor,
                "light": light_factor,
                "weather": weather_factor,
                "distance": distance_factor
            }
        }
    
    def activate(self, activation_params: Dict[str, Any] = {}) -> bool:
        """
        Activate the active camouflage system.
        
        Args:
            activation_params: Parameters for activation
            
        Returns:
            Success status
        """
        if not self.initialized:
            return False
            
        if self.status["cooldown_remaining"] > 0:
            return False  # Still in cooldown
            
        # Set default parameters if none provided
        if activation_params is None:
            activation_params = {}
            
        # Extract activation parameters
        power_level = activation_params.get("power_level", 0.8)
        environment = activation_params.get("environment", "auto")
        pattern_complexity = activation_params.get("pattern_complexity", self.camouflage_params.pattern_complexity)
        
        # Update camouflage parameters
        self.camouflage_params.power_level = power_level
        self.camouflage_params.pattern_complexity = pattern_complexity
        
        # Update system status
        self.status["active"] = True
        self.status["mode"] = "active"
        self.status["power_level"] = power_level
        self.status["current_environment"] = environment
        self.status["adaptation_progress"] = 0.2  # Initial adaptation
        
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the active camouflage system.
        
        Returns:
            Success status
        """
        if not self.initialized or not self.status["active"]:
            return False
            
        # Update system status
        self.status["active"] = False
        self.status["mode"] = "standby"
        self.status["power_level"] = 0.0
        self.status["adaptation_progress"] = 0.0
        self.status["cooldown_remaining"] = self.specs.cooldown_period
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stealth system."""
        return self.status
    
    def adjust_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Adjust operational parameters of the active camouflage system.
        
        Args:
            parameters: New parameters to set
            
        Returns:
            Success status
        """
        if not self.initialized:
            return False
            
        if "power_level" in parameters:
            power_level = parameters["power_level"]
            if 0.0 <= power_level <= 1.0:
                self.status["power_level"] = power_level
                self.camouflage_params.power_level = power_level
                
        if "environment" in parameters:
            self.status["current_environment"] = parameters["environment"]
            # Reset adaptation progress when environment changes
            self.status["adaptation_progress"] = 0.2
                
        if "pattern_complexity" in parameters:
            complexity = parameters["pattern_complexity"]
            if 0.0 <= complexity <= 1.0:
                self.camouflage_params.pattern_complexity = complexity
                
        if "adaptation_speed" in parameters:
            speed = parameters["adaptation_speed"]
            if 0.0 <= speed <= 1.0:
                self.camouflage_params.adaptation_speed = speed
                
        return True
    
    def update_system(self, time_delta: float) -> None:
        """
        Update system status based on time elapsed.
        
        Args:
            time_delta: Time elapsed in seconds
        """
        if not self.initialized:
            return
            
        if self.status["active"]:
            # Convert time_delta to minutes
            time_delta_min = time_delta / 60.0
            
            # Update remaining operation time
            self.status["remaining_operation_time"] -= time_delta_min
            
            # Check if operation time has expired
            if self.status["remaining_operation_time"] <= 0:
                self.deactivate()
                self.status["remaining_operation_time"] = 0.0
                
            # Update adaptation progress
            if self.status["adaptation_progress"] < 1.0:
                adaptation_increment = time_delta * self.camouflage_params.adaptation_speed * 0.05
                self.status["adaptation_progress"] = min(self.status["adaptation_progress"] + adaptation_increment, 1.0)
                
        elif self.status["cooldown_remaining"] > 0:
            # Update cooldown time
            time_delta_min = time_delta / 60.0
            self.status["cooldown_remaining"] -= time_delta_min
            
            if self.status["cooldown_remaining"] <= 0:
                self.status["cooldown_remaining"] = 0.0
                self.status["remaining_operation_time"] = self.config.operational_duration_minutes

    def analyze_visual_environment(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze visual environment to adapt camouflage pattern.
        
        Args:
            image_data: RGB image data of the environment
            
        Returns:
            Analysis results
        """
        if not self.initialized or not self.status["active"]:
            return {"success": False, "message": "System not active"}
            
        # Initialize visual signature matcher if not already done
        if not hasattr(self, 'visual_matcher'):
            from src.stealth.camouflage.visual_signature import VisualSignatureMatcher, MatchingAlgorithm
            self.visual_matcher = VisualSignatureMatcher(
                matching_algorithm=MatchingAlgorithm.ADAPTIVE_BLENDING
            )
            
        # Analyze environment
        signature = self.visual_matcher.analyze_environment(image_data)
        
        # Store signature for current environment if one is set
        if self.status["current_environment"] and self.status["current_environment"] != "auto":
            self.visual_matcher.store_reference_signature(
                self.status["current_environment"], 
                signature
            )
            
        # Update adaptation progress based on signature analysis
        # More complex environments take longer to adapt to
        complexity_factor = 1.0 - (signature.pattern_complexity * 0.5)
        self.status["adaptation_progress"] = min(
            self.status["adaptation_progress"] + (0.1 * complexity_factor),
            1.0
        )
        
        return {
            "success": True,
            "signature": {
                "dominant_colors": signature.dominant_colors,
                "pattern_complexity": signature.pattern_complexity,
                "edge_density": signature.edge_density,
                "light_levels": signature.light_levels,
                "contrast_ratio": signature.contrast_ratio
            },
            "adaptation_progress": self.status["adaptation_progress"]
        }
        
    def generate_camouflage_pattern(self, 
                                   environment_type: Optional[str] = None, 
                                   resolution: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """
        Generate camouflage pattern based on current environment.
        
        Args:
            environment_type: Override environment type
            resolution: Output resolution (width, height)
            
        Returns:
            RGB image data of generated camouflage pattern
        """
        if not self.initialized:
            return np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            
        # Initialize visual signature matcher if not already done
        if not hasattr(self, 'visual_matcher'):
            from src.stealth.camouflage.visual_signature import VisualSignatureMatcher, MatchingAlgorithm
            self.visual_matcher = VisualSignatureMatcher(
                resolution=resolution,
                matching_algorithm=MatchingAlgorithm.ADAPTIVE_BLENDING
            )
            
        # Use provided environment type or current environment
        env_type = environment_type or self.status["current_environment"] or "urban"
        
        # Get reference signature if available
        signature = None
        if env_type != "auto":
            signature = self.visual_matcher.get_reference_signature(env_type)
            
        # Generate pattern
        pattern = self.visual_matcher.generate_camouflage_pattern(
            signature=signature,
            environment_type=env_type
        )
        
        # Store current pattern
        self.status["current_pattern"] = pattern
        
        return pattern
        
    def blend_to_new_environment(self, 
                               new_environment: str, 
                               blend_time: float = 2.0) -> bool:
        """
        Blend camouflage pattern to a new environment.
        
        Args:
            new_environment: New environment type
            blend_time: Time to blend in seconds
            
        Returns:
            Success status
        """
        if not self.initialized or not self.status["active"]:
            return False
            
        # Initialize visual signature matcher if not already done
        if not hasattr(self, 'visual_matcher'):
            from src.stealth.camouflage.visual_signature import VisualSignatureMatcher, MatchingAlgorithm
            self.visual_matcher = VisualSignatureMatcher(
                matching_algorithm=MatchingAlgorithm.ADAPTIVE_BLENDING
            )
            
        # Store current pattern and environment
        old_environment = self.status["current_environment"]
        
        # Generate pattern for new environment
        new_pattern = self.generate_camouflage_pattern(environment_type=new_environment)
        
        # Update environment
        self.status["current_environment"] = new_environment
        
        # Reset adaptation progress
        self.status["adaptation_progress"] = 0.3
        
        # In a real system, we would gradually blend between patterns
        # Here we just update the current pattern
        self.status["current_pattern"] = new_pattern
        
        return True
        
    def calculate_visual_signature_match(self, 
                                       environment_image: np.ndarray) -> float:
        """
        Calculate how well the current camouflage matches the environment.
        
        Args:
            environment_image: RGB image of the environment
            
        Returns:
            Match score (0.0-1.0)
        """
        if not self.initialized or not self.status["active"]:
            return 0.0
            
        # Initialize visual signature matcher if not already done
        if not hasattr(self, 'visual_matcher'):
            from src.stealth.camouflage.visual_signature import VisualSignatureMatcher, MatchingAlgorithm
            self.visual_matcher = VisualSignatureMatcher(
                matching_algorithm=MatchingAlgorithm.ADAPTIVE_BLENDING
            )
            
        # Analyze environment
        env_signature = self.visual_matcher.analyze_environment(environment_image)
        
        # Get current pattern signature
        if self.status["current_pattern"] is not None:
            pattern_signature = self.visual_matcher.analyze_environment(self.status["current_pattern"])
            
            # Calculate similarity
            similarity = self.visual_matcher.calculate_signature_similarity(
                env_signature, pattern_signature
            )
            
            # Apply power level and adaptation progress
            match_score = similarity * self.status["power_level"] * self.status["adaptation_progress"]
            
            return match_score
        else:
            return 0.0
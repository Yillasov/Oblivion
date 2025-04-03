#!/usr/bin/env python3
"""
Environmental sensing for camouflage optimization.
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

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.stealth.camouflage.visual_signature import VisualSignature
from src.stealth.camouflage.adaptive_surface import AdaptiveSurfaceController


@dataclass
class EnvironmentalConditions:
    """Environmental conditions data structure."""
    light_level: float  # 0.0-1.0
    temperature: float  # Celsius
    humidity: float  # 0.0-1.0
    weather: str  # clear, rain, fog, etc.
    terrain_type: str  # urban, forest, desert, etc.


class EnvironmentalSensor:
    """
    Simple environmental sensing for camouflage optimization.
    Detects environmental conditions to optimize adaptive camouflage.
    """
    
    def __init__(self, surface_controller: AdaptiveSurfaceController):
        """
        Initialize environmental sensor.
        
        Args:
            surface_controller: Adaptive surface controller to update
        """
        self.surface_controller = surface_controller
        self.current_conditions = EnvironmentalConditions(
            light_level=0.5,
            temperature=20.0,
            humidity=0.5,
            weather="clear",
            terrain_type="urban"
        )
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image to detect environmental conditions.
        
        Args:
            image: RGB image of environment
            
        Returns:
            Detected environmental conditions
        """
        # Extract light level from image brightness
        light_level = float(np.mean(image) / 255.0)
        
        # Simple terrain classification based on color distribution
        hsv = np.mean(image, axis=(0, 1))
        
        # Very basic terrain classification
        if np.mean(image[:,:,1]) > np.mean(image[:,:,0]) and np.mean(image[:,:,1]) > np.mean(image[:,:,2]):
            terrain_type = "forest"
        elif np.mean(image[:,:,2]) > np.mean(image[:,:,0]) and np.mean(image[:,:,2]) > np.mean(image[:,:,1]):
            terrain_type = "ocean"
        elif np.mean(image) > 180:
            terrain_type = "desert"
        elif np.mean(image) < 60:
            terrain_type = "night"
        else:
            terrain_type = "urban"
        
        # Update current conditions
        self.current_conditions.light_level = light_level
        self.current_conditions.terrain_type = terrain_type
        
        return {
            "light_level": light_level,
            "terrain_type": terrain_type,
            "conditions": self.current_conditions
        }
    
    def optimize_camouflage(self, signature: VisualSignature) -> Dict[str, Any]:
        """
        Optimize camouflage based on environmental conditions and visual signature.
        
        Args:
            signature: Visual signature of environment
            
        Returns:
            Optimized surface parameters
        """
        # Update surface controller with signature and environment type
        params = self.surface_controller.update_surface(
            signature, 
            environment_type=self.current_conditions.terrain_type
        )
        
        # Apply environmental-specific optimizations
        if self.current_conditions.light_level < 0.3:
            # Low light optimization
            params["reflectivity"] = min(params["reflectivity"] * 0.8, 1.0)
        
        if self.current_conditions.terrain_type == "forest":
            # Forest optimization - increase texture depth
            params["texture_depth"] = min(params["texture_depth"] * 1.2, 1.0)
        
        return params
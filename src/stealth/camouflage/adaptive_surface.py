"""
Adaptive surface control system for active camouflage.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.stealth.camouflage.visual_signature import VisualSignature, VisualSignatureMatcher


class SurfaceMode(Enum):
    """Surface adaptation modes."""
    STATIC = "static"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    LEARNING = "learning"


@dataclass
class SurfaceParameters:
    """Parameters for adaptive surface control."""
    reflectivity: float  # 0.0-1.0
    texture_depth: float  # 0.0-1.0
    color_shift_rate: float  # Rate of color adaptation
    pattern_scale: float  # Scale factor for patterns
    transition_speed: float  # Speed of transitions


class AdaptiveSurfaceController:
    """
    Controls adaptive surface properties for active camouflage.
    Works with visual signature matching to optimize stealth.
    """
    
    def __init__(self, 
                signature_matcher: VisualSignatureMatcher,
                mode: SurfaceMode = SurfaceMode.REACTIVE):
        """
        Initialize adaptive surface controller.
        
        Args:
            signature_matcher: Visual signature matcher for environment analysis
            mode: Surface adaptation mode
        """
        self.signature_matcher = signature_matcher
        self.mode = mode
        self.current_parameters = SurfaceParameters(
            reflectivity=0.5,
            texture_depth=0.3,
            color_shift_rate=0.2,
            pattern_scale=1.0,
            transition_speed=0.5
        )
        self.adaptation_history = []
        
    def update_surface(self, 
                      signature: VisualSignature, 
                      environment_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Update surface parameters based on visual signature.
        
        Args:
            signature: Current visual signature of environment
            environment_type: Optional environment type identifier
            
        Returns:
            Updated surface parameters
        """
        # Adapt reflectivity based on light levels
        self.current_parameters.reflectivity = 1.0 - signature.light_levels
        
        # Adapt texture depth based on pattern complexity
        self.current_parameters.texture_depth = signature.pattern_complexity
        
        # Adapt color shift rate based on contrast ratio
        self.current_parameters.color_shift_rate = signature.contrast_ratio * 0.5
        
        # Adapt pattern scale based on edge density
        self.current_parameters.pattern_scale = 1.0 + signature.edge_density
        
        # Store adaptation in history
        self.adaptation_history.append({
            "signature": signature,
            "parameters": self.current_parameters,
            "environment_type": environment_type
        })
        
        # Return current parameters as dictionary
        return {
            "reflectivity": self.current_parameters.reflectivity,
            "texture_depth": self.current_parameters.texture_depth,
            "color_shift_rate": self.current_parameters.color_shift_rate,
            "pattern_scale": self.current_parameters.pattern_scale,
            "transition_speed": self.current_parameters.transition_speed
        }
    
    def apply_surface_control(self, 
                             pattern: np.ndarray, 
                             parameters: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply surface control to modify camouflage pattern.
        
        Args:
            pattern: Base camouflage pattern
            parameters: Optional override parameters
            
        Returns:
            Modified pattern with surface controls applied
        """
        if parameters is None:
            parameters = {
                "reflectivity": self.current_parameters.reflectivity,
                "texture_depth": self.current_parameters.texture_depth,
                "pattern_scale": self.current_parameters.pattern_scale
            }
        
        # Apply reflectivity adjustment
        reflectivity = parameters.get("reflectivity", 0.5)
        pattern = pattern * reflectivity
        
        # Apply texture depth
        texture_depth = parameters.get("texture_depth", 0.3)
        if texture_depth > 0.5:
            # Enhance texture by increasing local contrast
            pattern = np.clip(pattern * (1.0 + texture_depth * 0.5), 0, 255).astype(np.uint8)
        
        return pattern
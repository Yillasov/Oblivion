"""
Integration module for connecting landing gear systems with other Oblivion components.
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np

from src.landing_gear.base import NeuromorphicLandingGear
from src.landing_gear.types import LandingGearType
from src.landing_gear.neuromorphic_control import NeuromorphicLandingController, AdaptiveMode


class LandingGearIntegration:
    """Integration layer for landing gear systems."""
    
    def __init__(self, landing_gear: NeuromorphicLandingGear):
        """Initialize with a landing gear system."""
        self.landing_gear = landing_gear
        self.controller = NeuromorphicLandingController(landing_gear)
        self.event_handlers: Dict[str, List[Callable]] = {
            "pre_landing": [],
            "landing": [],
            "post_landing": [],
            "pre_takeoff": [],
            "takeoff": [],
            "emergency": []
        }
        self.avionics_data: Dict[str, Any] = {}
        self.mission_params: Dict[str, Any] = {}
    
    def register_event_handler(self, event_type: str, handler: Callable) -> bool:
        """Register a handler for specific landing gear events."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            return True
        return False
    
    def update_avionics_data(self, data: Dict[str, Any]) -> None:
        """Update avionics data for landing gear integration."""
        self.avionics_data = data
        
        # Extract relevant data for landing gear
        altitude = data.get("altitude", 0.0)
        airspeed = data.get("airspeed", 0.0)
        vertical_speed = data.get("vertical_speed", 0.0)
        
        # Automatically prepare landing gear based on flight parameters
        if altitude < 500 and vertical_speed < -1.0:  # Descending below 500m
            self._trigger_event("pre_landing", {
                "altitude": altitude,
                "airspeed": airspeed,
                "vertical_speed": vertical_speed
            })
            
            # Deploy landing gear if not already deployed
            if not self.landing_gear.status.get("deployed", False):
                self.landing_gear.deploy()
                
                # Configure gear for landing based on conditions
                self._configure_for_landing(airspeed, vertical_speed)
        
        # Detect landing
        if altitude < 5 and abs(vertical_speed) < 0.5 and self.landing_gear.status.get("deployed", False):
            self._trigger_event("landing", {
                "touchdown_speed": vertical_speed,
                "ground_speed": airspeed
            })
    
    def set_mission_parameters(self, params: Dict[str, Any]) -> None:
        """Set mission-specific parameters for landing gear operation."""
        self.mission_params = params
        
        # Configure landing gear based on mission parameters
        if "terrain_type" in params:
            self._adapt_to_terrain(params["terrain_type"])
        
        if "stealth_mode" in params and params["stealth_mode"]:
            # Change to use the mode property directly instead of a non-existent set_mode method
            self.controller.mode = AdaptiveMode.STEALTH
    
    def _configure_for_landing(self, airspeed: float, vertical_speed: float) -> None:
        """Configure landing gear for optimal landing based on conditions."""
        gear_type = self.landing_gear.specs.gear_type
        
        # Type-specific landing configurations
        if gear_type == LandingGearType.ADAPTIVE_SHOCK_ABSORBING:
            # Adjust shock absorption based on vertical speed
            from src.landing_gear.implementations import AdaptiveShockGear
            if isinstance(self.landing_gear, AdaptiveShockGear):
                # Softer for hard landings, stiffer for gentle landings
                stiffness = 0.7 - min(0.5, abs(vertical_speed) / 10.0)
                self.landing_gear.adjust_stiffness(stiffness)
                
        elif gear_type == LandingGearType.RETRACTABLE_MORPHING:
            # Configure morphing state based on landing conditions
            from src.landing_gear.implementations import RetractableMorphingGear
            if isinstance(self.landing_gear, RetractableMorphingGear):
                if airspeed > 100:
                    self.landing_gear.morph("high_speed")
                else:
                    self.landing_gear.morph("landing")
    
    def _adapt_to_terrain(self, terrain_type: str) -> None:
        """Adapt landing gear to specific terrain type."""
        gear_type = self.landing_gear.specs.gear_type
        
        # Type-specific terrain adaptations
        if gear_type == LandingGearType.ADAPTIVE_SHOCK_ABSORBING:
            from src.landing_gear.implementations import AdaptiveShockGear
            if isinstance(self.landing_gear, AdaptiveShockGear):
                # Map terrain types to landing gear terrain adaptations
                terrain_mapping = {
                    "runway": "normal",
                    "grass": "soft",
                    "gravel": "rough",
                    "water": "soft",
                    "desert": "soft",
                    "snow": "soft",
                    "ice": "hard",
                    "mountain": "uneven"
                }
                
                adaptation = terrain_mapping.get(terrain_type, "normal")
                self.landing_gear.adapt_to_terrain(adaptation)
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Trigger registered event handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                handler(data)
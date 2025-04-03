#!/usr/bin/env python3
"""
Mission-adaptive cooling system for stealth operations.
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

from typing import Dict, Any, Optional
from enum import Enum

from src.stealth.metamaterial.exhaust_cooling import ExhaustCoolingSystem, ExhaustMixingMode
from src.testing.scenarios.mission_profiles import MissionProfile


class StealthPriority(Enum):
    """Stealth priority levels for mission-adaptive cooling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class MissionAdaptiveCooling:
    """
    Adapts cooling systems based on mission profiles to optimize
    stealth characteristics and power consumption.
    """
    
    def __init__(self, cooling_system: ExhaustCoolingSystem):
        """
        Initialize mission-adaptive cooling.
        
        Args:
            cooling_system: Exhaust cooling system to control
        """
        self.cooling_system = cooling_system
        self.current_mission_type = None
        self.stealth_priority = StealthPriority.MEDIUM
        self.power_conservation = 0.5  # 0.0-1.0
        
    def adapt_to_mission(self, mission_profile: MissionProfile) -> Dict[str, Any]:
        """
        Adapt cooling system to mission profile.
        
        Args:
            mission_profile: Mission profile to adapt to
            
        Returns:
            Adaptation results
        """
        # Extract mission parameters
        mission_type = mission_profile.mission_type.value
        self.current_mission_type = mission_type
        
        # Extract environmental conditions
        ambient_temp = mission_profile.environment_conditions.get("temperature", 20.0)
        altitude = mission_profile.environment_conditions.get("altitude", 0.0)
        
        # Determine stealth priority based on mission type and threats
        if mission_type == "strike" or len(mission_profile.threat_scenarios) > 2:
            self.stealth_priority = StealthPriority.HIGH
        elif mission_type == "reconnaissance":
            self.stealth_priority = StealthPriority.MAXIMUM
        elif mission_type == "transport":
            self.stealth_priority = StealthPriority.LOW
        else:
            self.stealth_priority = StealthPriority.MEDIUM
            
        # Set cooling mode based on stealth priority
        cooling_mode = self._get_cooling_mode_for_priority()
        self.cooling_system.set_mixing_mode(cooling_mode)
        
        # Activate cooling system
        self.cooling_system.activate()
        
        return {
            "mission_type": mission_type,
            "stealth_priority": self.stealth_priority.value,
            "cooling_mode": cooling_mode.value,
            "power_consumption": self.cooling_system.power_consumption,
            "ambient_temperature": ambient_temp,
            "altitude": altitude
        }
    
    def _get_cooling_mode_for_priority(self) -> ExhaustMixingMode:
        """Get appropriate cooling mode for current stealth priority."""
        if self.stealth_priority == StealthPriority.MAXIMUM:
            return ExhaustMixingMode.MULTI_STAGE
        elif self.stealth_priority == StealthPriority.HIGH:
            return ExhaustMixingMode.BYPASS_MIXING
        elif self.stealth_priority == StealthPriority.MEDIUM:
            return ExhaustMixingMode.ACTIVE_DILUTION
        else:
            return ExhaustMixingMode.PASSIVE
    
    def adapt_to_threat(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt cooling system to immediate threat.
        
        Args:
            threat_data: Threat information
            
        Returns:
            Adaptation results
        """
        threat_type = threat_data.get("type", "unknown")
        threat_distance = threat_data.get("distance", 100.0)
        threat_detection_capability = threat_data.get("detection_capability", {})
        
        # Check if threat has IR detection capability
        has_ir_detection = threat_detection_capability.get("infrared", False)
        
        # Determine if emergency mode is needed
        emergency_mode = False
        if has_ir_detection and threat_distance < 20.0:
            emergency_mode = True
            self.cooling_system.set_mixing_mode(ExhaustMixingMode.EMERGENCY)
        elif has_ir_detection and threat_distance < 50.0:
            # Increase stealth priority temporarily
            original_priority = self.stealth_priority
            self.stealth_priority = StealthPriority.MAXIMUM
            self.cooling_system.set_mixing_mode(self._get_cooling_mode_for_priority())
            self.stealth_priority = original_priority
        
        return {
            "threat_type": threat_type,
            "threat_distance": threat_distance,
            "has_ir_detection": has_ir_detection,
            "emergency_mode": emergency_mode,
            "cooling_mode": self.cooling_system.mixing_mode.value,
            "power_consumption": self.cooling_system.power_consumption
        }
    
    def optimize_for_duration(self, mission_duration: float) -> Dict[str, Any]:
        """
        Optimize cooling for extended mission duration.
        
        Args:
            mission_duration: Mission duration in minutes
            
        Returns:
            Optimization results
        """
        # For long missions, balance stealth and power consumption
        if mission_duration > 120.0:  # More than 2 hours
            self.power_conservation = 0.8
            
            # If not in high-threat environment, reduce cooling power
            if self.stealth_priority in [StealthPriority.LOW, StealthPriority.MEDIUM]:
                if self.cooling_system.mixing_mode == ExhaustMixingMode.ACTIVE_DILUTION:
                    self.cooling_system.set_mixing_mode(ExhaustMixingMode.PASSIVE)
                elif self.cooling_system.mixing_mode == ExhaustMixingMode.MULTI_STAGE:
                    self.cooling_system.set_mixing_mode(ExhaustMixingMode.BYPASS_MIXING)
        else:
            # For shorter missions, prioritize stealth
            self.power_conservation = 0.3
            
        return {
            "mission_duration": mission_duration,
            "power_conservation": self.power_conservation,
            "cooling_mode": self.cooling_system.mixing_mode.value,
            "power_consumption": self.cooling_system.power_consumption
        }
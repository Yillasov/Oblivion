"""
Nuclear Thermal Propulsion Safety System for advanced space propulsion.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time

from src.propulsion.base import PropulsionInterface
from src.propulsion.thermal_management import ThermalProfile


class SafetyStatus(Enum):
    """Safety status levels for nuclear thermal propulsion."""
    NOMINAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY_SHUTDOWN = 3
    CONTAINMENT_BREACH = 4


class ShieldingStatus(Enum):
    """Status of radiation shielding systems."""
    OPTIMAL = 0
    DEGRADED = 1
    COMPROMISED = 2
    FAILED = 3


@dataclass
class NuclearSafetyParameters:
    """Safety parameters for nuclear thermal propulsion."""
    max_core_temperature: float  # Maximum safe core temperature in K
    max_radiation_level: float  # Maximum safe radiation level in mSv/h
    coolant_flow_min: float  # Minimum coolant flow rate in kg/s
    pressure_limits: Tuple[float, float]  # Min/max pressure in MPa
    emergency_coolant_capacity: float  # Emergency coolant capacity in kg
    containment_integrity: float  # Containment integrity factor (0-1)
    shielding_thickness: float  # Radiation shielding thickness in cm
    scram_response_time: float  # Emergency shutdown response time in seconds


class NuclearThermalSafetySystem:
    """Safety system for nuclear thermal propulsion."""
    
    def __init__(self, safety_params: NuclearSafetyParameters):
        """Initialize nuclear thermal safety system."""
        self.safety_params = safety_params
        self.safety_status = SafetyStatus.NOMINAL
        self.shielding_status = ShieldingStatus.OPTIMAL
        self.core_temperature = 300.0  # K
        self.radiation_level = 0.01  # mSv/h
        self.coolant_flow = 0.0  # kg/s
        self.pressure = 0.1  # MPa
        self.last_check_time = time.time()
        self.emergency_coolant_remaining = safety_params.emergency_coolant_capacity
        self.containment_integrity = safety_params.containment_integrity
        self.scram_activated = False
        self.safety_log: List[Dict[str, Any]] = []
        
    def check_safety_status(self, propulsion_data: Dict[str, Any]) -> SafetyStatus:
        """
        Check safety status based on current propulsion data.
        
        Args:
            propulsion_data: Current propulsion system data
            
        Returns:
            Current safety status
        """
        # Update internal state
        self.core_temperature = propulsion_data.get("core_temperature", self.core_temperature)
        self.radiation_level = propulsion_data.get("radiation_level", self.radiation_level)
        self.coolant_flow = propulsion_data.get("coolant_flow", self.coolant_flow)
        self.pressure = propulsion_data.get("pressure", self.pressure)
        
        # Check for critical conditions
        if self.core_temperature > self.safety_params.max_core_temperature * 1.2:
            self.safety_status = SafetyStatus.CONTAINMENT_BREACH
        elif self.core_temperature > self.safety_params.max_core_temperature:
            self.safety_status = SafetyStatus.EMERGENCY_SHUTDOWN
        elif self.radiation_level > self.safety_params.max_radiation_level:
            self.safety_status = SafetyStatus.CRITICAL
        elif (self.coolant_flow < self.safety_params.coolant_flow_min and 
              self.core_temperature > 500.0):
            self.safety_status = SafetyStatus.CRITICAL
        elif (self.pressure < self.safety_params.pressure_limits[0] or 
              self.pressure > self.safety_params.pressure_limits[1]):
            self.safety_status = SafetyStatus.WARNING
        else:
            self.safety_status = SafetyStatus.NOMINAL
            
        # Log safety status
        self._log_safety_status()
        
        return self.safety_status
    
    def activate_emergency_systems(self) -> Dict[str, Any]:
        """
        Activate emergency safety systems.
        
        Returns:
            Status of emergency systems
        """
        # Activate SCRAM if not already activated
        if not self.scram_activated and self.safety_status in [
            SafetyStatus.CRITICAL, SafetyStatus.EMERGENCY_SHUTDOWN
        ]:
            self.scram_activated = True
            
        # Deploy emergency coolant if needed
        coolant_deployed = 0.0
        if (self.safety_status == SafetyStatus.EMERGENCY_SHUTDOWN and 
            self.emergency_coolant_remaining > 0):
            coolant_deployed = min(
                self.emergency_coolant_remaining, 
                self.safety_params.coolant_flow_min * 2
            )
            self.emergency_coolant_remaining -= coolant_deployed
            
        # Update containment integrity based on conditions
        if self.safety_status == SafetyStatus.CONTAINMENT_BREACH:
            self.containment_integrity *= 0.8  # Degradation during breach
            
        return {
            "scram_activated": self.scram_activated,
            "coolant_deployed": coolant_deployed,
            "emergency_coolant_remaining": self.emergency_coolant_remaining,
            "containment_integrity": self.containment_integrity,
            "safety_status": self.safety_status.name
        }
    
    def get_safety_recommendations(self) -> List[str]:
        """Get safety recommendations based on current status."""
        recommendations = []
        
        if self.safety_status == SafetyStatus.WARNING:
            recommendations.append("Reduce power output to decrease core temperature")
            recommendations.append("Increase coolant flow rate")
            
        elif self.safety_status == SafetyStatus.CRITICAL:
            recommendations.append("Initiate power reduction sequence immediately")
            recommendations.append("Maximize coolant flow")
            recommendations.append("Prepare emergency shutdown procedures")
            
        elif self.safety_status == SafetyStatus.EMERGENCY_SHUTDOWN:
            recommendations.append("SCRAM reactor immediately")
            recommendations.append("Deploy emergency coolant systems")
            recommendations.append("Evacuate non-essential personnel from reactor section")
            
        elif self.safety_status == SafetyStatus.CONTAINMENT_BREACH:
            recommendations.append("Initiate full containment protocols")
            recommendations.append("Deploy all available coolant resources")
            recommendations.append("Evacuate all personnel from affected sections")
            recommendations.append("Prepare for emergency jettison if structural integrity compromised")
            
        return recommendations
    
    def get_thermal_profile(self) -> ThermalProfile:
        """Get thermal profile for integration with thermal management system."""
        return ThermalProfile(
            name="nuclear_core",
            max_temperature=self.safety_params.max_core_temperature,
            optimal_temperature=self.safety_params.max_core_temperature * 0.7,
            cooling_rate=0.05,  # K/s passive cooling
            heating_rate=2.0,  # K/s at full power
            thermal_mass=5000.0,  # J/K
            current_temperature=self.core_temperature
        )
    
    def reset_scram(self) -> bool:
        """
        Reset SCRAM system after emergency.
        
        Returns:
            Success status
        """
        if self.safety_status in [SafetyStatus.NOMINAL, SafetyStatus.WARNING]:
            self.scram_activated = False
            return True
        return False
    
    def _log_safety_status(self) -> None:
        """Log current safety status and parameters."""
        current_time = time.time()
        
        log_entry = {
            "timestamp": current_time,
            "status": self.safety_status.name,
            "core_temperature": self.core_temperature,
            "radiation_level": self.radiation_level,
            "coolant_flow": self.coolant_flow,
            "pressure": self.pressure,
            "containment_integrity": self.containment_integrity,
            "emergency_coolant": self.emergency_coolant_remaining,
            "scram_activated": self.scram_activated
        }
        
        self.safety_log.append(log_entry)
        self.last_check_time = current_time
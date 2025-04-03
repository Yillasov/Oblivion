import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Replace the import with the enum definition directly if the file doesn't exist yet
# or update the import path if it's located elsewhere
from src.landing_gear.types import LandingGearType


@dataclass
class TelemetryData:
    """Telemetry data for landing gear systems."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    position: float = 0.0  # Position as percentage (0.0 = retracted, 1.0 = fully deployed)
    load: float = 0.0  # Current load in kg
    temperature: float = 25.0  # Temperature in Celsius
    hydraulic_pressure: float = 0.0  # Hydraulic pressure in MPa
    vibration: float = 0.0  # Vibration level
    power_consumption: float = 0.0  # Power consumption in watts
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LandingGearSpecs:
    """Specifications for landing gear systems."""
    # Basic physical properties
    weight: float  # Weight in kg
    dimensions: Dict[str, float]  # Dimensions in meters
    power_requirements: float  # Power requirements in watts
    max_load_capacity: float  # Maximum load capacity in kg
    gear_type: LandingGearType  # Type of landing gear
    deployment_time: float  # Time to deploy in seconds
    retraction_time: float  # Time to retract in seconds
    
    # Environmental specifications
    operational_temperature: Dict[str, float] = field(default_factory=lambda: {"min": -40.0, "max": 85.0})
    operational_altitude: Dict[str, float] = field(default_factory=lambda: {"min": 0.0, "max": 15000.0})  # in meters
    weather_resistance: Dict[str, float] = field(default_factory=lambda: {
        "rain": 0.8,  # 0-1 scale
        "snow": 0.6,
        "dust": 0.7,
        "ice": 0.5
    })
    
    # Materials and construction
    materials: Dict[str, str] = field(default_factory=dict)
    shock_absorption: float = 0.7  # 0-1 scale
    
    # Type-specific properties
    morphing_capabilities: Dict[str, Any] = field(default_factory=dict)  # For RETRACTABLE_MORPHING
    electromagnetic_specs: Dict[str, Any] = field(default_factory=dict)  # For ELECTROMAGNETIC_CATAPULT
    vtol_specs: Dict[str, Any] = field(default_factory=dict)  # For VTOL_ROTORS
    cushion_specs: Dict[str, Any] = field(default_factory=dict)  # For AIR_CUSHION
    adaptive_shock_specs: Dict[str, Any] = field(default_factory=dict)  # For ADAPTIVE_SHOCK_ABSORBING
    magnetic_specs: Dict[str, Any] = field(default_factory=dict)  # For MAGNETIC_LEVITATION
    inflatable_specs: Dict[str, Any] = field(default_factory=dict)  # For INFLATABLE
    parachute_specs: Dict[str, Any] = field(default_factory=dict)  # For PARACHUTE_ASSISTED
    skid_specs: Dict[str, Any] = field(default_factory=dict)  # For AUTONOMOUS_SKID
    rocket_specs: Dict[str, Any] = field(default_factory=dict)  # For ROCKET_ASSISTED
    
    # Performance characteristics
    max_landing_speed: float = 100.0  # Maximum safe landing speed in km/h
    max_takeoff_speed: float = 100.0  # Maximum takeoff speed in km/h
    ground_clearance: float = 0.3  # Ground clearance in meters
    stability_rating: float = 0.8  # 0-1 scale
    
    # Maintenance and reliability
    maintenance_interval: int = 500  # Hours between maintenance
    expected_lifetime: int = 5000  # Expected operational hours
    redundancy_level: int = 1  # Number of redundant systems
    
    # Integration with aircraft
    mounting_points: List[Dict[str, float]] = field(default_factory=list)  # Coordinates for mounting
    control_interface: str = "standard"  # Type of control interface
    
    # Additional specifications
    additional_specs: Dict[str, Any] = field(default_factory=dict)
    
    def get_type_specific_specs(self) -> Dict[str, Any]:
        """Get specifications specific to this landing gear type."""
        type_map = {
            LandingGearType.RETRACTABLE_MORPHING: self.morphing_capabilities,
            LandingGearType.ELECTROMAGNETIC_CATAPULT: self.electromagnetic_specs,
            LandingGearType.VTOL_ROTORS: self.vtol_specs,
            LandingGearType.AIR_CUSHION: self.cushion_specs,
            LandingGearType.ADAPTIVE_SHOCK_ABSORBING: self.adaptive_shock_specs,
            LandingGearType.MAGNETIC_LEVITATION: self.magnetic_specs,
            LandingGearType.INFLATABLE: self.inflatable_specs,
            LandingGearType.PARACHUTE_ASSISTED: self.parachute_specs,
            LandingGearType.AUTONOMOUS_SKID: self.skid_specs,
            LandingGearType.ROCKET_ASSISTED: self.rocket_specs
        }
        return type_map.get(self.gear_type, {})
    
    def calculate_performance_index(self) -> float:
        """Calculate a simple performance index for this landing gear."""
        # Higher is better
        weight_factor = 1.0 - (self.weight / 1000)  # Lighter is better
        load_factor = self.max_load_capacity / 10000  # Higher capacity is better
        speed_factor = self.max_landing_speed / 200  # Higher speed tolerance is better
        
        # Simple weighted average
        return (weight_factor * 0.3 + 
                load_factor * 0.3 + 
                speed_factor * 0.2 + 
                self.stability_rating * 0.2)


class LandingGearInterface(ABC):
    """Interface for all landing gear systems."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the landing gear system."""
        pass
    
    @abstractmethod
    def get_specifications(self) -> LandingGearSpecs:
        """Get the physical specifications of the landing gear."""
        pass
    
    @abstractmethod
    def deploy(self) -> bool:
        """Deploy the landing gear."""
        pass
    
    @abstractmethod
    def retract(self) -> bool:
        """Retract the landing gear."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the landing gear."""
        pass
    
    @abstractmethod
    def emergency_deploy(self) -> bool:
        """Emergency deployment of landing gear."""
        pass
    
    @abstractmethod
    def lock(self) -> bool:
        """Lock the landing gear in current position."""
        pass
    
    @abstractmethod
    def unlock(self) -> bool:
        """Unlock the landing gear."""
        pass
    
    @abstractmethod
    def calculate_impact(self) -> Dict[str, float]:
        """Calculate the impact of this landing gear on UCAV performance."""
        pass
    
    @abstractmethod
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run diagnostics on the landing gear system."""
        pass
    
    @abstractmethod
    def get_telemetry(self) -> TelemetryData:
        """Get current telemetry data from the landing gear."""
        pass
    
    @abstractmethod
    def get_telemetry_history(self, limit: int = 100) -> List[TelemetryData]:
        """Get historical telemetry data."""
        pass
    
    @abstractmethod
    def clear_telemetry_history(self) -> bool:
        """Clear telemetry history."""
        pass


class NeuromorphicLandingGear(LandingGearInterface):
    """Base class for neuromorphic landing gear systems."""
    
    def __init__(self, specs: LandingGearSpecs):
        self.specs = specs
        self.status = {
            "deployed": False,
            "locked": False,
            "health": 1.0,
            "operational": False,
            "diagnostic": "not_run"
        }
        self.initialized = False
        self.telemetry_history: List[TelemetryData] = []
        self.max_telemetry_history = 1000
    
    def initialize(self) -> bool:
        """Initialize the landing gear system."""
        self.initialized = True
        self.status["operational"] = True
        self._record_telemetry()
        return True
    
    def get_specifications(self) -> LandingGearSpecs:
        """Get the physical specifications of the landing gear."""
        return self.specs
    
    def deploy(self) -> bool:
        """Deploy the landing gear."""
        if not self.initialized:
            return False
        self.status["deployed"] = True
        self._record_telemetry()
        return True
    
    def retract(self) -> bool:
        """Retract the landing gear."""
        if not self.initialized or self.status["locked"]:
            return False
        self.status["deployed"] = False
        self._record_telemetry()
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the landing gear."""
        return self.status
    
    def emergency_deploy(self) -> bool:
        """Emergency deployment of landing gear."""
        self.status["deployed"] = True
        self.status["locked"] = True
        self._record_telemetry()
        return True
    
    def lock(self) -> bool:
        """Lock the landing gear in current position."""
        if not self.initialized:
            return False
        self.status["locked"] = True
        self._record_telemetry()
        return True
    
    def unlock(self) -> bool:
        """Unlock the landing gear."""
        if not self.initialized:
            return False
        self.status["locked"] = False
        self._record_telemetry()
        return True
    
    def calculate_impact(self) -> Dict[str, float]:
        """Calculate the impact of this landing gear on UCAV performance."""
        # Basic impact calculation
        deployed_factor = 1.0 if self.status["deployed"] else 0.2
        return {
            "weight_impact": self.specs.weight,
            "drag_coefficient": 0.08 * deployed_factor,
            "power_consumption": self.specs.power_requirements * deployed_factor
        }
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run diagnostics on the landing gear system."""
        if not self.initialized:
            return {"status": "error", "message": "System not initialized"}
        
        # Simulate diagnostics
        self.status["diagnostic"] = "passed"
        self._record_telemetry()
        return {
            "status": "success",
            "health": self.status["health"],
            "mechanical_integrity": 0.98,
            "electrical_systems": 0.99,
            "hydraulic_pressure": 0.95,
            "sensor_readings": {
                "position_sensors": "operational",
                "load_sensors": "operational",
                "temperature": 25.0
            }
        }
    
    def get_telemetry(self) -> TelemetryData:
        """Get current telemetry data from the landing gear."""
        return self._generate_telemetry()
    
    def get_telemetry_history(self, limit: int = 100) -> List[TelemetryData]:
        """Get historical telemetry data."""
        return self.telemetry_history[-limit:] if limit < len(self.telemetry_history) else self.telemetry_history
    
    def clear_telemetry_history(self) -> bool:
        """Clear telemetry history."""
        self.telemetry_history = []
        return True
    
    def _generate_telemetry(self) -> TelemetryData:
        """Generate telemetry data based on current state."""
        # In a real system, this would read from actual sensors
        position = 1.0 if self.status["deployed"] else 0.0
        
        # Simulate some basic telemetry
        telemetry = TelemetryData(
            position=position,
            load=0.0,  # No load when not on ground
            temperature=25.0 + (5.0 * position),  # Slightly higher temp when deployed
            hydraulic_pressure=15.0 * position,  # Pressure when deployed
            vibration=0.05 * position,  # Small vibration when deployed
            power_consumption=self.specs.power_requirements * position
        )
        
        return telemetry
    
    def _record_telemetry(self) -> None:
        """Record current telemetry to history."""
        telemetry = self._generate_telemetry()
        self.telemetry_history.append(telemetry)
        
        # Trim history if needed
        if len(self.telemetry_history) > self.max_telemetry_history:
            self.telemetry_history = self.telemetry_history[-self.max_telemetry_history:]
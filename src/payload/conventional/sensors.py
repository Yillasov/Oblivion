"""
Conventional sensor payload systems for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, field

from src.payload.base import NeuromorphicPayload, PayloadSpecs
from src.payload.types import SensorType, PayloadCategory


@dataclass
class SensorSpecs(PayloadSpecs):
    """Extended specifications for sensor systems."""
    sensor_type: SensorType
    range: float  # Maximum effective range in km
    resolution: Dict[str, float]  # Resolution specifications
    field_of_view: float  # Field of view in degrees
    data_rate: float  # Data rate in MB/s


class ConventionalSensor(NeuromorphicPayload):
    """Base class for conventional sensor systems."""
    
    def __init__(self, specs: SensorSpecs, hardware_interface=None):
        super().__init__(hardware_interface)
        self.specs = specs
        self.status = {"active": False, "data_collection": False}
    
    def get_specifications(self) -> PayloadSpecs:
        return self.specs
    
    def calculate_impact(self) -> Dict[str, float]:
        return {
            "weight_impact": self.specs.weight,
            "drag_coefficient": 0.02,
            "power_consumption": self.specs.power_requirements
        }
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        if not self.initialized:
            return False
        self.status["active"] = True
        self.status["data_collection"] = True
        return True
    
    def get_status(self) -> Dict[str, Any]:
        return self.status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.initialized:
            return {"error": "Sensor not initialized"}
        
        # Process sensor data using neuromorphic computing
        if input_data.get("computation") == "target_detection":
            return {
                "targets_detected": 3,
                "target_data": [
                    {"id": 1, "confidence": 0.95, "distance": 15.2},
                    {"id": 2, "confidence": 0.87, "distance": 22.8},
                    {"id": 3, "confidence": 0.76, "distance": 35.1}
                ]
            }
        return {"error": "Unknown computation type"}
    
    def train(self, training_data: Dict[str, Any]) -> bool:
        return True if self.initialized else False


class RadarSystem(ConventionalSensor):
    """Radar sensor system."""
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "AESA-X":
            specs = SensorSpecs(
                weight=120.0,
                volume={"length": 0.8, "width": 0.6, "height": 0.3},
                power_requirements=2500.0,
                mounting_points=["nose", "fuselage"],
                sensor_type=SensorType.RADAR,
                range=150.0,
                resolution={"range": 0.5, "azimuth": 1.0},
                field_of_view=120.0,
                data_rate=850.0
            )
        else:
            raise ValueError(f"Unknown radar model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        self.modes = ["search", "track", "terrain_mapping"]
        self.current_mode = "search"
    
    def set_mode(self, mode: str) -> bool:
        if mode in self.modes:
            self.current_mode = mode
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["current_mode"] = self.current_mode
        return status


class ElectroOpticalSystem(ConventionalSensor):
    """Electro-optical sensor system."""
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "EO-IR-500":
            specs = SensorSpecs(
                weight=45.0,
                volume={"length": 0.5, "width": 0.4, "height": 0.3},
                power_requirements=350.0,
                mounting_points=["fuselage", "nose"],
                sensor_type=SensorType.ELECTRO_OPTICAL,
                range=50.0,
                resolution={"visual": 0.05, "ir": 0.1},
                field_of_view=60.0,
                data_rate=320.0
            )
        else:
            raise ValueError(f"Unknown EO system model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        self.current_spectrum = "visual"  # visual, ir, multi
    
    def switch_spectrum(self, spectrum: str) -> bool:
        if spectrum in ["visual", "ir", "multi"]:
            self.current_spectrum = spectrum
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["current_spectrum"] = self.current_spectrum
        return status
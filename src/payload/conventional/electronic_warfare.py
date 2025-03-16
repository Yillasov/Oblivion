"""
Electronic warfare payload systems for UCAV platforms.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, field

from src.payload.base import NeuromorphicPayload, PayloadSpecs
from src.payload.types import ElectronicWarfareType, PayloadCategory


@dataclass
class ElectronicWarfareSpecs(PayloadSpecs):
    """Extended specifications for electronic warfare systems."""
    ew_type: ElectronicWarfareType
    frequency_range: Dict[str, float]  # Frequency range in MHz
    power_output: float  # Power output in watts
    coverage_angle: float  # Coverage angle in degrees
    modes: List[str]  # Available operational modes


class ElectronicWarfareSystem(NeuromorphicPayload):
    """Base class for electronic warfare systems."""
    
    def __init__(self, specs: ElectronicWarfareSpecs, hardware_interface=None):
        super().__init__(hardware_interface)
        self.specs = specs
        self.status = {
            "active": False,
            "current_mode": None,
            "target_frequencies": [],
            "effectiveness": 0.0
        }
    
    def get_specifications(self) -> PayloadSpecs:
        return self.specs
    
    def calculate_impact(self) -> Dict[str, float]:
        return {
            "weight_impact": self.specs.weight,
            "drag_coefficient": 0.03,
            "power_consumption": self.specs.power_requirements
        }
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        if not self.initialized:
            return False
        
        self.status["active"] = True
        self.status["current_mode"] = target_data.get("mode", self.specs.modes[0])
        self.status["target_frequencies"] = target_data.get("frequencies", [])
        
        # Use neuromorphic processing to calculate effectiveness
        if self.hardware_interface:
            effectiveness = self.process_data({
                "target": target_data,
                "computation": "effectiveness_calculation"
            })
            self.status["effectiveness"] = effectiveness.get("effectiveness", 0.0)
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        return self.status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.initialized:
            return {"error": "System not initialized"}
        
        computation_type = input_data.get("computation", "")
        
        if computation_type == "effectiveness_calculation":
            # Calculate effectiveness using neuromorphic computing
            return {
                "effectiveness": 0.85,
                "coverage": 0.92,
                "power_efficiency": 0.78
            }
        elif computation_type == "threat_analysis":
            # Analyze threats using neuromorphic computing
            return {
                "threats_detected": 2,
                "threat_data": [
                    {"id": 1, "type": "radar", "frequency": 9500, "threat_level": "high"},
                    {"id": 2, "type": "communication", "frequency": 1200, "threat_level": "medium"}
                ]
            }
        
        return {"error": "Unknown computation type"}
    
    def train(self, training_data: Dict[str, Any]) -> bool:
        return True if self.initialized else False


class JammingSystem(ElectronicWarfareSystem):
    """Electronic jamming system."""
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "ALQ-250":
            specs = ElectronicWarfareSpecs(
                weight=180.0,
                volume={"length": 1.2, "width": 0.5, "height": 0.4},
                power_requirements=3500.0,
                mounting_points=["wing_tip", "fuselage"],
                ew_type=ElectronicWarfareType.JAMMING,
                frequency_range={"min": 500, "max": 18000},
                power_output=2000.0,
                coverage_angle=180.0,
                modes=["spot_jamming", "barrage_jamming", "sweep_jamming"]
            )
        else:
            raise ValueError(f"Unknown jamming system model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        self.jamming_patterns = ["noise", "deception", "smart_noise"]
        self.current_pattern = "noise"
    
    def set_jamming_pattern(self, pattern: str) -> bool:
        if pattern in self.jamming_patterns:
            self.current_pattern = pattern
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["current_pattern"] = self.current_pattern
        return status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        base_result = super().process_data(input_data)
        
        # Add jamming-specific processing
        if input_data.get("computation") == "effectiveness_calculation":
            if self.current_pattern == "smart_noise":
                base_result["effectiveness"] += 0.1
            
            # Add neuromorphic adaptive jamming capabilities
            base_result["adaptive_response"] = True
            base_result["counter_detection_probability"] = 0.25
        
        return base_result


class SignalsIntelligenceSystem(ElectronicWarfareSystem):
    """Signals intelligence (SIGINT) system."""
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "SIG-500":
            specs = ElectronicWarfareSpecs(
                weight=85.0,
                volume={"length": 0.8, "width": 0.4, "height": 0.3},
                power_requirements=1200.0,
                mounting_points=["fuselage"],
                ew_type=ElectronicWarfareType.SIGNALS_INTELLIGENCE,
                frequency_range={"min": 20, "max": 40000},
                power_output=50.0,
                coverage_angle=360.0,
                modes=["passive_collection", "active_scanning", "direction_finding"]
            )
        else:
            raise ValueError(f"Unknown SIGINT system model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        self.collection_mode = "passive"
        self.data_collected = []
    
    def set_collection_mode(self, mode: str) -> bool:
        if mode in ["passive", "active"]:
            self.collection_mode = mode
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["collection_mode"] = self.collection_mode
        status["signals_collected"] = len(self.data_collected)
        return status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        base_result = super().process_data(input_data)
        
        # Add SIGINT-specific processing
        if input_data.get("computation") == "threat_analysis":
            # Enhanced signal analysis using neuromorphic computing
            base_result["signal_classification"] = [
                {"type": "radar", "confidence": 0.98, "origin": "air-defense"},
                {"type": "communication", "confidence": 0.92, "origin": "command-post"}
            ]
            base_result["decryption_status"] = {
                "attempted": 2,
                "successful": 1
            }
        
        return base_result
"""
Adaptive countermeasure systems for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Union, Set
import numpy as np
from dataclasses import dataclass, field

from src.payload.base import NeuromorphicPayload, PayloadSpecs
from src.payload.types import (
    CountermeasureType, PayloadCategory, JammingFrequencyBand,
    ChaffType, LaserDefenseType, DecoySignatureType, EMPStrength,
    CyberAttackVector, AcousticDisruptionMode
)


@dataclass
class CountermeasureSpecs(PayloadSpecs):
    """Specifications for countermeasure systems."""
    countermeasure_type: CountermeasureType
    response_time: float  # Response time in seconds
    effectiveness_rating: float  # Effectiveness rating (0-1)
    capacity: int  # Number of deployments available
    coverage_angle: float  # Coverage angle in degrees
    
    # Advanced properties for all countermeasure types
    energy_consumption: float = 100.0  # Energy consumption in watts
    thermal_signature: float = 0.3  # Thermal signature (0-1)
    stealth_impact: float = 0.1  # Impact on platform stealth (0-1)
    cooldown_time: float = 2.0  # Time between deployments in seconds
    neuromorphic_processing_requirements: Dict[str, Any] = field(default_factory=lambda: {
        "snn_neurons": 1000,
        "learning_enabled": True,
        "adaptation_rate": 0.5
    })
    
    # Specific properties for different countermeasure types
    frequency_bands: Set[JammingFrequencyBand] = field(default_factory=set)  # For jammers
    chaff_type: Optional[ChaffType] = None  # For chaff systems
    laser_defense_type: Optional[LaserDefenseType] = None  # For laser defense
    decoy_signature_types: Set[DecoySignatureType] = field(default_factory=set)  # For decoys
    emp_strength: Optional[EMPStrength] = None  # For EMP systems
    cyber_attack_vectors: Set[CyberAttackVector] = field(default_factory=set)  # For cyber attacks
    acoustic_disruption_modes: Set[AcousticDisruptionMode] = field(default_factory=set)  # For acoustic systems
    
    # Environmental adaptation parameters
    environmental_adaptation: Dict[str, Union[List[float], float]] = field(default_factory=lambda: {
        "temperature_tolerance": [-40.0, 85.0],  # Operating temperature range in Celsius
        "altitude_effectiveness": 1.0,  # Effectiveness modifier based on altitude
        "humidity_resistance": 0.9,  # Resistance to humidity (0-1)
        "rain_effectiveness": 0.8  # Effectiveness in rain (0-1)
    })
    
    # Integration parameters
    integration_complexity: float = 0.5  # Complexity of integration (0-1)
    maintenance_hours: float = 10.0  # Required maintenance hours per 100 flight hours
    software_version: str = "1.0.0"  # Software version
    hardware_compatibility: List[str] = field(default_factory=lambda: ["standard"])
    
    def get_deployment_time(self, threat_level: float = 0.5) -> float:
        """
        Calculate deployment time based on response time and threat level.
        
        Args:
            threat_level: Threat level (0-1)
            
        Returns:
            Deployment time in seconds
        """
        # Higher threat levels result in faster deployment
        return max(0.1, self.response_time * (1.0 - (threat_level * 0.5)))
    
    def get_effectiveness_against_threat(self, threat_type: str) -> float:
        """
        Calculate effectiveness against a specific threat type.
        
        Args:
            threat_type: Type of threat
            
        Returns:
            Effectiveness rating (0-1)
        """
        # Base effectiveness
        effectiveness = self.effectiveness_rating
        
        # This would be expanded with specific logic for each countermeasure type
        # and threat type combination
        return min(1.0, effectiveness)


class AdaptiveCountermeasure(NeuromorphicPayload):
    """Base class for adaptive countermeasure systems."""
    
    def __init__(self, specs: CountermeasureSpecs, hardware_interface=None):
        super().__init__(hardware_interface)
        self.specs = specs
        self.status = {
            "active": False,
            "deployments_remaining": specs.capacity,
            "current_threat": None,
            "effectiveness": 0.0
        }
    
    def get_specifications(self) -> PayloadSpecs:
        return self.specs
    
    def calculate_impact(self) -> Dict[str, float]:
        return {
            "weight_impact": self.specs.weight,
            "drag_coefficient": 0.02,
            "power_consumption": self.specs.power_requirements
        }
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy countermeasures against a target threat.
        
        Args:
            target_data: Data about the target threat
            
        Returns:
            Success status
        """
        if not self.initialized or self.status["deployments_remaining"] <= 0:
            return False
        
        # Use neuromorphic processing for threat assessment
        response = self.process_data({
            "threat": target_data,
            "computation": "threat_response"
        })
        
        if response.get("deploy", False):
            self.status["active"] = True
            self.status["deployments_remaining"] -= 1
            self.status["current_threat"] = target_data.get("type", "unknown")
            self.status["effectiveness"] = response.get("effectiveness", 0.0)
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        return self.status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.initialized:
            return {"error": "System not initialized"}
        
        computation_type = input_data.get("computation", "")
        
        if computation_type == "threat_response":
            # Neuromorphic threat response calculation
            return {
                "deploy": True,
                "effectiveness": 0.85,
                "optimal_timing": 0.3,  # seconds
                "deployment_pattern": "adaptive"
            }
        elif computation_type == "threat_prediction":
            # Neuromorphic threat prediction
            return {
                "threats": [
                    {"type": "radar_guided", "probability": 0.75, "time_to_impact": 12.0},
                    {"type": "infrared", "probability": 0.25, "time_to_impact": 15.0}
                ],
                "recommended_action": "prepare_countermeasures"
            }
        
        return {"error": "Unknown computation type"}
    
    def train(self, training_data: Dict[str, Any]) -> bool:
        return True if self.initialized else False


class AdaptiveDecoy(AdaptiveCountermeasure):
    """Adaptive decoy countermeasure system."""
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "DECOY-X":
            specs = CountermeasureSpecs(
                weight=65.0,
                volume={"length": 0.8, "width": 0.3, "height": 0.3},
                power_requirements=500.0,
                mounting_points=["fuselage", "wing"],
                countermeasure_type=CountermeasureType.DECOY,
                response_time=0.2,
                effectiveness_rating=0.9,
                capacity=8,
                coverage_angle=270.0
            )
        else:
            raise ValueError(f"Unknown decoy model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        self.decoy_modes = ["radar", "infrared", "multi_spectral"]
        self.current_mode = "multi_spectral"
    
    def set_decoy_mode(self, mode: str) -> bool:
        if mode in self.decoy_modes:
            self.current_mode = mode
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["current_mode"] = self.current_mode
        return status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        base_result = super().process_data(input_data)
        
        if input_data.get("computation") == "threat_response":
            # Enhanced response with neuromorphic adaptation
            threat_type = input_data.get("threat", {}).get("type", "unknown")
            
            # Adapt decoy mode based on threat
            if threat_type == "radar_guided" and self.current_mode != "radar":
                self.current_mode = "radar"
            elif threat_type == "infrared" and self.current_mode != "infrared":
                self.current_mode = "infrared"
            
            # Adjust effectiveness based on mode matching
            if (threat_type == "radar_guided" and self.current_mode == "radar") or \
               (threat_type == "infrared" and self.current_mode == "infrared") or \
               (self.current_mode == "multi_spectral"):
                base_result["effectiveness"] += 0.1
            
            base_result["adaptive_response"] = True
        
        return base_result
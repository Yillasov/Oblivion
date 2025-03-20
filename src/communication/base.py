"""
Base communication system for UCAV platforms.

This module provides the foundation for all communication systems
with neuromorphic integration capabilities.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.core.integration.neuromorphic_system import NeuromorphicSystem


class CommunicationType(Enum):
    """Types of communication systems."""
    QUANTUM = "quantum"
    OPTICAL = "optical"
    MESH_NETWORK = "mesh_network"
    SATELLITE = "satellite"
    COGNITIVE_RADIO = "cognitive_radio"
    STEGANOGRAPHIC = "steganographic"
    TERAHERTZ = "terahertz"
    NEUROMORPHIC = "neuromorphic"
    SELF_HEALING = "self_healing"
    MOLECULAR = "molecular"


@dataclass
class CommunicationSpecs:
    """Specifications for communication systems."""
    weight: float  # Weight in kg
    volume: Dict[str, float]  # Volume specifications in meters
    power_requirements: float  # Power requirements in watts
    bandwidth: float  # Bandwidth in Mbps
    range: float  # Range in km
    latency: float  # Latency in ms
    encryption_level: int  # Encryption level (1-10)
    resilience_rating: float  # Resilience to interference/jamming (0-1)
    additional_specs: Dict[str, Any] = field(default_factory=dict)


class CommunicationSystem:
    """Base class for all communication systems."""
    
    def __init__(self, specs: CommunicationSpecs, hardware_interface=None):
        """
        Initialize communication system.
        
        Args:
            specs: Communication system specifications
            hardware_interface: Interface to neuromorphic hardware
        """
        self.specs = specs
        self.neuromorphic_system = NeuromorphicSystem(hardware_interface)
        self.initialized = False
        self.active = False
        self.status = {"operational": False, "channel_quality": 0.0}
    
    def initialize(self) -> bool:
        """Initialize the communication system."""
        if self.initialized:
            return True
            
        try:
            self.neuromorphic_system.initialize()
            self.initialized = True
            self.status["operational"] = True
            return True
        except Exception as e:
            self.status["error"] = str(e)
            return False
    
    def get_specifications(self) -> CommunicationSpecs:
        """Get communication system specifications."""
        return self.specs
    
    def establish_link(self, target_data: Dict[str, Any]) -> bool:
        """Establish communication link with target."""
        if not self.initialized:
            return False
        self.active = True
        return True
    
    def terminate_link(self) -> bool:
        """Terminate communication link."""
        if not self.active:
            return False
        self.active = False
        return True
    
    def send_data(self, data: Dict[str, Any]) -> bool:
        """Send data through communication link."""
        if not self.active:
            return False
        return True
    
    def receive_data(self) -> Dict[str, Any]:
        """Receive data from communication link."""
        if not self.active:
            return {"error": "Communication link not active"}
        return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of communication system."""
        self.status["active"] = self.active
        return self.status
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the communication system."""
        return {
            "effective_range": self.specs.range,
            "actual_bandwidth": self.specs.bandwidth,
            "actual_latency": self.specs.latency,
            "power_consumption": self.specs.power_requirements
        }
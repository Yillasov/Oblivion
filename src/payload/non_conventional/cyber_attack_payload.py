"""
Cyber Attack Payload implementation for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
from enum import Enum, auto
import hashlib
import random

from src.payload.base import NeuromorphicPayload, PayloadSpecs
from src.payload.types import PayloadCategory, CyberAttackVector


class CyberAttackMode(Enum):
    """Cyber attack operational modes."""
    PASSIVE_MONITORING = auto()    # Passive signal intelligence gathering
    ACTIVE_SCANNING = auto()       # Active scanning of target systems
    COMMUNICATIONS_JAMMING = auto() # Disruption of communications
    SYSTEM_INFILTRATION = auto()   # Infiltration of target systems
    DATA_EXFILTRATION = auto()     # Extraction of data from target systems
    SYSTEM_MANIPULATION = auto()   # Manipulation of target system behavior


class CyberAttackPayload(NeuromorphicPayload):
    """
    Advanced cyber attack system capable of electronic intelligence gathering,
    communications disruption, and system infiltration.
    """
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "CYBER-100":
            specs = PayloadSpecs(
                weight=15.0,
                volume={"length": 0.3, "width": 0.25, "height": 0.1},
                power_requirements=120.0,
                mounting_points=["fuselage", "internal_bay"]
                # Removed the payload_category parameter
            )
        elif model == "CYBER-300":
            specs = PayloadSpecs(
                weight=22.0,
                volume={"length": 0.4, "width": 0.3, "height": 0.15},
                power_requirements=180.0,
                mounting_points=["fuselage", "internal_bay", "wing_hardpoints"]
                # Removed the payload_category parameter
            )
        else:
            raise ValueError(f"Unknown cyber attack payload model: {model}")
            
        super().__init__(hardware_interface)
        self.specs = specs
        self.model = model
        
        # Store the payload category as a separate attribute
        self.payload_category = PayloadCategory.ELECTRONIC_WARFARE
        
        # Cyber attack system properties
        self.cyber_properties = {
            "mode": CyberAttackMode.PASSIVE_MONITORING,
            "attack_vectors": self._initialize_attack_vectors(),
            "frequency_range": {"min": 30.0, "max": 6000.0},  # MHz
            "transmission_power": 25.0 if model == "CYBER-300" else 15.0,  # Watts
            "antenna_gain": 12.0 if model == "CYBER-300" else 8.0,  # dBi
            "encryption_capabilities": self._initialize_encryption_capabilities(),
            "signal_processing": {
                "bandwidth": 500.0,  # MHz
                "sensitivity": -110.0,  # dBm
                "processing_gain": 30.0  # dB
            },
            "attack_history": [],
            "detected_systems": [],
            "infiltrated_systems": []
        }
        
        # Status tracking
        self.status = {
            "active": False,
            "current_mode": CyberAttackMode.PASSIVE_MONITORING,
            "current_vector": None,
            "target_locked": False,
            "attack_progress": 0.0,
            "attack_success_probability": 0.0,
            "power_consumption": 0.0,
            "thermal_signature": 0.2,
            "detection_risk": 0.1
        }
        
        self.initialized = True
    
    def _initialize_attack_vectors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize attack vectors with capabilities."""
        vectors = {}
        
        for vector in CyberAttackVector:
            vectors[vector.name] = {
                "enabled": True,
                "effectiveness": 0.7,
                "detection_risk": 0.3,
                "power_requirement": 10.0,  # Watts
                "cooldown_time": 30.0,  # seconds
                "last_used": 0.0,
                "success_count": 0,
                "failure_count": 0
            }
        
        # Adjust specific vectors based on model
        if self.model == "CYBER-300":
            vectors[CyberAttackVector.COMMUNICATIONS.name]["effectiveness"] = 0.85
            vectors[CyberAttackVector.NAVIGATION.name]["effectiveness"] = 0.8
            vectors[CyberAttackVector.COMMAND_CONTROL.name]["effectiveness"] = 0.75
        
        return vectors
    
    def _initialize_encryption_capabilities(self) -> Dict[str, Any]:
        """Initialize encryption capabilities."""
        capabilities = {
            "algorithms": ["AES-256", "RSA-2048", "ECC-P256", "ChaCha20-Poly1305"],
            "key_generation": True,
            "quantum_resistant": self.model == "CYBER-300",
            "secure_storage": True,
            "max_key_size": 4096 if self.model == "CYBER-300" else 2048
        }
        
        return capabilities
    
    def set_attack_mode(self, mode: CyberAttackMode) -> bool:
        """
        Set the cyber attack mode.
        
        Args:
            mode: Cyber attack mode
            
        Returns:
            Success status
        """
        if not isinstance(mode, CyberAttackMode):
            return False
            
        self.status["current_mode"] = mode
        
        # Update power consumption and detection risk based on mode
        if mode == CyberAttackMode.PASSIVE_MONITORING:
            self.status["power_consumption"] = 0.2 * self.specs.power_requirements
            self.status["detection_risk"] = 0.1
        elif mode == CyberAttackMode.ACTIVE_SCANNING:
            self.status["power_consumption"] = 0.5 * self.specs.power_requirements
            self.status["detection_risk"] = 0.4
        elif mode == CyberAttackMode.COMMUNICATIONS_JAMMING:
            self.status["power_consumption"] = 0.8 * self.specs.power_requirements
            self.status["detection_risk"] = 0.7
        elif mode == CyberAttackMode.SYSTEM_INFILTRATION:
            self.status["power_consumption"] = 0.6 * self.specs.power_requirements
            self.status["detection_risk"] = 0.5
        elif mode == CyberAttackMode.DATA_EXFILTRATION:
            self.status["power_consumption"] = 0.7 * self.specs.power_requirements
            self.status["detection_risk"] = 0.6
        elif mode == CyberAttackMode.SYSTEM_MANIPULATION:
            self.status["power_consumption"] = 0.9 * self.specs.power_requirements
            self.status["detection_risk"] = 0.8
            
        return True
    
    def select_attack_vector(self, vector: CyberAttackVector) -> bool:
        """
        Select an attack vector.
        
        Args:
            vector: Cyber attack vector
            
        Returns:
            Success status
        """
        if not isinstance(vector, CyberAttackVector):
            return False
            
        vector_name = vector.name
        if vector_name not in self.cyber_properties["attack_vectors"]:
            return False
            
        vector_data = self.cyber_properties["attack_vectors"][vector_name]
        if not vector_data["enabled"]:
            return False
            
        # Check cooldown
        current_time = time.time()
        if current_time - vector_data["last_used"] < vector_data["cooldown_time"]:
            return False
            
        self.status["current_vector"] = vector
        return True
    
    def scan_for_targets(self, scan_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Scan for potential cyber targets.
        
        Args:
            scan_parameters: Parameters for the scan
            
        Returns:
            List of detected systems
        """
        # Set to active scanning mode
        self.set_attack_mode(CyberAttackMode.ACTIVE_SCANNING)
        
        # Use neuromorphic processing for signal analysis
        scan_result = self.process_data({
            "computation": "signal_analysis",
            "frequency_range": scan_parameters.get("frequency_range", self.cyber_properties["frequency_range"]),
            "scan_duration": scan_parameters.get("duration", 10.0),
            "sensitivity": scan_parameters.get("sensitivity", self.cyber_properties["signal_processing"]["sensitivity"])
        })
        
        # Process detected systems
        detected_systems = scan_result.get("detected_systems", [])
        self.cyber_properties["detected_systems"] = detected_systems
        
        return detected_systems
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy cyber attack against a target.
        
        Args:
            target_data: Data about the target
            
        Returns:
            Success status
        """
        if not self.initialized:
            return False
            
        # Check if we have a valid attack vector
        if self.status["current_vector"] is None:
            return False
            
        vector_name = self.status["current_vector"].name
        vector_data = self.cyber_properties["attack_vectors"][vector_name]
        
        # Use neuromorphic processing to optimize attack
        attack_result = self.process_data({
            "computation": "cyber_attack_optimization",
            "target": target_data,
            "attack_vector": vector_name,
            "attack_mode": self.status["current_mode"].name
        })
        
        # Calculate success probability
        base_probability = vector_data["effectiveness"]
        target_vulnerability = target_data.get("vulnerability", 0.5)
        optimization_factor = attack_result.get("optimization_factor", 1.0)
        
        success_probability = min(0.95, base_probability * target_vulnerability * optimization_factor)
        self.status["attack_success_probability"] = success_probability
        
        # Simulate attack outcome
        attack_successful = random.random() < success_probability
        
        # Update attack history
        self.cyber_properties["attack_history"].append({
            "timestamp": time.time(),
            "target_id": target_data.get("id", "unknown"),
            "target_type": target_data.get("type", "unknown"),
            "vector": vector_name,
            "mode": self.status["current_mode"].name,
            "success": attack_successful,
            "probability": success_probability
        })
        
        # Update vector statistics
        vector_data["last_used"] = time.time()
        if attack_successful:
            vector_data["success_count"] += 1
            
            # Add to infiltrated systems if applicable
            if self.status["current_mode"] in [CyberAttackMode.SYSTEM_INFILTRATION, 
                                              CyberAttackMode.DATA_EXFILTRATION,
                                              CyberAttackMode.SYSTEM_MANIPULATION]:
                self.cyber_properties["infiltrated_systems"].append({
                    "id": target_data.get("id", f"system_{len(self.cyber_properties['infiltrated_systems'])}"),
                    "type": target_data.get("type", "unknown"),
                    "infiltration_time": time.time(),
                    "access_level": attack_result.get("access_level", "basic"),
                    "data_collected": attack_result.get("data_collected", {})
                })
        else:
            vector_data["failure_count"] += 1
        
        return attack_successful
    
    def update(self, dt: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update cyber attack system state over time.
        
        Args:
            dt: Time step in seconds
            environment_data: Environmental data
            
        Returns:
            Updated status
        """
        # Passive monitoring in background
        if self.status["active"] and self.status["current_mode"] == CyberAttackMode.PASSIVE_MONITORING:
            # Occasionally detect new systems
            if random.random() < 0.05 * dt:
                new_system = {
                    "id": f"system_{len(self.cyber_properties['detected_systems'])}",
                    "type": random.choice(["communications", "radar", "navigation", "command_control"]),
                    "frequency": random.uniform(self.cyber_properties["frequency_range"]["min"],
                                              self.cyber_properties["frequency_range"]["max"]),
                    "signal_strength": random.uniform(-100, -60),  # dBm
                    "detection_time": time.time(),
                    "vulnerability": random.uniform(0.2, 0.8)
                }
                self.cyber_properties["detected_systems"].append(new_system)
        
        # Update infiltrated systems
        for system in self.cyber_properties["infiltrated_systems"]:
            # Chance of losing access over time
            if random.random() < 0.01 * dt:
                system["access_lost"] = True
                system["access_lost_time"] = time.time()
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current cyber attack system status."""
        return {
            "active": self.status["active"],
            "current_mode": self.status["current_mode"].name,
            "current_vector": self.status["current_vector"].name if self.status["current_vector"] else None,
            "attack_progress": self.status["attack_progress"],
            "attack_success_probability": self.status["attack_success_probability"],
            "power_consumption": self.status["power_consumption"],
            "thermal_signature": self.status["thermal_signature"],
            "detection_risk": self.status["detection_risk"],
            "detected_systems_count": len(self.cyber_properties["detected_systems"]),
            "infiltrated_systems_count": len(self.cyber_properties["infiltrated_systems"]),
            "attack_history_count": len(self.cyber_properties["attack_history"])
        }
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using neuromorphic computing.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processing results
        """
        computation_type = input_data.get("computation", "")
        result = {}
        
        if computation_type == "signal_analysis":
            # Simulate signal analysis with neuromorphic processing
            frequency_range = input_data.get("frequency_range", self.cyber_properties["frequency_range"])
            scan_duration = input_data.get("scan_duration", 10.0)
            sensitivity = input_data.get("sensitivity", self.cyber_properties["signal_processing"]["sensitivity"])
            
            # Generate simulated detected systems
            detected_systems = []
            num_systems = random.randint(1, 5)
            
            for i in range(num_systems):
                system_type = random.choice(["communications", "radar", "navigation", "command_control"])
                
                # Generate frequency based on system type
                if system_type == "communications":
                    freq = random.uniform(30, 500)  # MHz
                elif system_type == "radar":
                    freq = random.uniform(1000, 6000)  # MHz
                elif system_type == "navigation":
                    freq = random.uniform(900, 1200)  # MHz
                else:
                    freq = random.uniform(100, 400)  # MHz
                
                # Only include if within scan range
                if frequency_range["min"] <= freq <= frequency_range["max"]:
                    detected_systems.append({
                        "id": f"system_{len(detected_systems)}",
                        "type": system_type,
                        "frequency": freq,
                        "signal_strength": random.uniform(-100, -60),  # dBm
                        "detection_time": time.time(),
                        "vulnerability": random.uniform(0.2, 0.8),
                        "encryption_detected": random.random() > 0.3,
                        "modulation": random.choice(["am", "fm", "qpsk", "ofdm", "fsk"])
                    })
            
            result["detected_systems"] = detected_systems
            result["scan_duration"] = scan_duration
            result["scan_bandwidth"] = frequency_range["max"] - frequency_range["min"]
            
        elif computation_type == "cyber_attack_optimization":
            # Simulate attack optimization with neuromorphic processing
            target = input_data.get("target", {})
            attack_vector = input_data.get("attack_vector", "")
            attack_mode = input_data.get("attack_mode", "")
            
            # Calculate optimization factor based on target and attack vector
            optimization_factor = 1.0
            
            # Match attack vector to target type for better effectiveness
            target_type = target.get("type", "unknown")
            if (attack_vector == CyberAttackVector.COMMUNICATIONS.name and 
                target_type == "communications"):
                optimization_factor = 1.3
            elif (attack_vector == CyberAttackVector.NAVIGATION.name and 
                  target_type == "navigation"):
                optimization_factor = 1.3
            elif (attack_vector == CyberAttackVector.SENSOR.name and 
                  target_type == "radar"):
                optimization_factor = 1.2
            elif (attack_vector == CyberAttackVector.COMMAND_CONTROL.name and 
                  target_type == "command_control"):
                optimization_factor = 1.4
                
            # Adjust for encryption if detected
            if target.get("encryption_detected", False):
                encryption_type = target.get("encryption_type", "unknown")
                if encryption_type in self.cyber_properties["encryption_capabilities"]["algorithms"]:
                    optimization_factor *= 0.9  # Slight penalty for known encryption
                else:
                    optimization_factor *= 0.7  # Larger penalty for unknown encryption
            
            # Generate attack parameters
            result["optimization_factor"] = optimization_factor
            result["recommended_frequency"] = target.get("frequency", 0.0)
            result["power_level"] = min(1.0, 0.7 + random.random() * 0.3)
            result["estimated_duration"] = random.uniform(5.0, 20.0)  # seconds
            
            # For successful attacks, determine access level and data
            if random.random() < 0.7 * optimization_factor:
                access_levels = ["basic", "user", "admin", "system"]
                access_weights = [0.4, 0.3, 0.2, 0.1]
                result["access_level"] = random.choices(access_levels, access_weights)[0]
                
                # Simulate collected data
                if attack_mode == CyberAttackMode.DATA_EXFILTRATION.name:
                    result["data_collected"] = {
                        "size": random.randint(1, 100),  # MB
                        "type": random.choice(["logs", "configurations", "credentials", "communications"]),
                        "encryption_status": "encrypted" if random.random() > 0.7 else "plaintext",
                        "integrity": random.uniform(0.5, 1.0)
                    }
        
        return result
    
    def calculate_impact(self) -> Dict[str, float]:
        """
        Calculate impact of the cyber attack system on the aircraft.
        
        Returns:
            Impact metrics
        """
        # Base impact
        impact = {
            "weight_impact": self.specs.weight,
            "power_consumption": self.status["power_consumption"],
            "thermal_signature": self.status["thermal_signature"],
            "detection_risk": self.status["detection_risk"],
            "drag_coefficient": 0.01  # Minimal drag as mostly internal
        }
        
        return impact
    
    def activate(self) -> bool:
        """
        Activate the cyber attack system.
        
        Returns:
            Success status
        """
        if not self.initialized:
            return False
            
        self.status["active"] = True
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the cyber attack system.
        
        Returns:
            Success status
        """
        self.status["active"] = False
        return True
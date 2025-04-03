"""
Holographic Decoy System implementation for UCAV platforms.
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

from src.payload.non_conventional.countermeasures import AdaptiveCountermeasure, CountermeasureSpecs
from src.payload.types import CountermeasureType, DecoySignatureType


class HolographicMode(Enum):
    """Holographic projection modes."""
    MIRROR = auto()  # Mirror of the host aircraft
    SWARM = auto()   # Multiple decoys simulating a swarm
    PHANTOM = auto() # Different aircraft type
    SCATTER = auto() # Multiple scattered signatures
    ADAPTIVE = auto() # Adaptive projection based on threat


class HolographicDecoy(AdaptiveCountermeasure):
    """
    Advanced holographic decoy system that projects realistic
    multi-spectral signatures to confuse enemy targeting systems.
    """
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "HD-200":
            specs = CountermeasureSpecs(
                weight=35.0,
                volume={"length": 0.4, "width": 0.3, "height": 0.2},
                power_requirements=150.0,
                mounting_points=["fuselage", "wing"],
                countermeasure_type=CountermeasureType.HOLOGRAPHIC_DECOY,
                response_time=0.002,
                effectiveness_rating=0.75,
                capacity=50,
                coverage_angle=360.0,
                energy_consumption=120.0,
                thermal_signature=0.4,
                stealth_impact=0.3,
                cooldown_time=5.0
            )
        elif model == "HD-500":
            specs = CountermeasureSpecs(
                weight=45.0,
                volume={"length": 0.5, "width": 0.35, "height": 0.25},
                power_requirements=220.0,
                mounting_points=["fuselage", "internal_bay"],
                countermeasure_type=CountermeasureType.HOLOGRAPHIC_DECOY,
                response_time=0.001,
                effectiveness_rating=0.85,
                capacity=100,
                coverage_angle=360.0,
                energy_consumption=180.0,
                thermal_signature=0.5,
                stealth_impact=0.35,
                cooldown_time=3.0
            )
        else:
            raise ValueError(f"Unknown holographic decoy model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        
        # Holographic decoy properties
        self.decoy_properties = {
            "mode": HolographicMode.MIRROR,
            "active_decoys": 0,
            "max_decoys": 4 if model == "HD-500" else 2,
            "projection_range": 500.0 if model == "HD-500" else 300.0,  # meters
            "signature_types": self._initialize_signature_types(),
            "decoy_positions": [],
            "last_update_time": 0.0,
            "power_level": 0.8,
            "coherence": 0.9,  # Hologram stability/realism
            "host_aircraft_signature": {}
        }
    
    def _initialize_signature_types(self) -> Dict[DecoySignatureType, Dict[str, Any]]:
        """Initialize signature type properties."""
        signatures = {}
        
        # Radar signature
        signatures[DecoySignatureType.RADAR] = {
            "effectiveness": 0.85,
            "power_consumption": 1.0,
            "detection_probability": 0.3,
            "frequency_range": (1.0, 18.0)  # GHz
        }
        
        # Infrared signature
        signatures[DecoySignatureType.INFRARED] = {
            "effectiveness": 0.8,
            "power_consumption": 0.9,
            "detection_probability": 0.35,
            "temperature_range": (20, 100)  # Celsius
        }
        
        # Visual signature
        signatures[DecoySignatureType.VISUAL] = {
            "effectiveness": 0.7,
            "power_consumption": 0.8,
            "detection_probability": 0.4,
            "light_conditions": ["day", "dusk", "night"]
        }
        
        # Multi-spectral signature
        signatures[DecoySignatureType.MULTI_SPECTRAL] = {
            "effectiveness": 0.9,
            "power_consumption": 1.2,
            "detection_probability": 0.25,
            "bands": ["visible", "near-IR", "thermal-IR", "radar"]
        }
        
        # Holographic signature
        signatures[DecoySignatureType.HOLOGRAPHIC] = {
            "effectiveness": 0.95,
            "power_consumption": 1.5,
            "detection_probability": 0.2,
            "projection_quality": 0.9
        }
        
        return signatures
    
    def set_decoy_mode(self, mode: HolographicMode) -> bool:
        """
        Set the holographic decoy mode.
        
        Args:
            mode: Decoy projection mode
            
        Returns:
            Success status
        """
        if not isinstance(mode, HolographicMode):
            return False
            
        self.decoy_properties["mode"] = mode
        return True
    
    def set_power_level(self, power_level: float) -> bool:
        """
        Set the power level for the holographic projector.
        
        Args:
            power_level: Power level as a percentage (0-100)
            
        Returns:
            Success status
        """
        if 0 <= power_level <= 100:
            self.decoy_properties["power_level"] = power_level / 100.0
            return True
        return False
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy holographic decoys against a target threat.
        
        Args:
            target_data: Data about the target threat
            
        Returns:
            Success status
        """
        # First check if base deployment is successful
        if not super().deploy(target_data):
            return False
        
        # Use neuromorphic processing to optimize decoy deployment
        decoy_result = self.process_data({
            "threat": target_data,
            "computation": "decoy_optimization",
            "current_mode": self.decoy_properties["mode"],
            "host_signature": self.decoy_properties["host_aircraft_signature"]
        })
        
        # Generate decoy positions based on mode
        self._generate_decoy_positions(target_data)
        
        # Set active decoys count
        mode = self.decoy_properties["mode"]
        if mode == HolographicMode.MIRROR:
            self.decoy_properties["active_decoys"] = 1
        elif mode == HolographicMode.SWARM:
            self.decoy_properties["active_decoys"] = min(self.decoy_properties["max_decoys"], 4)
        elif mode == HolographicMode.SCATTER:
            self.decoy_properties["active_decoys"] = min(self.decoy_properties["max_decoys"], 3)
        else:
            self.decoy_properties["active_decoys"] = 1
        
        self.decoy_properties["last_update_time"] = time.time()
        
        return True
    
    def _generate_decoy_positions(self, target_data: Dict[str, Any]) -> None:
        """
        Generate positions for holographic decoys.
        
        Args:
            target_data: Data about the target threat
        """
        mode = self.decoy_properties["mode"]
        max_range = self.decoy_properties["projection_range"]
        positions = []
        
        # Get host position
        host_pos = target_data.get("host_position", [0, 0, 0])
        threat_pos = target_data.get("threat_position", [1000, 1000, 1000])
        
        # Calculate threat direction
        threat_dir = np.array(threat_pos) - np.array(host_pos)
        if np.linalg.norm(threat_dir) > 0:
            threat_dir = threat_dir / np.linalg.norm(threat_dir)
        
        if mode == HolographicMode.MIRROR:
            # Single decoy opposite to the threat
            offset = -threat_dir * max_range * 0.5
            positions.append((host_pos[0] + offset[0], host_pos[1] + offset[1], host_pos[2] + offset[2]))
            
        elif mode == HolographicMode.SWARM:
            # Multiple decoys in formation
            for i in range(self.decoy_properties["max_decoys"]):
                angle = 2 * np.pi * i / self.decoy_properties["max_decoys"]
                x_offset = np.cos(angle) * max_range * 0.3
                y_offset = np.sin(angle) * max_range * 0.3
                z_offset = np.random.uniform(-50, 50)
                positions.append((host_pos[0] + x_offset, host_pos[1] + y_offset, host_pos[2] + z_offset))
                
        elif mode == HolographicMode.SCATTER:
            # Scattered random decoys
            for i in range(self.decoy_properties["max_decoys"]):
                x_offset = np.random.uniform(-max_range, max_range) * 0.5
                y_offset = np.random.uniform(-max_range, max_range) * 0.5
                z_offset = np.random.uniform(-100, 100)
                positions.append((host_pos[0] + x_offset, host_pos[1] + y_offset, host_pos[2] + z_offset))
                
        elif mode == HolographicMode.PHANTOM:
            # Single decoy with different signature
            offset = threat_dir * max_range * 0.4
            positions.append((host_pos[0] + offset[0], host_pos[1] + offset[1], host_pos[2] + offset[2]))
            
        elif mode == HolographicMode.ADAPTIVE:
            # Adaptive positioning based on threat
            # Position between host and threat
            offset = threat_dir * max_range * 0.3
            positions.append((host_pos[0] + offset[0], host_pos[1] + offset[1], host_pos[2] + offset[2]))
            
            # Position to the side
            perp = np.array([-threat_dir[1], threat_dir[0], 0])
            if np.linalg.norm(perp) > 0:
                perp = perp / np.linalg.norm(perp)
                side_offset = perp * max_range * 0.4
                positions.append((host_pos[0] + side_offset[0], host_pos[1] + side_offset[1], host_pos[2]))
        
        self.decoy_properties["decoy_positions"] = positions
    
    def update(self, dt: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update holographic decoy system state over time.
        
        Args:
            dt: Time step in seconds
            environment_data: Environmental data
            
        Returns:
            Updated status
        """
        if not self.status["active"]:
            return self.get_status()
        
        # Update decoy positions if needed
        if environment_data and "host_position" in environment_data:
            self._update_decoy_positions(environment_data)
        
        # Decrease effectiveness over time (decoys become less convincing)
        elapsed = time.time() - self.decoy_properties["last_update_time"]
        if elapsed > 10.0:  # Every 10 seconds
            self.decoy_properties["coherence"] *= 0.95
            self.decoy_properties["last_update_time"] = time.time()
            
            # If coherence drops too low, decoys become ineffective
            if self.decoy_properties["coherence"] < 0.3:
                self.deactivate()
        
        return self.get_status()
    
    def _update_decoy_positions(self, environment_data: Dict[str, Any]) -> None:
        """
        Update decoy positions based on host movement.
        
        Args:
            environment_data: Environmental data including host position
        """
        host_pos = environment_data.get("host_position", [0, 0, 0])
        host_vel = environment_data.get("host_velocity", [0, 0, 0])
        
        # Update positions based on mode
        mode = self.decoy_properties["mode"]
        positions = self.decoy_properties["decoy_positions"]
        updated_positions = []
        
        if mode == HolographicMode.MIRROR:
            # Mirror decoy follows host with offset
            if positions:
                offset = np.array(positions[0]) - np.array(host_pos)
                updated_positions.append((host_pos[0] + offset[0], host_pos[1] + offset[1], host_pos[2] + offset[2]))
                
        elif mode == HolographicMode.SWARM:
            # Swarm follows with formation integrity
            for i, pos in enumerate(positions):
                # Maintain relative position with small random movement
                rel_pos = np.array(pos) - np.array(host_pos)
                jitter = np.random.uniform(-10, 10, 3)
                updated_positions.append((
                    host_pos[0] + rel_pos[0] + jitter[0],
                    host_pos[1] + rel_pos[1] + jitter[1],
                    host_pos[2] + rel_pos[2] + jitter[2]
                ))
                
        elif mode == HolographicMode.SCATTER:
            # Scattered decoys move semi-independently
            for pos in positions:
                # Random movement with some correlation to host
                new_pos = (
                    pos[0] + host_vel[0] * 0.8 + np.random.uniform(-20, 20),
                    pos[1] + host_vel[1] * 0.8 + np.random.uniform(-20, 20),
                    pos[2] + host_vel[2] * 0.8 + np.random.uniform(-10, 10)
                )
                updated_positions.append(new_pos)
                
        else:  # PHANTOM or ADAPTIVE
            # Follow host movement with offset
            for pos in positions:
                rel_pos = np.array(pos) - np.array(host_pos)
                updated_positions.append((
                    host_pos[0] + rel_pos[0],
                    host_pos[1] + rel_pos[1],
                    host_pos[2] + rel_pos[2]
                ))
        
        self.decoy_properties["decoy_positions"] = updated_positions
    
    def get_status(self) -> Dict[str, Any]:
        """Get current holographic decoy status."""
        status = super().get_status()
        status.update({
            "decoy_properties": self.decoy_properties,
            "active_decoys": self.decoy_properties["active_decoys"],
            "coherence": self.decoy_properties["coherence"],
            "mode": self.decoy_properties["mode"]
        })
        return status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using neuromorphic computing.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processing results
        """
        base_result = super().process_data(input_data)
        
        computation_type = input_data.get("computation", "")
        
        if computation_type == "decoy_optimization":
            # Neuromorphic decoy optimization
            threat = input_data.get("threat", {})
            threat_type = threat.get("type", "unknown")
            threat_sensors = threat.get("sensors", [])
            
            # Determine optimal decoy mode based on threat
            optimal_mode = HolographicMode.MIRROR
            if threat_type == "missile_lock":
                optimal_mode = HolographicMode.SCATTER
            elif threat_type == "radar_tracking":
                optimal_mode = HolographicMode.SWARM
            elif threat_type == "visual_tracking":
                optimal_mode = HolographicMode.PHANTOM
            elif threat_type == "multi_sensor":
                optimal_mode = HolographicMode.ADAPTIVE
            
            # Calculate signature effectiveness against threat sensors
            signature_effectiveness = {}
            for sig_type, properties in self.decoy_properties["signature_types"].items():
                effectiveness = properties["effectiveness"]
                
                # Adjust effectiveness based on threat sensors
                if sig_type == DecoySignatureType.RADAR and "radar" in threat_sensors:
                    effectiveness *= 1.2
                elif sig_type == DecoySignatureType.INFRARED and "infrared" in threat_sensors:
                    effectiveness *= 1.1
                elif sig_type == DecoySignatureType.VISUAL and "optical" in threat_sensors:
                    effectiveness *= 0.9  # Visual is harder to spoof
                
                signature_effectiveness[sig_type] = min(1.0, effectiveness)
            
            # Calculate overall effectiveness
            overall_effectiveness = sum(signature_effectiveness.values()) / len(signature_effectiveness)
            
            # Add results to base result
            base_result["optimal_mode"] = optimal_mode
            base_result["signature_effectiveness"] = signature_effectiveness
            base_result["overall_effectiveness"] = overall_effectiveness
            
        return base_result
    
    def set_coherence(self, coherence: float) -> bool:
        """
        Set the coherence level for holographic projections.
        
        Args:
            coherence: Coherence level (0.0-1.0)
            
        Returns:
            Success status
        """
        if 0.0 <= coherence <= 1.0:
            self.decoy_properties["coherence"] = coherence
            return True
        return False
    
    def set_host_signature(self, signature_data: Dict[str, Any]) -> bool:
        """
        Set the host aircraft signature for accurate decoy generation.
        
        Args:
            signature_data: Host aircraft signature data
            
        Returns:
            Success status
        """
        if not signature_data:
            return False
            
        self.decoy_properties["host_aircraft_signature"] = signature_data
        return True
    
    def set_max_decoys(self, count: int) -> bool:
        """
        Set the maximum number of simultaneous decoys.
        
        Args:
            count: Maximum decoy count
            
        Returns:
            Success status
        """
        model_max = 4 if self.model == "HD-500" else 2
        if 1 <= count <= model_max:
            self.decoy_properties["max_decoys"] = count
            return True
        return False
    
    def deactivate(self) -> bool:
        """Deactivate the holographic decoy system."""
        # Instead of calling super().deactivate(), directly update the status
        self.status["active"] = False
        self.decoy_properties["active_decoys"] = 0
        self.decoy_properties["coherence"] = 0.9  # Reset coherence
        return True
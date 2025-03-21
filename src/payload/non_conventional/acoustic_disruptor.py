"""
Acoustic Wave Disruptor implementation for UCAV platforms.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
from enum import Enum, auto

from src.payload.non_conventional.countermeasures import AdaptiveCountermeasure, CountermeasureSpecs
from src.payload.types import CountermeasureType, AcousticDisruptionMode


class FrequencyProfile(Enum):
    """Acoustic frequency profiles for different targets."""
    SENSOR_DISRUPTION = auto()      # Disrupt acoustic sensors
    COMMUNICATION_JAMMING = auto()  # Jam communication systems
    STRUCTURAL_RESONANCE = auto()   # Target structural resonance frequencies
    PERSONNEL_DETERRENT = auto()    # Non-lethal personnel deterrent
    UNDERWATER_PROPAGATION = auto() # Optimized for underwater propagation
    STEALTH_MODE = auto()           # Low-detectability operation


class AcousticWaveDisruptor(AdaptiveCountermeasure):
    """
    Advanced acoustic wave disruptor that can generate targeted sound waves
    to disrupt enemy sensors, communications, and physical structures.
    """
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "AWD-100":
            specs = CountermeasureSpecs(
                weight=18.0,
                volume={"length": 0.4, "width": 0.3, "height": 0.2},
                power_requirements=75.0,
                mounting_points=["fuselage", "wing_hardpoints"],
                countermeasure_type=CountermeasureType.ACOUSTIC_WAVE_DISRUPTOR,  # Fixed: Changed from ACOUSTIC to ACOUSTIC_WAVE_DISRUPTOR
                response_time=0.02,
                effectiveness_rating=0.75,
                capacity=50,  # Number of disruption events
                coverage_angle=120.0,
                energy_consumption=60.0,
                thermal_signature=0.3,
                stealth_impact=0.4,
                cooldown_time=2.0
            )
        elif model == "AWD-300":
            specs = CountermeasureSpecs(
                weight=25.0,
                volume={"length": 0.5, "width": 0.4, "height": 0.25},
                power_requirements=120.0,
                mounting_points=["fuselage", "wing_hardpoints", "tail"],
                countermeasure_type=CountermeasureType.ACOUSTIC_WAVE_DISRUPTOR,  # Fixed: Changed from ACOUSTIC to ACOUSTIC_WAVE_DISRUPTOR
                response_time=0.01,
                effectiveness_rating=0.85,
                capacity=80,  # Number of disruption events
                coverage_angle=180.0,
                energy_consumption=90.0,
                thermal_signature=0.4,
                stealth_impact=0.5,
                cooldown_time=1.5
            )
        else:
            raise ValueError(f"Unknown acoustic disruptor model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        
        # Acoustic disruptor properties
        self.acoustic_properties = {
            "mode": AcousticDisruptionMode.TARGETED,
            "frequency_profile": FrequencyProfile.SENSOR_DISRUPTION,
            "current_frequency": 0.0,  # Hz
            "frequency_range": self._initialize_frequency_range(),
            "power_level": 0.8,  # 0.0-1.0
            "pulse_duration": 0.5,  # seconds
            "modulation_pattern": "sine",  # sine, square, sawtooth, etc.
            "beam_width": 15.0 if model == "AWD-300" else 30.0,  # degrees
            "max_range": 500.0 if model == "AWD-300" else 300.0,  # meters
            "last_disruption_time": 0.0,
            "disruption_count": 0,
            "effectiveness_history": []
        }
        
        # Status tracking
        self.status.update({
            "remaining_capacity": specs.capacity,
            "cooldown_remaining": 0.0,
            "active": False
        })
    
    def _initialize_frequency_range(self) -> Dict[str, Dict[str, float]]:
        """Initialize frequency ranges for different profiles."""
        ranges = {}
        
        # Frequency ranges in Hz
        ranges[FrequencyProfile.SENSOR_DISRUPTION.name] = {
            "min": 15000.0,  # 15 kHz
            "max": 40000.0,  # 40 kHz
            "optimal": 25000.0  # 25 kHz
        }
        
        ranges[FrequencyProfile.COMMUNICATION_JAMMING.name] = {
            "min": 100.0,  # 100 Hz
            "max": 5000.0,  # 5 kHz
            "optimal": 1500.0  # 1.5 kHz
        }
        
        ranges[FrequencyProfile.STRUCTURAL_RESONANCE.name] = {
            "min": 5.0,  # 5 Hz
            "max": 100.0,  # 100 Hz
            "optimal": 30.0  # 30 Hz
        }
        
        ranges[FrequencyProfile.PERSONNEL_DETERRENT.name] = {
            "min": 2000.0,  # 2 kHz
            "max": 15000.0,  # 15 kHz
            "optimal": 8000.0  # 8 kHz
        }
        
        ranges[FrequencyProfile.UNDERWATER_PROPAGATION.name] = {
            "min": 50.0,  # 50 Hz
            "max": 1000.0,  # 1 kHz
            "optimal": 200.0  # 200 Hz
        }
        
        ranges[FrequencyProfile.STEALTH_MODE.name] = {
            "min": 5.0,  # 5 Hz
            "max": 50.0,  # 50 Hz
            "optimal": 20.0  # 20 Hz
        }
        
        return ranges
    
    def set_disruption_mode(self, mode: AcousticDisruptionMode) -> bool:
        """
        Set the acoustic disruption mode.
        
        Args:
            mode: Acoustic disruption mode
            
        Returns:
            Success status
        """
        if not isinstance(mode, AcousticDisruptionMode):
            return False
            
        self.acoustic_properties["mode"] = mode
        return True
    
    def set_frequency_profile(self, profile: FrequencyProfile) -> bool:
        """
        Set the frequency profile for the acoustic disruptor.
        
        Args:
            profile: Frequency profile
            
        Returns:
            Success status
        """
        if not isinstance(profile, FrequencyProfile):
            return False
            
        self.acoustic_properties["frequency_profile"] = profile
        
        # Set current frequency to optimal for this profile
        profile_name = profile.name
        if profile_name in self.acoustic_properties["frequency_range"]:
            self.acoustic_properties["current_frequency"] = self.acoustic_properties["frequency_range"][profile_name]["optimal"]
            
        return True
    
    def set_custom_frequency(self, frequency: float) -> bool:
        """
        Set a custom frequency for the acoustic disruptor.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Success status
        """
        if frequency <= 0:
            return False
            
        self.acoustic_properties["current_frequency"] = frequency
        return True
    
    def set_power_level(self, power_level: float) -> bool:
        """
        Set the power level for the acoustic disruptor.
        
        Args:
            power_level: Power level (0.0-1.0)
            
        Returns:
            Success status
        """
        if not 0.0 <= power_level <= 1.0:
            return False
            
        self.acoustic_properties["power_level"] = power_level
        return True
    
    def set_pulse_duration(self, duration: float) -> bool:
        """
        Set the pulse duration for the acoustic disruptor.
        
        Args:
            duration: Pulse duration in seconds
            
        Returns:
            Success status
        """
        if duration <= 0:
            return False
            
        self.acoustic_properties["pulse_duration"] = duration
        return True
    
    def set_modulation_pattern(self, pattern: str) -> bool:
        """
        Set the modulation pattern for the acoustic disruptor.
        
        Args:
            pattern: Modulation pattern (sine, square, sawtooth, etc.)
            
        Returns:
            Success status
        """
        valid_patterns = ["sine", "square", "sawtooth", "triangle", "noise", "chirp"]
        if pattern not in valid_patterns:
            return False
            
        self.acoustic_properties["modulation_pattern"] = pattern
        return True
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy acoustic disruption against a target.
        
        Args:
            target_data: Data about the target
            
        Returns:
            Success status
        """
        # First check if base deployment is successful
        if not super().deploy(target_data):
            return False
        
        # Check if we have capacity remaining
        if self.status["remaining_capacity"] <= 0:
            return False
            
        # Check cooldown
        if self.status["cooldown_remaining"] > 0:
            return False
            
        # Use neuromorphic processing to optimize acoustic disruption
        disruption_result = self.process_data({
            "target": target_data,
            "computation": "acoustic_optimization",
            "current_mode": self.acoustic_properties["mode"],
            "current_frequency": self.acoustic_properties["current_frequency"],
            "current_profile": self.acoustic_properties["frequency_profile"]
        })
        
        # Apply optimized parameters if available
        if "optimal_frequency" in disruption_result:
            self.acoustic_properties["current_frequency"] = disruption_result["optimal_frequency"]
            
        if "optimal_mode" in disruption_result:
            self.acoustic_properties["mode"] = disruption_result["optimal_mode"]
            
        if "optimal_modulation" in disruption_result:
            self.acoustic_properties["modulation_pattern"] = disruption_result["optimal_modulation"]
        
        # Deploy acoustic disruption
        self.status["remaining_capacity"] -= 1
        self.acoustic_properties["disruption_count"] += 1
        self.acoustic_properties["last_disruption_time"] = time.time()
        self.status["active"] = True
        
        # Start cooldown
        self.status["cooldown_remaining"] = self.specs.cooldown_time
        
        # Record effectiveness for learning
        if "effectiveness_estimate" in disruption_result:
            self.acoustic_properties["effectiveness_history"].append({
                "mode": self.acoustic_properties["mode"].name,
                "frequency": self.acoustic_properties["current_frequency"],
                "profile": self.acoustic_properties["frequency_profile"].name,
                "target_type": target_data.get("type", "unknown"),
                "effectiveness": disruption_result["effectiveness_estimate"],
                "timestamp": time.time()
            })
        
        return True
    
    def update(self, dt: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update acoustic disruptor state over time.
        
        Args:
            dt: Time step in seconds
            environment_data: Environmental data
            
        Returns:
            Updated status
        """
        # Update cooldown if active
        if self.status["cooldown_remaining"] > 0:
            self.status["cooldown_remaining"] = max(0.0, self.status["cooldown_remaining"] - dt)
            
            # Deactivate after cooldown
            if self.status["cooldown_remaining"] <= 0:
                self.status["active"] = False
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current acoustic disruptor status."""
        status = super().get_status()
        status.update({
            "acoustic_properties": self.acoustic_properties,
            "remaining_capacity": self.status["remaining_capacity"],
            "cooldown_remaining": self.status["cooldown_remaining"],
            "active": self.status["active"]
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
        
        if computation_type == "acoustic_optimization":
            # Neuromorphic acoustic disruption optimization
            target = input_data.get("target", {})
            current_mode = input_data.get("current_mode", AcousticDisruptionMode.TARGETED)
            current_frequency = input_data.get("current_frequency", 1000.0)
            current_profile = input_data.get("current_profile", FrequencyProfile.SENSOR_DISRUPTION)
            
            # Extract target information
            target_type = target.get("type", "unknown")
            target_distance = target.get("distance", 100.0)
            target_environment = target.get("environment", "air")
            target_vulnerabilities = target.get("vulnerabilities", [])
            
            # Determine optimal parameters based on target
            optimal_frequency = current_frequency
            optimal_mode = current_mode
            optimal_modulation = self.acoustic_properties["modulation_pattern"]
            effectiveness_estimate = 0.6  # Base effectiveness
            
            # Optimize frequency based on target type
            if target_type == "sensor_array":
                optimal_frequency = self.acoustic_properties["frequency_range"][FrequencyProfile.SENSOR_DISRUPTION.name]["optimal"]
                optimal_mode = AcousticDisruptionMode.TARGETED
                optimal_modulation = "chirp"
                effectiveness_estimate = 0.85
                
            elif target_type == "communication_system":
                optimal_frequency = self.acoustic_properties["frequency_range"][FrequencyProfile.COMMUNICATION_JAMMING.name]["optimal"]
                optimal_mode = AcousticDisruptionMode.BROADBAND
                optimal_modulation = "noise"
                effectiveness_estimate = 0.8
                
            elif target_type == "structure":
                optimal_frequency = self.acoustic_properties["frequency_range"][FrequencyProfile.STRUCTURAL_RESONANCE.name]["optimal"]
                optimal_mode = AcousticDisruptionMode.RESONANT
                optimal_modulation = "sine"
                effectiveness_estimate = 0.7
                
            elif target_type == "personnel":
                optimal_frequency = self.acoustic_properties["frequency_range"][FrequencyProfile.PERSONNEL_DETERRENT.name]["optimal"]
                optimal_mode = AcousticDisruptionMode.PULSED
                optimal_modulation = "square"
                effectiveness_estimate = 0.9
                
            elif target_type == "underwater":
                optimal_frequency = self.acoustic_properties["frequency_range"][FrequencyProfile.UNDERWATER_PROPAGATION.name]["optimal"]
                optimal_mode = AcousticDisruptionMode.CONTINUOUS
                optimal_modulation = "sine"
                effectiveness_estimate = 0.75
                
            # Adjust for distance
            if target_distance > self.acoustic_properties["max_range"] * 0.8:
                effectiveness_estimate *= 0.7
                
            # Adjust for environment
            if target_environment == "underwater" and current_profile != FrequencyProfile.UNDERWATER_PROPAGATION:
                effectiveness_estimate *= 0.5
                optimal_frequency = self.acoustic_properties["frequency_range"][FrequencyProfile.UNDERWATER_PROPAGATION.name]["optimal"]
            
            # Add results to base result
            base_result["optimal_frequency"] = optimal_frequency
            base_result["optimal_mode"] = optimal_mode
            base_result["optimal_modulation"] = optimal_modulation
            base_result["effectiveness_estimate"] = effectiveness_estimate
            base_result["power_recommendation"] = min(1.0, self.acoustic_properties["power_level"] * 1.2)
            
        return base_result
    
    def calculate_effective_range(self, frequency: float, power_level: float, environment: str = "air") -> float:
        """
        Calculate effective range for given parameters.
        
        Args:
            frequency: Frequency in Hz
            power_level: Power level (0.0-1.0)
            environment: Environment type
            
        Returns:
            Effective range in meters
        """
        base_range = self.acoustic_properties["max_range"] * power_level
        
        # Adjust for frequency (higher frequencies attenuate faster)
        if frequency > 10000:  # 10 kHz
            frequency_factor = 0.7
        elif frequency > 1000:  # 1 kHz
            frequency_factor = 0.85
        else:
            frequency_factor = 1.0
            
        # Adjust for environment
        if environment == "underwater":
            environment_factor = 1.5  # Better propagation underwater at optimal frequencies
        elif environment == "urban":
            environment_factor = 0.6  # Buildings absorb and reflect
        else:  # air
            environment_factor = 1.0
            
        return base_range * frequency_factor * environment_factor
    
    def reload(self) -> bool:
        """
        Reload the acoustic disruptor to full capacity.
        
        Returns:
            Success status
        """
        self.status["remaining_capacity"] = self.specs.capacity
        return True
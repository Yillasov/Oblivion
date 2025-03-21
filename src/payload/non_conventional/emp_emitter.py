"""
Electromagnetic Pulse (EMP) Emitter implementation for UCAV platforms.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
from enum import Enum, auto

from src.payload.non_conventional.countermeasures import AdaptiveCountermeasure, CountermeasureSpecs
from src.payload.types import CountermeasureType, EMPStrength


class EMPMode(Enum):
    """EMP emission modes."""
    SINGLE_PULSE = auto()  # Single high-energy pulse
    MULTI_PULSE = auto()   # Multiple sequential pulses
    DIRECTIONAL = auto()   # Focused directional emission
    WIDE_AREA = auto()     # Broad area coverage
    SUSTAINED = auto()     # Continuous lower-power emission
    ADAPTIVE = auto()      # Adaptive emission based on threat


class EMPEmitter(AdaptiveCountermeasure):
    """
    Advanced electromagnetic pulse emitter that can disrupt
    or disable electronic systems in the target area.
    """
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "EMP-100":
            specs = CountermeasureSpecs(
                weight=45.0,
                volume={"length": 0.5, "width": 0.4, "height": 0.3},
                power_requirements=250.0,
                mounting_points=["fuselage", "internal_bay"],
                countermeasure_type=CountermeasureType.ELECTROMAGNETIC_PULSE,  # Changed from ELECTRONIC to ELECTROMAGNETIC_PULSE
                response_time=0.005,
                effectiveness_rating=0.7,
                capacity=10,
                coverage_angle=120.0,
                energy_consumption=200.0,
                thermal_signature=0.6,
                stealth_impact=0.5,
                cooldown_time=30.0
            )
        elif model == "EMP-300":
            specs = CountermeasureSpecs(
                weight=65.0,
                volume={"length": 0.6, "width": 0.5, "height": 0.4},
                power_requirements=400.0,
                mounting_points=["internal_bay"],
                countermeasure_type=CountermeasureType.ELECTROMAGNETIC_PULSE,  # Changed from ELECTRONIC to ELECTROMAGNETIC_PULSE
                response_time=0.003,
                effectiveness_rating=0.85,
                capacity=15,
                coverage_angle=180.0,
                energy_consumption=350.0,
                thermal_signature=0.7,
                stealth_impact=0.6,
                cooldown_time=20.0
            )
        else:
            raise ValueError(f"Unknown EMP emitter model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        
        # EMP emitter properties
        self.emp_properties = {
            "mode": EMPMode.SINGLE_PULSE,
            "strength": EMPStrength.MEDIUM,
            "charge_level": 0.0,
            "max_range": 300.0 if model == "EMP-300" else 150.0,  # meters
            "pulse_count": 0,
            "max_pulses": 3 if model == "EMP-300" else 1,
            "last_pulse_time": 0.0,
            "charging": False,
            "frequency_range": self._initialize_frequency_range(),
            "directional_angle": 30.0,  # degrees
            "target_coordinates": None,
            "effectiveness_by_target": {}
        }
        
        # Status tracking
        self.status.update({
            "remaining_charges": specs.capacity,
            "cooldown_remaining": 0.0
        })
    
    def _initialize_frequency_range(self) -> Dict[str, Tuple[float, float]]:
        """Initialize frequency ranges for different target types."""
        ranges = {}
        
        # Frequency ranges in MHz
        ranges["radar"] = (1000.0, 10000.0)  # 1-10 GHz
        ranges["communications"] = (30.0, 3000.0)  # 30 MHz - 3 GHz
        ranges["navigation"] = (100.0, 1500.0)  # 100 MHz - 1.5 GHz
        ranges["control_systems"] = (10.0, 500.0)  # 10-500 MHz
        ranges["power_systems"] = (0.1, 100.0)  # 100 kHz - 100 MHz
        
        return ranges
    
    def set_emp_mode(self, mode: EMPMode) -> bool:
        """
        Set the EMP emission mode.
        
        Args:
            mode: EMP emission mode
            
        Returns:
            Success status
        """
        if not isinstance(mode, EMPMode):
            return False
            
        self.emp_properties["mode"] = mode
        return True
    
    def set_emp_strength(self, strength: EMPStrength) -> bool:
        """
        Set the EMP emission strength.
        
        Args:
            strength: EMP emission strength
            
        Returns:
            Success status
        """
        if not isinstance(strength, EMPStrength):
            return False
            
        self.emp_properties["strength"] = strength
        return True
    
    def charge_capacitors(self) -> bool:
        """
        Begin charging the EMP capacitors.
        
        Returns:
            Success status
        """
        if self.status["cooldown_remaining"] > 0:
            return False
            
        if self.emp_properties["charging"]:
            return True  # Already charging
            
        if self.status["remaining_charges"] <= 0:
            return False  # No charges left
            
        self.emp_properties["charging"] = True
        self.emp_properties["charge_level"] = 0.0
        return True
    
    def set_target_coordinates(self, coordinates: Tuple[float, float, float]) -> bool:
        """
        Set target coordinates for directional EMP.
        
        Args:
            coordinates: (x, y, z) coordinates
            
        Returns:
            Success status
        """
        if not coordinates or len(coordinates) != 3:
            return False
            
        self.emp_properties["target_coordinates"] = coordinates
        return True
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy EMP against a target threat.
        
        Args:
            target_data: Data about the target threat
            
        Returns:
            Success status
        """
        # First check if base deployment is successful
        if not super().deploy(target_data):
            return False
        
        # Check if capacitors are charged
        if self.emp_properties["charge_level"] < 0.95:
            return False
            
        # Check if we have charges remaining
        if self.status["remaining_charges"] <= 0:
            return False
            
        # Use neuromorphic processing to optimize EMP deployment
        emp_result = self.process_data({
            "threat": target_data,
            "computation": "emp_optimization",
            "current_mode": self.emp_properties["mode"],
            "current_strength": self.emp_properties["strength"]
        })
        
        # Apply optimized parameters if available
        if "optimal_mode" in emp_result:
            self.emp_properties["mode"] = emp_result["optimal_mode"]
            
        if "optimal_strength" in emp_result:
            self.emp_properties["strength"] = emp_result["optimal_strength"]
            
        # Execute EMP pulse based on mode
        success = self._emit_emp_pulse(target_data)
        
        if success:
            # Update status
            self.status["remaining_charges"] -= 1
            self.emp_properties["pulse_count"] += 1
            self.emp_properties["last_pulse_time"] = time.time()
            self.emp_properties["charge_level"] = 0.0
            self.emp_properties["charging"] = False
            
            # Set cooldown
            self.status["cooldown_remaining"] = self.specs.cooldown_time
            
            # Calculate effectiveness against target
            target_id = target_data.get("id", "unknown")
            effectiveness = self._calculate_emp_effectiveness(target_data)
            self.emp_properties["effectiveness_by_target"][target_id] = effectiveness
        
        return success
    
    def _emit_emp_pulse(self, target_data: Dict[str, Any]) -> bool:
        """
        Emit an EMP pulse based on current mode and strength.
        
        Args:
            target_data: Target data
            
        Returns:
            Success status
        """
        mode = self.emp_properties["mode"]
        strength = self.emp_properties["strength"]
        
        # Get target position
        target_pos = target_data.get("position", None)
        if target_pos is None and mode == EMPMode.DIRECTIONAL:
            return False
            
        # Check if target is in range
        if target_pos:
            host_pos = target_data.get("host_position", [0, 0, 0])
            distance = np.linalg.norm(np.array(target_pos) - np.array(host_pos))
            
            if distance > self.emp_properties["max_range"]:
                return False  # Target out of range
        
        # Execute based on mode
        if mode == EMPMode.SINGLE_PULSE:
            # Single high-energy pulse
            return True
            
        elif mode == EMPMode.MULTI_PULSE:
            # Multiple sequential pulses
            # In real implementation, this would schedule multiple pulses
            return True
            
        elif mode == EMPMode.DIRECTIONAL:
            # Focused directional emission
            if self.emp_properties["target_coordinates"] is None:
                self.emp_properties["target_coordinates"] = target_pos
            return True
            
        elif mode == EMPMode.WIDE_AREA:
            # Broad area coverage
            return True
            
        elif mode == EMPMode.SUSTAINED:
            # Continuous lower-power emission
            # In real implementation, this would start a continuous emission
            return True
            
        elif mode == EMPMode.ADAPTIVE:
            # Adaptive emission based on threat
            target_type = target_data.get("type", "unknown")
            
            if target_type == "radar":
                # Use directional mode for radar
                self.emp_properties["mode"] = EMPMode.DIRECTIONAL
            elif target_type == "swarm":
                # Use wide area for swarms
                self.emp_properties["mode"] = EMPMode.WIDE_AREA
            elif target_type == "communications":
                # Use sustained for communications
                self.emp_properties["mode"] = EMPMode.SUSTAINED
            
            return True
            
        return False
    
    def _calculate_emp_effectiveness(self, target_data: Dict[str, Any]) -> float:
        """
        Calculate EMP effectiveness against a specific target.
        
        Args:
            target_data: Target data
            
        Returns:
            Effectiveness value (0.0-1.0)
        """
        # Base effectiveness from specs
        base_effectiveness = self.specs.effectiveness_rating
        
        # Adjust based on strength
        strength_factor = 0.5
        if self.emp_properties["strength"] == EMPStrength.LOW:
            strength_factor = 0.3
        elif self.emp_properties["strength"] == EMPStrength.MEDIUM:
            strength_factor = 0.6
        elif self.emp_properties["strength"] == EMPStrength.HIGH:
            strength_factor = 0.9
        elif self.emp_properties["strength"] == EMPStrength.DIRECTIONAL:
            strength_factor = 0.8
        elif self.emp_properties["strength"] == EMPStrength.SUSTAINED:
            strength_factor = 0.5
        elif self.emp_properties["strength"] == EMPStrength.PULSED:
            strength_factor = 0.7
        elif self.emp_properties["strength"] == EMPStrength.ADAPTIVE:
            strength_factor = 0.75
        
        # Adjust based on distance
        distance_factor = 1.0
        target_pos = target_data.get("position", None)
        if target_pos:
            host_pos = target_data.get("host_position", [0, 0, 0])
            distance = np.linalg.norm(np.array(target_pos) - np.array(host_pos))
            max_range = self.emp_properties["max_range"]
            
            if distance > max_range:
                distance_factor = 0.0
            else:
                distance_factor = 1.0 - (distance / max_range) ** 2
        
        # Adjust based on target type
        target_type = target_data.get("type", "unknown")
        target_factor = 0.5
        
        if target_type == "radar":
            target_factor = 0.8
        elif target_type == "communications":
            target_factor = 0.9
        elif target_type == "navigation":
            target_factor = 0.7
        elif target_type == "drone":
            target_factor = 0.6
        elif target_type == "vehicle":
            target_factor = 0.5
        elif target_type == "missile":
            target_factor = 0.4
        
        # Calculate final effectiveness
        effectiveness = base_effectiveness * strength_factor * distance_factor * target_factor
        return min(1.0, effectiveness)
    
    def update(self, dt: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update EMP emitter state over time.
        
        Args:
            dt: Time step in seconds
            environment_data: Environmental data
            
        Returns:
            Updated status
        """
        # Update cooldown timer
        if self.status["cooldown_remaining"] > 0:
            self.status["cooldown_remaining"] = max(0.0, self.status["cooldown_remaining"] - dt)
        
        # Update charging
        if self.emp_properties["charging"] and self.emp_properties["charge_level"] < 1.0:
            # Charging rate depends on model
            charge_rate = 0.2 if self.model == "EMP-300" else 0.1  # per second
            self.emp_properties["charge_level"] = min(1.0, self.emp_properties["charge_level"] + charge_rate * dt)
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current EMP emitter status."""
        status = super().get_status()
        status.update({
            "emp_properties": self.emp_properties,
            "charge_level": self.emp_properties["charge_level"],
            "mode": self.emp_properties["mode"],
            "strength": self.emp_properties["strength"],
            "remaining_charges": self.status["remaining_charges"],
            "cooldown_remaining": self.status["cooldown_remaining"]
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
        
        if computation_type == "emp_optimization":
            # Neuromorphic EMP optimization
            threat = input_data.get("threat", {})
            threat_type = threat.get("type", "unknown")
            
            # Determine optimal EMP mode based on threat
            optimal_mode = EMPMode.SINGLE_PULSE
            if threat_type == "radar":
                optimal_mode = EMPMode.DIRECTIONAL
            elif threat_type == "swarm":
                optimal_mode = EMPMode.WIDE_AREA
            elif threat_type == "communications":
                optimal_mode = EMPMode.SUSTAINED
            elif threat_type == "missile":
                optimal_mode = EMPMode.MULTI_PULSE
            
            # Determine optimal strength based on threat
            optimal_strength = EMPStrength.MEDIUM
            if threat_type == "radar" or threat_type == "missile":
                optimal_strength = EMPStrength.HIGH
            elif threat_type == "communications":
                optimal_strength = EMPStrength.SUSTAINED
            elif threat_type == "drone":
                optimal_strength = EMPStrength.DIRECTIONAL
            
            # Calculate effectiveness against different target types
            target_effectiveness = {}
            for target_type, freq_range in self.emp_properties["frequency_range"].items():
                effectiveness = 0.7  # Base effectiveness
                
                # Adjust based on EMP strength
                if self.emp_properties["strength"] == EMPStrength.HIGH:
                    effectiveness *= 1.3
                elif self.emp_properties["strength"] == EMPStrength.LOW:
                    effectiveness *= 0.7
                
                # Adjust based on target type match
                if target_type == threat_type:
                    effectiveness *= 1.2
                
                target_effectiveness[target_type] = min(1.0, effectiveness)
            
            # Calculate overall effectiveness
            overall_effectiveness = sum(target_effectiveness.values()) / len(target_effectiveness)
            
            # Add results to base result
            base_result["optimal_mode"] = optimal_mode
            base_result["optimal_strength"] = optimal_strength
            base_result["target_effectiveness"] = target_effectiveness
            base_result["overall_effectiveness"] = overall_effectiveness
            
        return base_result
    
    def get_charge_status(self) -> float:
        """
        Get current capacitor charge level.
        
        Returns:
            Charge level (0.0-1.0)
        """
        return self.emp_properties["charge_level"]
    
    def set_directional_angle(self, angle: float) -> bool:
        """
        Set the directional angle for focused EMP.
        
        Args:
            angle: Directional angle in degrees
            
        Returns:
            Success status
        """
        if 5.0 <= angle <= 180.0:
            self.emp_properties["directional_angle"] = angle
            return True
        return False
    
    def abort_charging(self) -> bool:
        """
        Abort the capacitor charging process.
        
        Returns:
            Success status
        """
        if not self.emp_properties["charging"]:
            return False
            
        self.emp_properties["charging"] = False
        self.emp_properties["charge_level"] = 0.0
        return True
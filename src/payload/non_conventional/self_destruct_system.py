"""
Self-Destructing Drone System implementation for UCAV platforms.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
from enum import Enum, auto

from src.payload.non_conventional.countermeasures import AdaptiveCountermeasure, CountermeasureSpecs
from src.payload.types import CountermeasureType


class DestructionMode(Enum):
    """Self-destruction modes."""
    COMPLETE = auto()       # Complete destruction of the entire drone
    PARTIAL = auto()        # Partial destruction (key components only)
    ELECTRONIC = auto()     # Electronic systems destruction only
    STAGED = auto()         # Multi-stage destruction sequence
    DELAYED = auto()        # Timed destruction after delay
    PROXIMITY = auto()      # Destruction based on proximity to target


class DestructionMechanism(Enum):
    """Self-destruction mechanisms."""
    EXPLOSIVE = auto()      # Conventional explosives
    THERMITE = auto()       # Thermite-based destruction
    ACID = auto()           # Acid-based destruction of components
    MECHANICAL = auto()     # Mechanical shredding of components
    EMP = auto()            # Electromagnetic pulse for electronics
    HYBRID = auto()         # Combination of multiple mechanisms


class SelfDestructSystem(AdaptiveCountermeasure):
    """
    Advanced self-destruct system that can eliminate the drone
    to prevent capture or for offensive purposes.
    """
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "SDS-100":
            specs = CountermeasureSpecs(
                weight=15.0,
                volume={"length": 0.3, "width": 0.2, "height": 0.1},
                power_requirements=50.0,
                mounting_points=["central_fuselage"],
                countermeasure_type=CountermeasureType.SELF_DESTRUCTING_DRONE,  # Changed from SELF_DESTRUCT to SELF_DESTRUCTING_DRONE
                response_time=0.001,
                effectiveness_rating=0.95,
                capacity=1,
                coverage_angle=360.0,
                energy_consumption=100.0,
                thermal_signature=0.2,
                stealth_impact=0.1,
                cooldown_time=0.0  # No cooldown for obvious reasons
            )
        elif model == "SDS-300":
            specs = CountermeasureSpecs(
                weight=25.0,
                volume={"length": 0.4, "width": 0.3, "height": 0.15},
                power_requirements=80.0,
                mounting_points=["central_fuselage", "wing_roots"],
                countermeasure_type=CountermeasureType.SELF_DESTRUCTING_DRONE,  # Changed from SELF_DESTRUCT to SELF_DESTRUCTING_DRONE
                response_time=0.0005,
                effectiveness_rating=0.99,
                capacity=1,
                coverage_angle=360.0,
                energy_consumption=150.0,
                thermal_signature=0.3,
                stealth_impact=0.15,
                cooldown_time=0.0
            )
        else:
            raise ValueError(f"Unknown self-destruct system model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        
        # Self-destruct system properties
        self.destruct_properties = {
            "mode": DestructionMode.COMPLETE,
            "mechanism": DestructionMechanism.EXPLOSIVE if model == "SDS-100" else DestructionMechanism.HYBRID,
            "armed": False,
            "countdown_active": False,
            "countdown_time": 0.0,
            "authorization_required": True,
            "authorization_code": None,
            "failsafe_active": True,
            "tamper_detection": True,
            "destruction_radius": 10.0 if model == "SDS-100" else 20.0,  # meters
            "component_targets": self._initialize_component_targets(),
            "last_status_check": time.time()
        }
        
        # Status tracking
        self.status.update({
            "armed": False,
            "countdown_active": False,
            "countdown_remaining": 0.0,
            "system_integrity": 1.0
        })
    
    def _initialize_component_targets(self) -> Dict[str, float]:
        """Initialize component destruction priorities."""
        targets = {}
        
        # Component priorities (0.0-1.0)
        targets["flight_computer"] = 1.0
        targets["navigation_system"] = 0.9
        targets["communication_system"] = 0.95
        targets["sensor_suite"] = 0.8
        targets["payload_bay"] = 0.7
        targets["propulsion"] = 0.6
        targets["power_system"] = 0.85
        targets["memory_storage"] = 1.0
        
        return targets
    
    def set_destruction_mode(self, mode: DestructionMode) -> bool:
        """
        Set the self-destruction mode.
        
        Args:
            mode: Destruction mode
            
        Returns:
            Success status
        """
        if not isinstance(mode, DestructionMode):
            return False
            
        self.destruct_properties["mode"] = mode
        return True
    
    def set_destruction_mechanism(self, mechanism: DestructionMechanism) -> bool:
        """
        Set the self-destruction mechanism.
        
        Args:
            mechanism: Destruction mechanism
            
        Returns:
            Success status
        """
        if not isinstance(mechanism, DestructionMechanism):
            return False
            
        self.destruct_properties["mechanism"] = mechanism
        return True
    
    def arm_system(self, authorization_code: Optional[str] = None) -> bool:
        """
        Arm the self-destruct system.
        
        Args:
            authorization_code: Authorization code if required
            
        Returns:
            Success status
        """
        if self.destruct_properties["authorization_required"] and authorization_code != self.destruct_properties["authorization_code"]:
            return False
            
        self.destruct_properties["armed"] = True
        self.status["armed"] = True
        return True
    
    def disarm_system(self, authorization_code: Optional[str] = None) -> bool:
        """
        Disarm the self-destruct system.
        
        Args:
            authorization_code: Authorization code if required
            
        Returns:
            Success status
        """
        if self.destruct_properties["authorization_required"] and authorization_code != self.destruct_properties["authorization_code"]:
            return False
            
        self.destruct_properties["armed"] = False
        self.status["armed"] = False
        
        # Cancel any active countdown
        if self.destruct_properties["countdown_active"]:
            self.destruct_properties["countdown_active"] = False
            self.status["countdown_active"] = False
            self.status["countdown_remaining"] = 0.0
            
        return True
    
    def set_authorization_code(self, code: str) -> bool:
        """
        Set the authorization code for the self-destruct system.
        
        Args:
            code: Authorization code
            
        Returns:
            Success status
        """
        if not code or len(code) < 8:
            return False  # Code too short
            
        self.destruct_properties["authorization_code"] = code
        return True
    
    def start_countdown(self, seconds: float, authorization_code: Optional[str] = None) -> bool:
        """
        Start the self-destruct countdown.
        
        Args:
            seconds: Countdown time in seconds
            authorization_code: Authorization code if required
            
        Returns:
            Success status
        """
        if not self.destruct_properties["armed"]:
            return False
            
        if self.destruct_properties["authorization_required"] and authorization_code != self.destruct_properties["authorization_code"]:
            return False
            
        if seconds <= 0:
            return False
            
        self.destruct_properties["countdown_active"] = True
        self.destruct_properties["countdown_time"] = seconds
        self.status["countdown_active"] = True
        self.status["countdown_remaining"] = seconds
        
        return True
    
    def abort_countdown(self, authorization_code: Optional[str] = None) -> bool:
        """
        Abort the self-destruct countdown.
        
        Args:
            authorization_code: Authorization code if required
            
        Returns:
            Success status
        """
        if not self.destruct_properties["countdown_active"]:
            return False
            
        if self.destruct_properties["authorization_required"] and authorization_code != self.destruct_properties["authorization_code"]:
            return False
            
        self.destruct_properties["countdown_active"] = False
        self.status["countdown_active"] = False
        self.status["countdown_remaining"] = 0.0
        
        return True
    
    def execute_self_destruct(self, authorization_code: Optional[str] = None) -> bool:
        """
        Execute immediate self-destruction.
        
        Args:
            authorization_code: Authorization code if required
            
        Returns:
            Success status
        """
        if not self.destruct_properties["armed"]:
            return False
            
        if self.destruct_properties["authorization_required"] and authorization_code != self.destruct_properties["authorization_code"]:
            return False
            
        # In a real implementation, this would trigger the actual destruction
        # Here we just simulate the process
        
        # Use neuromorphic processing to optimize destruction sequence
        destruction_plan = self.process_data({
            "computation": "destruction_optimization",
            "mode": self.destruct_properties["mode"],
            "mechanism": self.destruct_properties["mechanism"],
            "component_targets": self.destruct_properties["component_targets"]
        })
        
        # Log the destruction event
        destruction_sequence = destruction_plan.get("destruction_sequence", [])
        destruction_effectiveness = destruction_plan.get("effectiveness", 0.0)
        
        # In a real implementation, this would be the point of no return
        # For simulation purposes, we just update the status
        self.status["system_integrity"] = 0.0
        
        return True
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy self-destruct system.
        
        Args:
            target_data: Data about the target or situation
            
        Returns:
            Success status
        """
        # First check if base deployment is successful
        if not super().deploy(target_data):
            return False
        
        # Check if system is armed
        if not self.destruct_properties["armed"]:
            return False
            
        # Execute self-destruct
        return self.execute_self_destruct()
    
    def update(self, dt: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update self-destruct system state over time.
        
        Args:
            dt: Time step in seconds
            environment_data: Environmental data
            
        Returns:
            Updated status
        """
        # Update countdown if active
        if self.status["countdown_active"] and self.status["countdown_remaining"] > 0:
            self.status["countdown_remaining"] = max(0.0, self.status["countdown_remaining"] - dt)
            
            # If countdown reaches zero, execute self-destruct
            if self.status["countdown_remaining"] <= 0:
                self.execute_self_destruct()
        
        # Perform periodic system integrity check
        current_time = time.time()
        if current_time - self.destruct_properties["last_status_check"] > 60.0:  # Check every minute
            self.destruct_properties["last_status_check"] = current_time
            
            # Check for tampering if enabled
            if self.destruct_properties["tamper_detection"] and environment_data:
                tamper_detected = environment_data.get("tamper_detected", False)
                if tamper_detected and self.destruct_properties["failsafe_active"]:
                    # Auto-execute self-destruct if tampering detected
                    self.arm_system()
                    self.execute_self_destruct()
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current self-destruct system status."""
        status = super().get_status()
        status.update({
            "destruct_properties": self.destruct_properties,
            "armed": self.status["armed"],
            "countdown_active": self.status["countdown_active"],
            "countdown_remaining": self.status["countdown_remaining"],
            "system_integrity": self.status["system_integrity"]
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
        
        if computation_type == "destruction_optimization":
            # Neuromorphic destruction sequence optimization
            mode = input_data.get("mode", DestructionMode.COMPLETE)
            mechanism = input_data.get("mechanism", DestructionMechanism.EXPLOSIVE)
            component_targets = input_data.get("component_targets", {})
            
            # Determine optimal destruction sequence
            destruction_sequence = []
            
            if mode == DestructionMode.COMPLETE:
                # Complete destruction - all components at once
                destruction_sequence = ["all_components"]
                
            elif mode == DestructionMode.PARTIAL:
                # Partial destruction - prioritize critical components
                sorted_components = sorted(component_targets.items(), key=lambda x: x[1], reverse=True)
                destruction_sequence = [comp for comp, _ in sorted_components if _ > 0.7]
                
            elif mode == DestructionMode.ELECTRONIC:
                # Electronic systems only
                electronic_components = ["flight_computer", "navigation_system", "communication_system", 
                                        "sensor_suite", "memory_storage"]
                destruction_sequence = electronic_components
                
            elif mode == DestructionMode.STAGED:
                # Multi-stage destruction
                # Stage 1: Memory and communications
                # Stage 2: Navigation and sensors
                # Stage 3: Remaining systems
                stage1 = ["memory_storage", "communication_system"]
                stage2 = ["navigation_system", "sensor_suite", "flight_computer"]
                stage3 = ["power_system", "propulsion", "payload_bay"]
                destruction_sequence = [stage1, stage2, stage3]
                
            elif mode == DestructionMode.DELAYED:
                # Same as complete but with delay
                destruction_sequence = ["all_components"]
                
            elif mode == DestructionMode.PROXIMITY:
                # Proximity-based destruction
                destruction_sequence = ["all_components"]
            
            # Calculate effectiveness based on mechanism
            effectiveness = 0.7  # Base effectiveness
            
            if mechanism == DestructionMechanism.EXPLOSIVE:
                effectiveness = 0.95
            elif mechanism == DestructionMechanism.THERMITE:
                effectiveness = 0.9
            elif mechanism == DestructionMechanism.ACID:
                effectiveness = 0.8
            elif mechanism == DestructionMechanism.MECHANICAL:
                effectiveness = 0.75
            elif mechanism == DestructionMechanism.EMP:
                effectiveness = 0.6  # Only affects electronics
            elif mechanism == DestructionMechanism.HYBRID:
                effectiveness = 0.98  # Most effective
            
            # Add results to base result
            base_result["destruction_sequence"] = destruction_sequence
            base_result["effectiveness"] = effectiveness
            base_result["estimated_time"] = 0.1 if mode != DestructionMode.STAGED else 1.5
            base_result["collateral_damage_radius"] = self.destruct_properties["destruction_radius"]
            
        return base_result
    
    def toggle_failsafe(self, enabled: bool, authorization_code: Optional[str] = None) -> bool:
        """
        Enable or disable the failsafe system.
        
        Args:
            enabled: Whether failsafe should be enabled
            authorization_code: Authorization code if required
            
        Returns:
            Success status
        """
        if self.destruct_properties["authorization_required"] and authorization_code != self.destruct_properties["authorization_code"]:
            return False
            
        self.destruct_properties["failsafe_active"] = enabled
        return True
    
    def toggle_tamper_detection(self, enabled: bool, authorization_code: Optional[str] = None) -> bool:
        """
        Enable or disable tamper detection.
        
        Args:
            enabled: Whether tamper detection should be enabled
            authorization_code: Authorization code if required
            
        Returns:
            Success status
        """
        if self.destruct_properties["authorization_required"] and authorization_code != self.destruct_properties["authorization_code"]:
            return False
            
        self.destruct_properties["tamper_detection"] = enabled
        return True
    
    def set_destruction_radius(self, radius: float, authorization_code: Optional[str] = None) -> bool:
        """
        Set the destruction radius for explosive mechanisms.
        
        Args:
            radius: Destruction radius in meters
            authorization_code: Authorization code if required
            
        Returns:
            Success status
        """
        if self.destruct_properties["authorization_required"] and authorization_code != self.destruct_properties["authorization_code"]:
            return False
            
        if radius <= 0:
            return False
            
        # Limit maximum radius based on model
        max_radius = 30.0 if self.model == "SDS-300" else 15.0
        self.destruct_properties["destruction_radius"] = min(radius, max_radius)
        return True
    
    def run_diagnostic(self) -> Dict[str, Any]:
        """
        Run a diagnostic check on the self-destruct system.
        
        Returns:
            Diagnostic results
        """
        diagnostic_results = {
            "system_integrity": self.status["system_integrity"],
            "armed_status": self.destruct_properties["armed"],
            "countdown_status": self.destruct_properties["countdown_active"],
            "failsafe_status": self.destruct_properties["failsafe_active"],
            "tamper_detection_status": self.destruct_properties["tamper_detection"],
            "mechanism_status": "operational",
            "trigger_status": "operational",
            "power_supply_status": "operational",
            "authorization_system_status": "operational"
        }
        
        # In a real implementation, this would perform actual hardware checks
        
        return diagnostic_results
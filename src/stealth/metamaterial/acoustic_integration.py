"""
Acoustic signature integration for mission-adaptive cooling systems.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from enum import Enum

from src.stealth.metamaterial.mission_adaptive_cooling import MissionAdaptiveCooling, StealthPriority
from src.stealth.acoustic.acoustic_reduction import AcousticReductionSystem, AcousticParameters
from src.simulation.stealth.acoustic_simulator import AcousticSignatureSimulator, FrequencyRange


class AcousticCoolingMode(Enum):
    """Acoustic-thermal integration modes."""
    INDEPENDENT = "independent"  # Systems operate independently
    COORDINATED = "coordinated"  # Systems coordinate but prioritize their primary function
    INTEGRATED = "integrated"    # Fully integrated operation with shared resources
    ACOUSTIC_PRIORITY = "acoustic_priority"  # Prioritize acoustic signature reduction
    THERMAL_PRIORITY = "thermal_priority"    # Prioritize thermal signature reduction


class AcousticThermalIntegrator:
    """
    Integrates acoustic signature reduction with thermal management systems.
    Coordinates operations to optimize overall stealth performance.
    """
    
    def __init__(self, 
                cooling_system: MissionAdaptiveCooling,
                acoustic_system: Optional[AcousticReductionSystem] = None,
                acoustic_simulator: Optional[AcousticSignatureSimulator] = None):
        """
        Initialize acoustic-thermal integration.
        
        Args:
            cooling_system: Mission-adaptive cooling system
            acoustic_system: Optional acoustic reduction system
            acoustic_simulator: Optional acoustic simulator for modeling
        """
        self.cooling_system = cooling_system
        self.acoustic_system = acoustic_system
        self.acoustic_simulator = acoustic_simulator
        self.integration_mode = AcousticCoolingMode.INDEPENDENT
        self.acoustic_model_cache = {}
        self.last_platform_state = {}
        
    def set_integration_mode(self, mode: AcousticCoolingMode) -> Dict[str, Any]:
        """
        Set the integration mode between acoustic and thermal systems.
        
        Args:
            mode: Integration mode
            
        Returns:
            Status information
        """
        self.integration_mode = mode
        
        # Configure systems based on integration mode
        if self.acoustic_system:
            if mode == AcousticCoolingMode.ACOUSTIC_PRIORITY:
                # Configure acoustic system for maximum performance
                self.acoustic_system.status["active_damping_enabled"] = True
                self.acoustic_system.status["power_level"] = 0.9
                
                # Reduce thermal system power to compensate
                if self.cooling_system.cooling_system.active:
                    self.cooling_system.power_conservation = 0.7
                    
            elif mode == AcousticCoolingMode.THERMAL_PRIORITY:
                # Reduce acoustic system power to prioritize thermal
                self.acoustic_system.status["power_level"] = 0.6
                self.acoustic_system.status["active_damping_enabled"] = True
                
                # Maximize thermal system
                self.cooling_system.power_conservation = 0.3
                
            elif mode == AcousticCoolingMode.INTEGRATED:
                # Balanced approach
                self.acoustic_system.status["power_level"] = 0.75
                self.acoustic_system.status["active_damping_enabled"] = True
                self.cooling_system.power_conservation = 0.5
        
        return {
            "integration_mode": mode.value,
            "acoustic_system_active": self.acoustic_system.status["active"] if self.acoustic_system else False,
            "cooling_system_active": self.cooling_system.cooling_system.active,
            "power_distribution": {
                "acoustic": self.acoustic_system.status["power_level"] if self.acoustic_system else 0.0,
                "thermal": 1.0 - self.cooling_system.power_conservation if self.cooling_system else 0.0
            }
        }
    
    def update_platform_state(self, platform_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update platform state and recalculate stealth parameters.
        
        Args:
            platform_state: Current platform state
            
        Returns:
            Updated stealth metrics
        """
        self.last_platform_state = platform_state
        
        # Extract relevant data
        speed = platform_state.get("speed", 0.0)
        altitude = platform_state.get("altitude", 0.0)
        propulsion_state = platform_state.get("propulsion", {})
        environmental_conditions = platform_state.get("environment", {})
        
        # Calculate acoustic signature if simulator is available
        acoustic_signature = None
        if self.acoustic_simulator:
            acoustic_result = self.acoustic_simulator.calculate_signature(
                platform_state, environmental_conditions
            )
            acoustic_signature = acoustic_result.get("total_signature_db", 0.0)
            
            # Cache the acoustic model results
            self.acoustic_model_cache = {
                "signature": acoustic_signature,
                "components": acoustic_result.get("components", {}),
                "reduction_factor": acoustic_result.get("factors", {}).get("reduction", 1.0)
            }
        
        # Calculate thermal signature from cooling system
        thermal_signature = None
        if self.cooling_system.cooling_system.active:
            # Get exhaust profiles
            exhaust_profiles = self.cooling_system.cooling_system.exhaust_profiles
            if exhaust_profiles:
                # Use the first exhaust as representative
                first_exhaust_id = next(iter(exhaust_profiles))
                thermal_result = self.cooling_system.cooling_system.calculate_ir_reduction(first_exhaust_id)
                thermal_signature = thermal_result.get("mixed_temp", 0.0)
        
        # Coordinate systems if in integrated mode
        if self.integration_mode in [AcousticCoolingMode.INTEGRATED, AcousticCoolingMode.COORDINATED]:
            self._coordinate_systems(acoustic_signature, thermal_signature, speed, altitude)
        
        return {
            "acoustic_signature": acoustic_signature,
            "thermal_signature": thermal_signature,
            "integration_mode": self.integration_mode.value,
            "acoustic_model": self.acoustic_model_cache,
            "speed": speed,
            "altitude": altitude
        }
    
    def _coordinate_systems(self, 
                          acoustic_signature: Optional[float], 
                          thermal_signature: Optional[float],
                          speed: float,
                          altitude: float) -> None:
        """
        Coordinate acoustic and thermal systems for optimal stealth.
        
        Args:
            acoustic_signature: Current acoustic signature in dB
            thermal_signature: Current thermal signature in °C
            speed: Current speed in m/s
            altitude: Current altitude in m
        """
        if not self.acoustic_system or acoustic_signature is None:
            return
            
        # Adjust systems based on speed and altitude
        if speed > 100.0:  # High speed
            # At high speed, airframe noise dominates, so prioritize acoustic
            if self.integration_mode == AcousticCoolingMode.INTEGRATED:
                self.acoustic_system.status["power_level"] = min(0.9, self.acoustic_system.status["power_level"] + 0.1)
                self.cooling_system.power_conservation = min(0.7, self.cooling_system.power_conservation + 0.1)
        
        if altitude > 5000.0:  # High altitude
            # At high altitude, IR detection is more likely, prioritize thermal
            if self.integration_mode == AcousticCoolingMode.INTEGRATED:
                self.cooling_system.power_conservation = max(0.3, self.cooling_system.power_conservation - 0.1)
                self.acoustic_system.status["power_level"] = max(0.6, self.acoustic_system.status["power_level"] - 0.1)
    
    def predict_acoustic_signature(self, 
                                 future_state: Dict[str, Any],
                                 time_horizon: float = 60.0) -> Dict[str, Any]:
        """
        Predict future acoustic signature based on planned maneuvers.
        
        Args:
            future_state: Planned future platform state
            time_horizon: Time horizon in seconds
            
        Returns:
            Predicted acoustic signature
        """
        if not self.acoustic_simulator:
            return {"error": "Acoustic simulator not available"}
            
        # Extract planned parameters
        planned_speed = future_state.get("speed", 0.0)
        planned_altitude = future_state.get("altitude", 0.0)
        planned_propulsion = future_state.get("propulsion", {})
        
        # Create a copy of current state and update with planned values
        predicted_state = dict(self.last_platform_state)
        predicted_state["speed"] = planned_speed
        predicted_state["altitude"] = planned_altitude
        predicted_state["propulsion"] = planned_propulsion
        
        # Get environmental conditions
        environmental_conditions = predicted_state.get("environment", {})
        
        # Calculate predicted signature
        predicted_signature = self.acoustic_simulator.calculate_signature(
            predicted_state, environmental_conditions
        )
        
        # Calculate potential reduction with optimal settings
        optimal_reduction = self._calculate_optimal_acoustic_reduction(
            predicted_signature.get("total_signature_db", 0.0),
            planned_speed,
            planned_altitude
        )
        
        return {
            "current_signature_db": self.acoustic_model_cache.get("signature", 0.0),
            "predicted_signature_db": predicted_signature.get("total_signature_db", 0.0),
            "optimal_signature_db": predicted_signature.get("total_signature_db", 0.0) * optimal_reduction,
            "reduction_potential": (1.0 - optimal_reduction) * 100.0,  # percentage
            "time_horizon_seconds": time_horizon,
            "planned_speed": planned_speed,
            "planned_altitude": planned_altitude
        }
    
    def _calculate_optimal_acoustic_reduction(self, 
                                           base_signature: float,
                                           speed: float,
                                           altitude: float) -> float:
        """
        Calculate optimal acoustic reduction factor.
        
        Args:
            base_signature: Base acoustic signature in dB
            speed: Speed in m/s
            altitude: Altitude in m
            
        Returns:
            Optimal reduction factor (0.0-1.0, lower is better)
        """
        if not self.acoustic_system:
            return 1.0
            
        # Start with maximum possible reduction
        max_reduction = self.acoustic_system.acoustic_params.active_damping_capability
        
        # Adjust for speed (higher speed = less effective reduction)
        speed_factor = max(0.5, min(1.0, 1.0 - (speed / 300.0)))
        
        # Adjust for altitude (higher altitude = more effective reduction due to thinner air)
        altitude_factor = min(1.2, 1.0 + (altitude / 10000.0))
        
        # Calculate optimal reduction
        optimal_reduction = 1.0 - (max_reduction * speed_factor * altitude_factor)
        
        return max(0.1, optimal_reduction)  # Ensure at least 10% reduction
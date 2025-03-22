"""
Atmospheric Pressure Guidance System for UCAV platforms.

Provides navigation assistance based on atmospheric pressure patterns
and gradients for enhanced flight efficiency and stealth.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.navigation.error_handling import safe_navigation_operation
from src.navigation.common import NavigationDataProvider
from src.navigation.base import NavigationSystem, NavigationSpecs

# Configure logger
logger = logging.getLogger(__name__)


class PressurePatternType(Enum):
    """Types of atmospheric pressure patterns."""
    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    RIDGE = "ridge"
    TROUGH = "trough"
    FRONT = "front"
    GRADIENT = "gradient"


@dataclass
class AtmosphericGuidanceConfig:
    """Configuration for atmospheric pressure guidance."""
    min_pressure_delta: float = 50.0  # Pa
    max_altitude_change: float = 300.0  # meters
    update_rate: float = 1.0  # Hz
    use_pressure_gradients: bool = True
    enable_stealth_routing: bool = False
    prediction_horizon: float = 60.0  # seconds


class AtmosphericPressureGuidance(NavigationSystem, NavigationDataProvider):
    """
    Atmospheric Pressure Guidance system.
    
    Uses atmospheric pressure patterns to optimize flight paths
    for efficiency, stealth, and safety.
    """
    
    def __init__(self, config: AtmosphericGuidanceConfig = AtmosphericGuidanceConfig()):
        """
        Initialize atmospheric pressure guidance system.
        
        Args:
            config: System configuration
        """
        specs = NavigationSpecs(
            weight=0.3,  # kg
            volume={"value": 0.0002},  # m³
            power_requirements=0.5,  # Watts
            drift_rate=0.0,  # m/s
            initialization_time=2.0,  # seconds
            accuracy={"value": 50.0},  # m
            update_rate=config.update_rate  # Hz
        )
        super().__init__(specs)
        
        self.config = config
        
        # Current atmospheric data
        self.current_pressure = 101325.0  # Pa (standard sea level)
        self.pressure_history: List[Dict[str, float]] = []
        self.detected_patterns: List[Dict[str, Any]] = []
        
        # Guidance recommendations
        self.current_recommendation: Dict[str, Any] = {}
        
        logger.info("Initialized atmospheric pressure guidance system")
    
    @safe_navigation_operation
    def initialize(self) -> bool:
        """
        Initialize the atmospheric pressure guidance system.
        
        Returns:
            Success status
        """
        if self.initialized:
            return True
            
        logger.info("Initializing atmospheric pressure guidance system")
        
        # Initialize base system
        super().initialize()
        
        # Reset data
        self.pressure_history = []
        self.detected_patterns = []
        
        # Update status
        self.status["operational"] = True
        self.initialized = True
        
        logger.info("Atmospheric pressure guidance system initialized successfully")
        return True
    
    @safe_navigation_operation
    def update(self, delta_time: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update guidance based on atmospheric pressure data.
        
        Args:
            delta_time: Time since last update in seconds
            environment_data: Optional environment data including atmospheric conditions
            
        Returns:
            Current guidance recommendations
        """
        if not self.active:
            return {"error": "Atmospheric pressure guidance system not active"}
        
        # Extract atmospheric data
        if not environment_data or "atmospheric_conditions" not in environment_data:
            return {"error": "No atmospheric data available"}
            
        atmos_data = environment_data["atmospheric_conditions"]
        
        # Update current pressure
        if "pressure" in atmos_data:
            self.current_pressure = atmos_data["pressure"]
            
            # Store in history
            self._update_pressure_history(atmos_data)
        
        # Detect pressure patterns
        self._detect_pressure_patterns(environment_data)
        
        # Generate guidance recommendations
        self._generate_recommendations(environment_data)
        
        return self.current_recommendation
    
    def _update_pressure_history(self, atmos_data: Dict[str, Any]) -> None:
        """Update pressure history with new data."""
        entry = {
            "pressure": atmos_data["pressure"],
            "timestamp": atmos_data.get("timestamp", 0.0),
            "altitude": atmos_data.get("altitude", 0.0),
            "temperature": atmos_data.get("temperature", 288.15)
        }
        
        self.pressure_history.append(entry)
        
        # Limit history size
        if len(self.pressure_history) > 100:
            self.pressure_history.pop(0)
    
    def _detect_pressure_patterns(self, environment_data: Dict[str, Any]) -> None:
        """Detect atmospheric pressure patterns."""
        if len(self.pressure_history) < 3:
            return
            
        # Clear previous patterns
        self.detected_patterns = []
        
        # Get current and previous pressure readings
        current = self.pressure_history[-1]["pressure"]
        previous = self.pressure_history[-2]["pressure"]
        
        # Simple gradient detection
        pressure_delta = current - previous
        
        if abs(pressure_delta) > self.config.min_pressure_delta:
            pattern_type = (PressurePatternType.HIGH_PRESSURE 
                          if pressure_delta > 0 
                          else PressurePatternType.LOW_PRESSURE)
            
            self.detected_patterns.append({
                "type": pattern_type,
                "strength": abs(pressure_delta),
                "direction": np.sign(pressure_delta)
            })
        
        # Detect fronts (rapid pressure changes)
        if len(self.pressure_history) > 5:
            recent_pressures = [entry["pressure"] for entry in self.pressure_history[-5:]]
            pressure_variance = np.var(recent_pressures)
            
            if pressure_variance > self.config.min_pressure_delta * 2:
                self.detected_patterns.append({
                    "type": PressurePatternType.FRONT,
                    "strength": pressure_variance,
                    "direction": 0
                })
    
    def _generate_recommendations(self, environment_data: Dict[str, Any]) -> None:
        """Generate flight recommendations based on pressure patterns."""
        # Default recommendation
        recommendation = {
            "altitude_change": 0.0,
            "heading_change": 0.0,
            "speed_change": 0.0,
            "pattern_detected": False,
            "stealth_opportunity": False,
            "efficiency_gain": 0.0
        }
        
        # Process detected patterns
        for pattern in self.detected_patterns:
            recommendation["pattern_detected"] = True
            
            if pattern["type"] in [PressurePatternType.HIGH_PRESSURE, PressurePatternType.LOW_PRESSURE]:
                # Recommend altitude changes for pressure areas
                altitude_change = pattern["direction"] * min(
                    self.config.max_altitude_change * (pattern["strength"] / 1000.0),
                    self.config.max_altitude_change
                )
                recommendation["altitude_change"] = altitude_change
                
                # High pressure systems often provide efficiency gains
                if pattern["type"] == PressurePatternType.HIGH_PRESSURE:
                    recommendation["efficiency_gain"] = min(0.1, pattern["strength"] / 10000.0)
            
            elif pattern["type"] == PressurePatternType.FRONT:
                # Recommend avoiding fronts by changing heading
                recommendation["heading_change"] = 15.0  # degrees
                recommendation["speed_change"] = -0.05  # 5% reduction
        
        # Check for stealth opportunities (low pressure areas can mask acoustic signatures)
        if self.config.enable_stealth_routing and any(
            p["type"] == PressurePatternType.LOW_PRESSURE for p in self.detected_patterns
        ):
            recommendation["stealth_opportunity"] = True
        
        self.current_recommendation = recommendation
    
    def get_navigation_data(self) -> Dict[str, Any]:
        """
        Get navigation data from atmospheric pressure guidance.
        
        Returns:
            Navigation data including recommendations
        """
        if not self.active:
            return {"error": "Atmospheric pressure guidance system not active"}
            
        return {
            "current_pressure": self.current_pressure,
            "detected_patterns": [
                {"type": p["type"].value, "strength": p["strength"]} 
                for p in self.detected_patterns
            ],
            "recommendations": self.current_recommendation,
            "pressure_gradient": self._calculate_pressure_gradient()
        }
    
    def _calculate_pressure_gradient(self) -> Dict[str, float]:
        """Calculate current pressure gradient."""
        if len(self.pressure_history) < 2:
            return {"magnitude": 0.0, "direction": 0.0}
            
        # Simple time-based gradient
        current = self.pressure_history[-1]
        previous = self.pressure_history[-2]
        
        time_delta = current.get("timestamp", 0) - previous.get("timestamp", 0)
        if time_delta <= 0:
            time_delta = 0.1  # Avoid division by zero
            
        pressure_delta = current["pressure"] - previous["pressure"]
        gradient = pressure_delta / time_delta
        
        return {
            "magnitude": abs(gradient),
            "direction": np.sign(gradient)
        }
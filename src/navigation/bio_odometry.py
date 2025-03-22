"""
Bio-Inspired Odometry System for UCAV platforms.

Implements path integration mechanisms inspired by insect navigation,
particularly desert ants and honeybees.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.navigation.error_handling import safe_navigation_operation
from src.navigation.common import PositionProvider
from src.navigation.base import NavigationSystem, NavigationSpecs

# Configure logger
logger = logging.getLogger(__name__)


class BioInspirationSource(Enum):
    """Biological inspiration sources for odometry."""
    DESERT_ANT = "desert_ant"  # Cataglyphis fortis
    HONEYBEE = "honeybee"  # Apis mellifera
    MONARCH = "monarch"  # Monarch butterfly
    DUNG_BEETLE = "dung_beetle"  # Celestial compass
    CRICKET = "cricket"  # Gryllus bimaculatus


@dataclass
class BioOdometryConfig:
    """Configuration for bio-inspired odometry."""
    inspiration: BioInspirationSource = BioInspirationSource.DESERT_ANT
    step_integration_noise: float = 0.02  # Noise in step integration (0-1)
    direction_noise: float = 0.01  # Noise in direction sensing (radians)
    adaptation_rate: float = 0.3  # How quickly the system adapts to errors
    use_landmarks: bool = True  # Whether to use landmarks for correction
    max_path_memory: int = 1000  # Maximum path steps to remember


class BioInspiredOdometry(NavigationSystem, PositionProvider):
    """
    Bio-inspired odometry system.
    
    Implements path integration mechanisms inspired by insect navigation,
    particularly desert ants and honeybees.
    """
    
    def __init__(self, config: BioOdometryConfig = BioOdometryConfig()):
        """
        Initialize bio-inspired odometry system.
        
        Args:
            config: System configuration
        """
        specs = NavigationSpecs(
            weight=0.2,  # kg
            volume=0.0001,  # m³
            power_requirements={"voltage": 3.3, "current": 0.1},  # V, A
            drift_rate=0.05,  # m/s
            initialization_time=1.0,  # seconds
            accuracy=2.0,  # m
            update_rate=10.0  # Hz
        )
        super().__init__(specs)
        
        self.config = config
        
        # Current position estimate
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.heading = 0.0  # radians
        
        # Path integration memory
        self.path_memory: List[Dict[str, float]] = []
        
        # Landmark memory
        self.landmarks: Dict[str, Dict[str, float]] = {}
        
        # Confidence in current position
        self.position_confidence = 1.0
        
        # Initialize bio-inspired parameters based on inspiration source
        self._initialize_bio_parameters()
        
        logger.info(f"Initialized bio-inspired odometry with {config.inspiration.value} model")
    
    def _initialize_bio_parameters(self) -> None:
        """Initialize parameters based on biological inspiration source."""
        if self.config.inspiration == BioInspirationSource.DESERT_ANT:
            # Desert ants use step counting and celestial compass
            self.step_counter = 0
            self.step_length = 0.05  # meters per step
            self.celestial_compass_weight = 0.8
            self.path_integration_weight = 0.9
            self.landmark_weight = 0.3
            
        elif self.config.inspiration == BioInspirationSource.HONEYBEE:
            # Honeybees use optic flow and sun compass
            self.optic_flow_scale = 0.2
            self.sun_compass_weight = 0.7
            self.path_integration_weight = 0.8
            self.landmark_weight = 0.5
            
        elif self.config.inspiration == BioInspirationSource.MONARCH:
            # Monarchs use sun compass and magnetic sensing
            self.sun_compass_weight = 0.6
            self.magnetic_sense_weight = 0.5
            self.path_integration_weight = 0.7
            self.landmark_weight = 0.4
            
        else:
            # Default parameters
            self.path_integration_weight = 0.8
            self.landmark_weight = 0.4
    
    @safe_navigation_operation
    def initialize(self) -> bool:
        """
        Initialize the bio-inspired odometry system.
        
        Returns:
            Success status
        """
        if self.initialized:
            return True
            
        logger.info("Initializing bio-inspired odometry system")
        
        # Initialize base system
        super().initialize()
        
        # Reset position and path memory
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.path_memory = []
        
        # Update status
        self.status["operational"] = True
        self.initialized = True
        
        logger.info("Bio-inspired odometry system initialized successfully")
        return True
    
    @safe_navigation_operation
    def update(self, delta_time: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update position estimate using bio-inspired path integration.
        
        Args:
            delta_time: Time since last update in seconds
            environment_data: Optional environment data including motion and sensors
            
        Returns:
            Current position state
        """
        if not self.active:
            return {"error": "Bio-inspired odometry system not active"}
        
        # Extract motion data
        motion_data = self._extract_motion_data(environment_data)
        
        # Update position using path integration
        self._update_path_integration(motion_data, delta_time)
        
        # Update using landmarks if available
        if self.config.use_landmarks and environment_data and "landmarks" in environment_data:
            self._update_landmark_correction(environment_data["landmarks"])
        
        # Update position confidence (decays with distance traveled)
        distance_moved = np.sqrt(motion_data["velocity_x"]**2 + 
                               motion_data["velocity_y"]**2) * delta_time
        self.position_confidence *= max(0.9, 1.0 - 0.01 * distance_moved)
        
        # Store path step if significant movement
        if distance_moved > 0.1:
            self._store_path_step()
        
        return {
            "position": self.position,
            "heading": self.heading,
            "confidence": self.position_confidence,
            "path_memory_size": len(self.path_memory)
        }
    
    def _extract_motion_data(self, environment_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract motion data from environment data."""
        motion_data = {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_z": 0.0,
            "angular_velocity": 0.0,
            "heading": self.heading
        }
        
        if not environment_data:
            return motion_data
        
        # Extract velocity if available
        if "velocity" in environment_data:
            vel = environment_data["velocity"]
            if isinstance(vel, dict):
                motion_data["velocity_x"] = vel.get("x", 0.0)
                motion_data["velocity_y"] = vel.get("y", 0.0)
                motion_data["velocity_z"] = vel.get("z", 0.0)
            elif isinstance(vel, (list, tuple)) and len(vel) >= 3:
                motion_data["velocity_x"] = vel[0]
                motion_data["velocity_y"] = vel[1]
                motion_data["velocity_z"] = vel[2]
        
        # Extract heading if available
        if "heading" in environment_data:
            motion_data["heading"] = environment_data["heading"]
        
        # Extract angular velocity if available
        if "angular_velocity" in environment_data:
            motion_data["angular_velocity"] = environment_data["angular_velocity"]
        
        return motion_data
    
    def _update_path_integration(self, motion_data: Dict[str, float], delta_time: float) -> None:
        """Update position using path integration."""
        # Update heading
        self.heading = motion_data["heading"]
        
        # Add noise to simulate biological imperfection
        heading_noise = np.random.normal(0, self.config.direction_noise)
        step_noise = np.random.normal(1.0, self.config.step_integration_noise)
        
        # Calculate displacement in world coordinates
        dx = motion_data["velocity_x"] * delta_time * step_noise
        dy = motion_data["velocity_y"] * delta_time * step_noise
        dz = motion_data["velocity_z"] * delta_time * step_noise
        
        # Update position
        self.position["x"] += dx
        self.position["y"] += dy
        self.position["z"] += dz
    
    def _update_landmark_correction(self, landmarks: List[Dict[str, Any]]) -> None:
        """Update position based on recognized landmarks."""
        if not landmarks:
            return
            
        # Process each landmark
        for landmark in landmarks:
            landmark_id = landmark.get("id", "")
            if not landmark_id:
                continue
                
            # Get landmark position
            landmark_pos = landmark.get("position", {})
            if not landmark_pos:
                continue
                
            # If this is a new landmark, store it
            if landmark_id not in self.landmarks:
                self.landmarks[landmark_id] = {
                    "x": self.position["x"] + landmark.get("relative_x", 0.0),
                    "y": self.position["y"] + landmark.get("relative_y", 0.0),
                    "z": self.position["z"] + landmark.get("relative_z", 0.0),
                    "confidence": 0.5
                }
                continue
            
            # Get stored landmark
            stored_landmark = self.landmarks[landmark_id]
            
            # Calculate position correction based on landmark
            correction_x = (stored_landmark["x"] - landmark.get("relative_x", 0.0)) - self.position["x"]
            correction_y = (stored_landmark["y"] - landmark.get("relative_y", 0.0)) - self.position["y"]
            correction_z = (stored_landmark["z"] - landmark.get("relative_z", 0.0)) - self.position["z"]
            
            # Apply correction with weight based on landmark confidence
            weight = self.landmark_weight * stored_landmark["confidence"]
            self.position["x"] += correction_x * weight * self.config.adaptation_rate
            self.position["y"] += correction_y * weight * self.config.adaptation_rate
            self.position["z"] += correction_z * weight * self.config.adaptation_rate
            
            # Increase position confidence
            self.position_confidence = min(1.0, self.position_confidence + 0.1 * weight)
            
            # Update landmark confidence
            stored_landmark["confidence"] = min(1.0, stored_landmark["confidence"] + 0.05)
    
    def _store_path_step(self) -> None:
        """Store current position in path memory."""
        # Add current position to path memory
        self.path_memory.append({
            "x": self.position["x"],
            "y": self.position["y"],
            "z": self.position["z"],
            "heading": self.heading,
            "confidence": self.position_confidence
        })
        
        # Limit path memory size
        if len(self.path_memory) > self.config.max_path_memory:
            self.path_memory.pop(0)
    
    def get_position(self) -> Dict[str, float]:
        """
        Get current position using bio-inspired odometry.
        
        Returns:
            Position dictionary with x, y, z coordinates
        """
        if not self.active:
            return {"x": float('nan'), "y": float('nan'), "z": float('nan')}
            
        return self.position
    
    def get_path_home(self) -> List[Dict[str, float]]:
        """
        Get path back to origin (home vector) - inspired by insect homing.
        
        Returns:
            List of waypoints to return home
        """
        if not self.path_memory:
            return [self.position]
            
        # Simple version: direct vector to origin
        return [{
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        }]
    
    def get_full_path_home(self) -> List[Dict[str, float]]:
        """
        Get full path back to origin by reversing the path memory.
        
        Returns:
            List of waypoints to return home
        """
        if not self.path_memory:
            return [self.position]
            
        # Reverse path memory to get path home
        return list(reversed(self.path_memory))
"""
Bio-inspired Target Recognition System

Implements a simple target recognition system based on biological visual processing
principles, including center-surround detection and motion sensitivity.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import cv2

from src.core.utils.logging_framework import get_logger

logger = get_logger("bio_target_recognition")


@dataclass
class VisualTarget:
    """Target information."""
    
    center: Tuple[int, int]  # Center coordinates (x, y)
    size: float              # Target size in pixels
    confidence: float        # Detection confidence
    velocity: Optional[Tuple[float, float]] = None  # Motion vector


class BioTargetRecognition:
    """Bio-inspired target recognition implementation."""
    
    def __init__(self):
        """Initialize target recognition system."""
        # Center-surround detection parameters
        self.center_size = 5
        self.surround_size = 15
        self.threshold = 0.3
        
        # Motion detection parameters
        self.motion_history = []
        self.max_history = 5
        
        # Target tracking
        self.current_targets: List[VisualTarget] = []
        self.prev_frame = None
        
        logger.info("Initialized bio-inspired target recognition system")
    
    def _create_center_surround_kernel(self) -> np.ndarray:
        """Create center-surround receptive field kernel."""
        kernel = np.zeros((self.surround_size, self.surround_size))
        
        # Create center (excitatory)
        y_center = self.surround_size // 2
        x_center = self.surround_size // 2
        
        for y in range(self.surround_size):
            for x in range(self.surround_size):
                # Distance from center
                dist = np.sqrt((y - y_center)**2 + (x - x_center)**2)
                
                if dist < self.center_size:
                    kernel[y, x] = 1.0  # Center (excitatory)
                elif dist < self.surround_size/2:
                    kernel[y, x] = -0.5  # Surround (inhibitory)
        
        # Normalize kernel
        kernel = kernel / np.abs(kernel).sum()
        return kernel
    
    def _detect_motion(self, frame: np.ndarray) -> np.ndarray:
        """Detect motion using temporal difference."""
        if self.prev_frame is None:
            self.prev_frame = frame
            return np.zeros_like(frame)
        
        # Simple temporal difference
        motion = cv2.absdiff(frame, self.prev_frame)
        self.prev_frame = frame.copy()
        
        return motion
    
    def _find_target_candidates(self, response: np.ndarray) -> List[Tuple[int, int]]:
        """Find potential target locations from detector response."""
        candidates = []
        
        # Find local maxima
        response_pad = np.pad(response, ((1,1), (1,1)), mode='edge')
        for y in range(1, response_pad.shape[0]-1):
            for x in range(1, response_pad.shape[1]-1):
                # 3x3 neighborhood
                neighborhood = response_pad[y-1:y+2, x-1:x+2]
                if response_pad[y, x] == np.max(neighborhood) and \
                   response_pad[y, x] > self.threshold:
                    candidates.append((x-1, y-1))
        
        return candidates
    
    def process_frame(self, frame: np.ndarray) -> List[VisualTarget]:
        """
        Process a frame to detect and track targets.
        
        Args:
            frame: Grayscale input frame (2D numpy array)
            
        Returns:
            List[VisualTarget]: Detected targets
        """
        # Convert to float and normalize
        frame_norm = frame.astype(float) / 255.0
        
        # Center-surround detection
        kernel = self._create_center_surround_kernel()
        response = cv2.filter2D(frame_norm, -1, kernel)
        
        # Motion detection
        motion = self._detect_motion(frame_norm)
        
        # Combine detection responses
        combined_response = response * (motion + 0.5)
        
        # Find target candidates
        candidates = self._find_target_candidates(combined_response)
        
        # Create target objects
        targets = []
        for x, y in candidates:
            # Estimate target size
            local_response = combined_response[
                max(0, y-5):min(frame.shape[0], y+6),
                max(0, x-5):min(frame.shape[1], x+6)
            ]
            size = np.sum(local_response > self.threshold/2)
            
            # Calculate confidence
            confidence = combined_response[y, x]
            
            # Create target
            target = VisualTarget(
                center=(x, y),
                size=float(size),
                confidence=float(confidence)
            )
            targets.append(target)
        
        # Update motion history
        self.motion_history.append(targets)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)
        
        # Update velocities for tracked targets
        if len(self.motion_history) >= 2:
            prev_targets = self.motion_history[-2]
            for target in targets:
                # Find closest previous target
                min_dist = float('inf')
                best_velocity = None
                
                for prev_target in prev_targets:
                    dx = target.center[0] - prev_target.center[0]
                    dy = target.center[1] - prev_target.center[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < min_dist and dist < 50:  # Max tracking distance
                        min_dist = dist
                        best_velocity = (float(dx), float(dy))
                
                target.velocity = best_velocity
        
        self.current_targets = targets
        logger.debug(f"Detected {len(targets)} targets")
        return targets
    
    def get_target_predictions(self, timesteps: int = 1) -> List[Tuple[int, int]]:
        """
        Predict target positions for future timesteps.
        
        Args:
            timesteps: Number of timesteps to predict ahead
            
        Returns:
            List[Tuple[int, int]]: Predicted target positions
        """
        predictions = []
        
        for target in self.current_targets:
            if target.velocity is not None:
                # Simple linear prediction
                pred_x = target.center[0] + target.velocity[0] * timesteps
                pred_y = target.center[1] + target.velocity[1] * timesteps
                predictions.append((int(pred_x), int(pred_y)))
        
        return predictions
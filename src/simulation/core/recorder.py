"""
Simulation Data Recording and Playback

Provides basic functionality for recording and replaying simulation states.
"""

import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("recorder")

@dataclass
class SimulationState:
    """Snapshot of simulation state at a specific time."""
    sim_time: float
    real_time: float
    frame: int
    state_data: Dict[str, Any]

class DataRecorder:
    """Records simulation data at specified intervals."""
    
    def __init__(self, save_interval: float = 0.1):
        self.save_interval = save_interval
        self.last_save_time = 0.0
        self.recorded_data = {}
        self.current_frame = 0
    
    def update(self, sim_time: float, state: Dict[str, Any]) -> None:
        """Record state if interval has passed."""
        if sim_time - self.last_save_time >= self.save_interval:
            self.recorded_data[self.current_frame] = SimulationState(
                sim_time=sim_time,
                real_time=time.time(),
                frame=self.current_frame,
                state_data=state.copy()
            )
            self.current_frame += 1
            self.last_save_time = sim_time
    
    def save_to_file(self, filename: str) -> None:
        """Save recorded data to JSON file."""
        with open(filename, 'w') as f:
            json.dump({
                'metadata': {
                    'save_interval': self.save_interval,
                    'total_frames': self.current_frame
                },
                'data': {k: v.__dict__ for k, v in self.recorded_data.items()}
            }, f, indent=2)
        logger.info(f"Saved {self.current_frame} frames to {filename}")

class PlaybackSystem:
    """Replays recorded simulation data."""
    
    def __init__(self):
        self.recorded_data = {}
        self.current_index = 0
        self.total_frames = 0
    
    def load_from_file(self, filename: str) -> None:
        """Load recorded data from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
            self.recorded_data = {
                int(k): SimulationState(**v)
                for k, v in data['data'].items()
            }
            self.total_frames = data['metadata']['total_frames']
        logger.info(f"Loaded {self.total_frames} frames from {filename}")
    
    def get_current_state(self) -> Optional[SimulationState]:
        """Get current playback state."""
        return self.recorded_data.get(self.current_index)
    
    def step(self, steps: int = 1) -> Optional[SimulationState]:
        """Step through recorded data."""
        self.current_index = max(0, min(self.current_index + steps, self.total_frames - 1))
        return self.get_current_state()
    
    def reset(self) -> None:
        """Reset playback to beginning."""
        self.current_index = 0
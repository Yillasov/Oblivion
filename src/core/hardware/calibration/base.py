#!/usr/bin/env python3
"""
Base class for hardware-specific calibration.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CalibrationResult:
    success: bool
    parameters: Dict[str, float]
    timestamp: datetime
    error_margin: float
    notes: Optional[str] = None

class HardwareCalibrator(ABC):
    
    
    def __init__(self):
        self.last_calibration: Optional[CalibrationResult] = None
        self.calibration_parameters: Dict[str, float] = {}
    
    @abstractmethod
    def perform_basic_calibration(self) -> CalibrationResult:
        """Perform basic hardware calibration."""
        pass
    
    @abstractmethod
    def calibrate_sensors(self) -> CalibrationResult:
        """Calibrate hardware sensors."""
        pass
    
    @abstractmethod
    def calibrate_timing(self) -> CalibrationResult:
        """Calibrate hardware timing parameters."""
        pass
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration status."""
        if not self.last_calibration:
            return {"calibrated": False}
            
        return {
            "calibrated": True,
            "last_calibration": self.last_calibration.timestamp.isoformat(),
            "parameters": self.last_calibration.parameters,
            "error_margin": self.last_calibration.error_margin
        }
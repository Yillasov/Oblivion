import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from src.core.hardware.hardware_exceptions import (
    HardwareError, InitializationError, 
    CommunicationError, ResourceError
)
from src.core.hardware.calibration.base import HardwareCalibrator

class BaseHardwareDriver(ABC):
    """Base class for neuromorphic hardware drivers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.initialized = False
        self.device_info = {}
        self.error_state = False
        self.last_error: Optional[Tuple[str, int]] = None
        self.calibrator: Optional[HardwareCalibrator] = None
    
    def safe_execute(self, operation: str) -> Dict[str, Any]:
        """Execute operation with error handling."""
        try:
            if not self.initialized and operation != "initialize":
                raise InitializationError("Hardware not initialized")
            
            if self.error_state:
                error_msg = self.last_error[0] if self.last_error else "Unknown error"
                raise HardwareError(f"Hardware in error state: {error_msg}")
            
            result = self._execute_operation(operation)
            return {"success": True, "data": result}
            
        except HardwareError as e:
            self.error_state = True
            self.last_error = (str(e), getattr(e, 'error_code', 0))
            return {
                "success": False,
                "error": str(e),
                "error_code": getattr(e, 'error_code', 0)
            }
    
    def _execute_operation(self, operation: str) -> Dict[str, Any]:
        """Execute hardware operation based on type."""
        operations = {
            "initialize": self.initialize,
            "shutdown": self.shutdown,
            "status": self.check_status,
            "capabilities": self.get_capabilities
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
            
        return operations[operation]()
    
    def reset_error_state(self) -> bool:
        """Attempt to reset error state."""
        try:
            if self.error_state:
                self.error_state = False
                self.last_error = None
                return self.initialize()
            return True
        except HardwareError:
            return False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize hardware connection."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Safely shutdown hardware."""
        pass
    
    @abstractmethod
    def check_status(self) -> Dict[str, Any]:
        """Check hardware status."""
        pass
    
    @abstractmethod
    def execute_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hardware-specific command."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get hardware capabilities."""
        pass
    
    def validate_config(self) -> bool:
        """Validate hardware configuration."""
        return True
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return self.device_info
    
    def calibrate(self) -> Dict[str, Any]:
        """Perform hardware calibration."""
        if not self.calibrator:
            return {"success": False, "error": "No calibrator available"}
            
        try:
            # Perform calibration sequence
            basic_result = self.calibrator.perform_basic_calibration()
            sensor_result = self.calibrator.calibrate_sensors()
            timing_result = self.calibrator.calibrate_timing()
            
            all_success = all([
                basic_result.success,
                sensor_result.success,
                timing_result.success
            ])
            
            return {
                "success": all_success,
                "basic_calibration": basic_result.parameters,
                "sensor_calibration": sensor_result.parameters,
                "timing_calibration": timing_result.parameters,
                "error_margin": max(
                    basic_result.error_margin,
                    sensor_result.error_margin,
                    timing_result.error_margin
                )
            }
            
        except Exception as e:
            self.error_state = True
            self.last_error = (str(e), -1)
            return {"success": False, "error": str(e)}
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration status."""
        if not self.calibrator:
            return {"calibrated": False}
        return self.calibrator.get_calibration_status()
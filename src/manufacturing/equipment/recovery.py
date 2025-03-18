from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EquipmentErrorCode(Enum):
    MOTOR_FAILURE = "motor_failure"
    SENSOR_ERROR = "sensor_error"
    CALIBRATION_ERROR = "calibration_error"
    POWER_FAILURE = "power_failure"
    COMMUNICATION_ERROR = "communication_error"
    OVERHEATING = "overheating"
    EMERGENCY_STOP = "emergency_stop"

class EquipmentRecoveryStrategy:
    """Base class for equipment recovery strategies."""
    
    def __init__(self, name: str, applicable_errors: List[EquipmentErrorCode]):
        self.name = name
        self.applicable_errors = applicable_errors
        self.last_attempt = None
    
    def can_handle(self, error_code: EquipmentErrorCode) -> bool:
        return error_code in self.applicable_errors
    
    def execute(self, equipment_id: str, error_details: Dict[str, Any]) -> bool:
        try:
            self.last_attempt = datetime.now()
            return self._execute_recovery(equipment_id, error_details)
        except Exception as e:
            logger.error(f"Recovery failed for {equipment_id}: {str(e)}")
            return False
    
    def _execute_recovery(self, equipment_id: str, error_details: Dict[str, Any]) -> bool:
        raise NotImplementedError()

class EmergencyShutdownRecovery(EquipmentRecoveryStrategy):
    """Emergency shutdown and safety check recovery."""
    
    def __init__(self):
        super().__init__(
            "Emergency Shutdown Recovery",
            [EquipmentErrorCode.EMERGENCY_STOP, EquipmentErrorCode.OVERHEATING]
        )
    
    def _execute_recovery(self, equipment_id: str, error_details: Dict[str, Any]) -> bool:
        logger.info(f"Executing emergency shutdown for {equipment_id}")
        # Implement emergency shutdown sequence
        return True

class PowerCycleRecovery(EquipmentRecoveryStrategy):
    """Power cycle and reinitialization recovery."""
    
    def __init__(self):
        super().__init__(
            "Power Cycle Recovery",
            [EquipmentErrorCode.POWER_FAILURE, EquipmentErrorCode.COMMUNICATION_ERROR]
        )
    
    def _execute_recovery(self, equipment_id: str, error_details: Dict[str, Any]) -> bool:
        logger.info(f"Executing power cycle for {equipment_id}")
        # Implement power cycle sequence
        return True

class CalibrationRecovery(EquipmentRecoveryStrategy):
    """Recalibration recovery strategy."""
    
    def __init__(self):
        super().__init__(
            "Calibration Recovery",
            [EquipmentErrorCode.CALIBRATION_ERROR, EquipmentErrorCode.SENSOR_ERROR]
        )
    
    def _execute_recovery(self, equipment_id: str, error_details: Dict[str, Any]) -> bool:
        logger.info(f"Executing recalibration for {equipment_id}")
        # Implement recalibration sequence
        return True

class EquipmentRecoveryManager:
    """Manager for equipment recovery strategies."""
    
    def __init__(self):
        self.strategies = []
        self.recovery_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_attempts = 3
        
        # Register default strategies
        self.register_strategy(EmergencyShutdownRecovery())
        self.register_strategy(PowerCycleRecovery())
        self.register_strategy(CalibrationRecovery())
    
    def register_strategy(self, strategy: EquipmentRecoveryStrategy) -> None:
        """Register a new recovery strategy."""
        self.strategies.append(strategy)
    
    def attempt_recovery(self, equipment_id: str, 
                        error_code: EquipmentErrorCode,
                        error_details: Dict[str, Any]) -> bool:
        """Attempt to recover from equipment error."""
        
        # Check attempt history
        if self._exceeded_max_attempts(equipment_id, error_code):
            logger.warning(f"Max recovery attempts exceeded for {equipment_id}")
            return False
        
        # Find appropriate strategy
        strategy = self._get_strategy(error_code)
        if not strategy:
            logger.error(f"No recovery strategy found for {error_code}")
            return False
        
        # Execute recovery
        result = strategy.execute(equipment_id, error_details)
        
        # Record attempt
        self._record_attempt(equipment_id, error_code, result)
        
        return result
    
    def _get_strategy(self, error_code: EquipmentErrorCode) -> Optional[EquipmentRecoveryStrategy]:
        """Get appropriate strategy for error code."""
        for strategy in self.strategies:
            if strategy.can_handle(error_code):
                return strategy
        return None
    
    def _exceeded_max_attempts(self, equipment_id: str, error_code: EquipmentErrorCode) -> bool:
        """Check if max recovery attempts exceeded."""
        recent_attempts = [
            attempt for attempt in self.recovery_history.get(equipment_id, [])
            if attempt["error_code"] == error_code
        ]
        return len(recent_attempts) >= self.max_attempts
    
    def _record_attempt(self, equipment_id: str, error_code: EquipmentErrorCode, success: bool):
        """Record recovery attempt."""
        if equipment_id not in self.recovery_history:
            self.recovery_history[equipment_id] = []
            
        self.recovery_history[equipment_id].append({
            "timestamp": datetime.now(),
            "error_code": error_code,
            "success": success
        })
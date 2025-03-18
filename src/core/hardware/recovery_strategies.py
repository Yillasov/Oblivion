"""
Automatic recovery strategies for common hardware failures.

Provides a set of recovery strategies that can be automatically applied
when hardware failures occur.
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import time
import threading

from src.core.utils.logging_framework import get_logger
from src.core.hardware.error_codes import HardwareErrorCode
from src.core.hardware.exceptions import NeuromorphicHardwareError
from src.core.hardware.hardware_registry import hardware_registry

logger = get_logger("hardware_recovery")


class RecoveryStrategy:
    """Base class for hardware recovery strategies."""
    
    def __init__(self, name: str, applicable_errors: List[HardwareErrorCode]):
        """
        Initialize recovery strategy.
        
        Args:
            name: Strategy name
            applicable_errors: List of error codes this strategy can handle
        """
        self.name = name
        self.applicable_errors = applicable_errors
        self.success_count = 0
        self.failure_count = 0
    
    def can_handle(self, error_code: HardwareErrorCode) -> bool:
        """Check if this strategy can handle the given error code."""
        return error_code in self.applicable_errors
    
    def apply(self, hardware_type: str, error_details: Dict[str, Any]) -> bool:
        """
        Apply recovery strategy.
        
        Args:
            hardware_type: Type of hardware
            error_details: Error details
            
        Returns:
            bool: Success status
        """
        try:
            result = self._execute_strategy(hardware_type, error_details)
            if result:
                self.success_count += 1
            else:
                self.failure_count += 1
            return result
        except Exception as e:
            logger.error(f"Recovery strategy '{self.name}' failed: {str(e)}")
            self.failure_count += 1
            return False
    
    def _execute_strategy(self, hardware_type: str, error_details: Dict[str, Any]) -> bool:
        """
        Execute the recovery strategy.
        
        Args:
            hardware_type: Type of hardware
            error_details: Error details
            
        Returns:
            bool: Success status
        """
        raise NotImplementedError("Recovery strategy must implement _execute_strategy")


class ResetStateStrategy(RecoveryStrategy):
    """Reset hardware state recovery strategy."""
    
    def __init__(self):
        """Initialize reset state strategy."""
        super().__init__(
            "Reset State",
            [
                HardwareErrorCode.SIMULATION_FAILED,
                HardwareErrorCode.LOIHI_CHIP_ERROR,
                HardwareErrorCode.SPINNAKER_CORE_FAILURE,
                HardwareErrorCode.SIMULATION_NUMERICAL_ERROR
            ]
        )
    
    def _execute_strategy(self, hardware_type: str, error_details: Dict[str, Any]) -> bool:
        """Execute reset state strategy."""
        logger.info(f"Applying reset state recovery for {hardware_type}")
        hardware = hardware_registry.get_hardware(hardware_type)
        if not hardware:
            return False
        
        return hardware.reset_state()


class ReinitializeStrategy(RecoveryStrategy):
    """Reinitialize hardware recovery strategy."""
    
    def __init__(self):
        """Initialize reinitialize strategy."""
        super().__init__(
            "Reinitialize Hardware",
            [
                HardwareErrorCode.INITIALIZATION_FAILED,
                HardwareErrorCode.COMMUNICATION_FAILED,
                HardwareErrorCode.HARDWARE_SWITCHING_FAILED
            ]
        )
    
    def _execute_strategy(self, hardware_type: str, error_details: Dict[str, Any]) -> bool:
        """Execute reinitialize strategy."""
        logger.info(f"Applying reinitialization recovery for {hardware_type}")
        hardware = hardware_registry.get_hardware(hardware_type)
        if not hardware:
            return False
        
        # First shutdown
        if hardware.initialized:
            hardware.shutdown()
        
        # Wait a moment before reinitializing
        time.sleep(1)
        
        # Reinitialize
        return hardware.initialize()


class FallbackHardwareStrategy(RecoveryStrategy):
    """Fallback to alternative hardware strategy."""
    
    def __init__(self):
        """Initialize fallback hardware strategy."""
        super().__init__(
            "Fallback Hardware",
            [
                HardwareErrorCode.RESOURCE_ALLOCATION_FAILED,
                HardwareErrorCode.UNSUPPORTED_FEATURE,
                HardwareErrorCode.LOIHI_MEMORY_OVERFLOW,
                HardwareErrorCode.LOIHI_SYNAPSE_LIMIT_EXCEEDED,
                HardwareErrorCode.SPINNAKER_ROUTING_ERROR
            ]
        )
        self.fallback_order = ["loihi", "spinnaker", "truenorth", "simulated"]
    
    def _execute_strategy(self, hardware_type: str, error_details: Dict[str, Any]) -> bool:
        """Execute fallback hardware strategy."""
        logger.info(f"Applying fallback hardware recovery from {hardware_type}")
        
        # Find next hardware in fallback chain
        try:
            current_index = self.fallback_order.index(hardware_type)
            for i in range(current_index + 1, len(self.fallback_order)):
                fallback_type = self.fallback_order[i]
                fallback_hw = hardware_registry.get_hardware(fallback_type)
                
                if fallback_hw and fallback_hw.initialize():
                    logger.info(f"Successfully switched to fallback hardware: {fallback_type}")
                    return True
        except ValueError:
            # Hardware type not in fallback order
            pass
        
        # If all else fails, try simulation mode
        if hardware_type != "simulated":
            sim_hw = hardware_registry.get_hardware("simulated")
            if sim_hw and sim_hw.initialize():
                logger.info("Successfully switched to simulation mode as last resort")
                return True
        
        return False


class RecoveryManager:
    """Manager for hardware recovery strategies."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RecoveryManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize recovery manager."""
        if self._initialized:
            return
            
        self.strategies = []
        self.max_attempts = 3
        self.attempt_counts = {}
        
        # Register default strategies
        self.register_strategy(ResetStateStrategy())
        self.register_strategy(ReinitializeStrategy())
        self.register_strategy(FallbackHardwareStrategy())
        
        self._initialized = True
    
    def register_strategy(self, strategy: RecoveryStrategy) -> None:
        """Register a recovery strategy."""
        self.strategies.append(strategy)
    
    def get_strategy_for_error(self, error_code: HardwareErrorCode) -> Optional[RecoveryStrategy]:
        """Get appropriate strategy for an error code."""
        for strategy in self.strategies:
            if strategy.can_handle(error_code):
                return strategy
        return None
    
    def attempt_recovery(self, hardware_type: str, error_code: HardwareErrorCode, 
                        error_details: Dict[str, Any]) -> bool:
        """
        Attempt to recover from a hardware error.
        
        Args:
            hardware_type: Type of hardware
            error_code: Error code
            error_details: Error details
            
        Returns:
            bool: Success status
        """
        # Check if we've exceeded max attempts for this hardware
        key = f"{hardware_type}:{error_code.name}"
        if key in self.attempt_counts and self.attempt_counts[key] >= self.max_attempts:
            logger.warning(f"Max recovery attempts ({self.max_attempts}) exceeded for {key}")
            return False
        
        # Increment attempt counter
        self.attempt_counts[key] = self.attempt_counts.get(key, 0) + 1
        
        # Get appropriate strategy
        strategy = self.get_strategy_for_error(error_code)
        if not strategy:
            logger.warning(f"No recovery strategy available for {error_code.name}")
            return False
        
        # Apply strategy
        logger.info(f"Attempting recovery with strategy: {strategy.name}")
        result = strategy.apply(hardware_type, error_details)
        
        # Reset counter on success
        if result:
            self.attempt_counts[key] = 0
            
        return result


# Global recovery manager instance
recovery_manager = RecoveryManager()


def attempt_recovery(hardware_type: str, error_code: HardwareErrorCode, 
                    error_details: Dict[str, Any] = {}) -> bool:
    """
    Attempt to recover from a hardware error.
    
    Args:
        hardware_type: Type of hardware
        error_code: Error code
        error_details: Error details
        
    Returns:
        bool: Success status
    """
    return recovery_manager.attempt_recovery(hardware_type, error_code, error_details or {})
"""
Base training optimization module for neuromorphic hardware.

Provides optimization strategies specifically for training on neuromorphic hardware.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type

from src.core.utils.logging_framework import get_logger

logger = get_logger("training_optimization")


class TrainingOptimizer(ABC):
    """Base class for hardware-specific training optimizations."""
    
    def __init__(self, hardware_type: str):
        """Initialize the training optimizer."""
        self.hardware_type = hardware_type
        logger.info(f"Initialized training optimizer for {hardware_type}")
    
    @abstractmethod
    def optimize_learning_rate(self, current_lr: float, epoch: int, performance: Dict[str, float]) -> float:
        """
        Optimize learning rate based on training performance.
        
        Args:
            current_lr: Current learning rate
            epoch: Current training epoch
            performance: Performance metrics (loss, accuracy, etc.)
            
        Returns:
            float: Optimized learning rate
        """
        pass
    
    @abstractmethod
    def optimize_batch_size(self, current_batch_size: int, hardware_metrics: Dict[str, Any]) -> int:
        """
        Optimize batch size based on hardware metrics.
        
        Args:
            current_batch_size: Current batch size
            hardware_metrics: Hardware performance metrics
            
        Returns:
            int: Optimized batch size
        """
        pass
    
    @abstractmethod
    def get_hardware_specific_recommendations(self) -> List[str]:
        """
        Get hardware-specific training recommendations.
        
        Returns:
            List[str]: List of recommendations
        """
        pass


class TrainingOptimizerRegistry:
    """Registry for training optimizers."""
    
    _optimizers: Dict[str, Type[TrainingOptimizer]] = {}
    
    @classmethod
    def register(cls, hardware_type: str, optimizer_class: Type[TrainingOptimizer]) -> None:
        """Register an optimizer for a hardware type."""
        cls._optimizers[hardware_type.lower()] = optimizer_class
    
    @classmethod
    def get_optimizer(cls, hardware_type: str) -> Optional[TrainingOptimizer]:
        """Get the optimizer for a hardware type."""
        optimizer_class = cls._optimizers.get(hardware_type.lower())
        if optimizer_class:
            return optimizer_class(hardware_type)
        return None


def get_training_optimizer(hardware_type: str) -> Optional[TrainingOptimizer]:
    """
    Get a training optimizer for the specified hardware type.
    
    Args:
        hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth')
        
    Returns:
        Optional[TrainingOptimizer]: Hardware-specific training optimizer
    """
    return TrainingOptimizerRegistry.get_optimizer(hardware_type)
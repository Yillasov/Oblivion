"""
Loihi-specific training optimizations.

Provides optimization strategies for training on Intel Loihi hardware.
"""

from typing import Dict, Any, List

from src.core.training.hardware_optimizations.base import TrainingOptimizer, TrainingOptimizerRegistry
from src.core.utils.logging_framework import get_logger

logger = get_logger("loihi_training_optimizer")


class LoihiTrainingOptimizer(TrainingOptimizer):
    """Training optimizer for Intel Loihi hardware."""
    
    def __init__(self, hardware_type: str = "loihi"):
        """Initialize the Loihi training optimizer."""
        super().__init__(hardware_type)
        
    def optimize_learning_rate(self, current_lr: float, epoch: int, performance: Dict[str, float]) -> float:
        """Optimize learning rate for Loihi hardware."""
        # Simple decay strategy for Loihi
        if epoch > 0 and epoch % 5 == 0:
            return current_lr * 0.9
        return current_lr
    
    def optimize_batch_size(self, current_batch_size: int, hardware_metrics: Dict[str, Any]) -> int:
        """Optimize batch size for Loihi hardware."""
        # Loihi performs better with smaller batch sizes
        utilization = hardware_metrics.get("core_utilization", 0.0)
        
        if utilization > 0.8:
            return max(1, current_batch_size - 4)
        return current_batch_size
    
    def get_hardware_specific_recommendations(self) -> List[str]:
        """Get Loihi-specific training recommendations."""
        return [
            "Use sparse connectivity for better performance on Loihi",
            "Prefer binary or ternary weights for optimal hardware utilization",
            "Limit neuron fan-in to 4096 for best performance"
        ]


# Register the optimizer
TrainingOptimizerRegistry.register("loihi", LoihiTrainingOptimizer)
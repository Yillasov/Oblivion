#!/usr/bin/env python3
"""
SpiNNaker-specific training optimizations.

Provides optimization strategies for training on SpiNNaker hardware.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List

from src.core.training.hardware_optimizations.base import TrainingOptimizer, TrainingOptimizerRegistry
from src.core.utils.logging_framework import get_logger

logger = get_logger("spinnaker_training_optimizer")


class SpiNNakerTrainingOptimizer(TrainingOptimizer):
    """Training optimizer for SpiNNaker hardware."""
    
    def __init__(self, hardware_type: str = "spinnaker"):
        """Initialize the SpiNNaker training optimizer."""
        super().__init__(hardware_type)
        
    def optimize_learning_rate(self, current_lr: float, epoch: int, performance: Dict[str, float]) -> float:
        """Optimize learning rate for SpiNNaker hardware."""
        # Step decay for SpiNNaker
        if epoch > 0 and epoch % 10 == 0:
            return current_lr * 0.8
        return current_lr
    
    def optimize_batch_size(self, current_batch_size: int, hardware_metrics: Dict[str, Any]) -> int:
        """Optimize batch size for SpiNNaker hardware."""
        # SpiNNaker can handle larger batches but with diminishing returns
        packet_loss = hardware_metrics.get("packet_loss", 0.0)
        
        if packet_loss > 0.05:
            return max(1, current_batch_size - 8)
        return current_batch_size
    
    def get_hardware_specific_recommendations(self) -> List[str]:
        """Get SpiNNaker-specific training recommendations."""
        return [
            "Distribute neurons evenly across cores to minimize packet routing",
            "Use local learning rules when possible to reduce communication overhead",
            "Consider time-multiplexing for large networks"
        ]


# Register the optimizer
TrainingOptimizerRegistry.register("spinnaker", SpiNNakerTrainingOptimizer)
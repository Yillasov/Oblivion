#!/usr/bin/env python3
"""
TrueNorth-specific training optimizations.

Provides optimization strategies for training on IBM TrueNorth hardware.
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

logger = get_logger("truenorth_training_optimizer")


class TrueNorthTrainingOptimizer(TrainingOptimizer):
    """Training optimizer for IBM TrueNorth hardware."""
    
    def __init__(self, hardware_type: str = "truenorth"):
        """Initialize the TrueNorth training optimizer."""
        super().__init__(hardware_type)
        
    def optimize_learning_rate(self, current_lr: float, epoch: int, performance: Dict[str, float]) -> float:
        """Optimize learning rate for TrueNorth hardware."""
        # TrueNorth requires offline training, so we use a more aggressive decay
        return current_lr * (0.95 ** epoch)
    
    def optimize_batch_size(self, current_batch_size: int, hardware_metrics: Dict[str, Any]) -> int:
        """Optimize batch size for TrueNorth hardware."""
        # TrueNorth has fixed hardware constraints
        # Batch size optimization is less relevant for offline training
        return min(64, current_batch_size)  # Cap at 64 for TrueNorth
    
    def get_hardware_specific_recommendations(self) -> List[str]:
        """Get TrueNorth-specific training recommendations."""
        return [
            "Use binary weights (0,1) for TrueNorth compatibility",
            "Limit connectivity to 256 inputs per neuron",
            "Consider offline training with constrained network conversion"
        ]


# Register the optimizer
TrainingOptimizerRegistry.register("truenorth", TrueNorthTrainingOptimizer)
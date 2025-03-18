"""
Hardware-specific training optimizations.

Provides optimization strategies for training on different neuromorphic hardware platforms.
"""

from src.core.training.hardware_optimizations.base import (
    TrainingOptimizer, 
    TrainingOptimizerRegistry,
    get_training_optimizer
)

# Import hardware-specific optimizers to register them
from src.core.training.hardware_optimizations.loihi_optimizer import LoihiTrainingOptimizer
from src.core.training.hardware_optimizations.spinnaker_optimizer import SpiNNakerTrainingOptimizer
from src.core.training.hardware_optimizations.truenorth_optimizer import TrueNorthTrainingOptimizer

__all__ = [
    'TrainingOptimizer',
    'TrainingOptimizerRegistry',
    'get_training_optimizer',
    'LoihiTrainingOptimizer',
    'SpiNNakerTrainingOptimizer',
    'TrueNorthTrainingOptimizer'
]
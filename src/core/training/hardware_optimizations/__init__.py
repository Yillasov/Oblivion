#!/usr/bin/env python3
"""
Hardware-specific training optimizations.

Provides optimization strategies for training on different neuromorphic hardware platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
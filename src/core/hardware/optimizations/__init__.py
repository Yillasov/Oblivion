#!/usr/bin/env python3
"""
Hardware-specific optimization modules.

Provides optimizations tailored to different neuromorphic hardware platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.hardware.optimizations.base import HardwareOptimizer, OptimizationRegistry
# Import hardware-specific optimizers to register them
from src.core.hardware.optimizations.loihi import LoihiOptimizer
from src.core.hardware.optimizations.truenorth import TrueNorthOptimizer
from src.core.hardware.optimizations.spinnaker import SpiNNakerOptimizer

__all__ = [
    'HardwareOptimizer',
    'OptimizationRegistry',
    'LoihiOptimizer',
    'TrueNorthOptimizer',
    'SpiNNakerOptimizer'
]


def get_optimizer(hardware_type: str) -> HardwareOptimizer:
    """
    Get the optimizer for a specific hardware type.
    
    Args:
        hardware_type: Type of hardware ('loihi', 'truenorth', 'spinnaker')
        
    Returns:
        HardwareOptimizer: Hardware-specific optimizer
    """
    optimizer = OptimizationRegistry.get_optimizer(hardware_type)
    if not optimizer:
        raise ValueError(f"No optimizer registered for hardware type: {hardware_type}")
    return optimizer
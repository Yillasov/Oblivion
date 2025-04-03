#!/usr/bin/env python3
"""
Hardware-specific training adapters.

Provides specialized training implementations for different neuromorphic hardware platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Type, Optional
from src.core.training.trainer_base import NeuromorphicTrainer, TrainingConfig

# Import hardware-specific adapters
from src.core.training.hardware_adapters.loihi_adapter import LoihiTrainer
from src.core.training.hardware_adapters.default_adapter import DefaultTrainer
from src.core.training.hardware_adapters.spinnaker_adapter import SpiNNakerTrainer
from src.core.training.hardware_adapters.truenorth_adapter import TrueNorthTrainer


class TrainerRegistry:
    """Registry for hardware-specific trainers."""
    
    _trainers: Dict[str, Type[NeuromorphicTrainer]] = {}
    
    @classmethod
    def register(cls, hardware_type: str, trainer_class: Type[NeuromorphicTrainer]) -> None:
        """Register a trainer for a specific hardware type."""
        cls._trainers[hardware_type.lower()] = trainer_class
    
    @classmethod
    def get_trainer(cls, hardware_type: str, config: Optional[TrainingConfig] = None) -> Optional[NeuromorphicTrainer]:
        """Get a trainer instance for the specified hardware type."""
        trainer_class = cls._trainers.get(hardware_type.lower())
        if trainer_class:
            return trainer_class(config)
        return None


# Register hardware-specific trainers
TrainerRegistry.register("loihi", LoihiTrainer)
TrainerRegistry.register("default", DefaultTrainer)
TrainerRegistry.register("simulated", DefaultTrainer)
TrainerRegistry.register("spinnaker", SpiNNakerTrainer)
TrainerRegistry.register("truenorth", TrueNorthTrainer)


def create_trainer(hardware_type: str, config: Optional[TrainingConfig] = None) -> NeuromorphicTrainer:
    """
    Create a hardware-specific trainer.
    
    Args:
        hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth', 'simulated')
        config: Optional training configuration
        
    Returns:
        NeuromorphicTrainer: Hardware-specific trainer instance
    """
    trainer = TrainerRegistry.get_trainer(hardware_type, config)
    if trainer is None:
        # Use default trainer as fallback
        trainer = DefaultTrainer(config)
    
    return trainer
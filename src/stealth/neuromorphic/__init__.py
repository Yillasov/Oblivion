"""
Neuromorphic stealth systems and controllers.

This module provides neuromorphic controllers and adaptive stealth systems
that can dynamically respond to changing threat environments.
"""

from src.stealth.neuromorphic.controller import (
    AdaptationStrategy,
    StealthNeuromorphicController
)

from src.stealth.neuromorphic.adaptive_stealth import (
    AdaptiveStealthSystem
)

from src.stealth.neuromorphic.integration import (
    NeuromorphicStealthIntegration
)

from src.stealth.neuromorphic.optimization import (
    StealthOptimizationScheduler,
    MultiObjectiveOptimizer
)

from src.stealth.neuromorphic.learning import (
    LearningMode,
    StealthEffectivenessLearner
)

__all__ = [
    'AdaptationStrategy',
    'StealthNeuromorphicController',
    'AdaptiveStealthSystem',
    'NeuromorphicStealthIntegration',
    'StealthOptimizationScheduler',
    'MultiObjectiveOptimizer',
    'LearningMode',
    'StealthEffectivenessLearner'
]
"""
Base interfaces and abstract classes for stealth systems.

This module provides the core interfaces and abstract classes for implementing
various stealth technologies in the Oblivion SDK.
"""

from src.stealth.base.interfaces import (
    StealthInterface,
    NeuromorphicStealth,
    StealthSpecs,
    StealthType
)

from src.stealth.base.config import (
    StealthEffectivenessLevel,
    StealthPowerMode,
    StealthOperationalMode,
    StealthMaterialConfig,
    StealthSignatureConfig,
    StealthSystemConfig,
    StealthConfigTemplates
)

__all__ = [
    'StealthInterface',
    'NeuromorphicStealth',
    'StealthSpecs',
    'StealthType',
    'StealthEffectivenessLevel',
    'StealthPowerMode',
    'StealthOperationalMode',
    'StealthMaterialConfig',
    'StealthSignatureConfig',
    'StealthSystemConfig',
    'StealthConfigTemplates'
]
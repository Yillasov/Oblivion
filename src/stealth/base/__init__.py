#!/usr/bin/env python3
"""
Base interfaces and abstract classes for stealth systems.

This module provides the core interfaces and abstract classes for implementing
various stealth technologies in the Oblivion SDK.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
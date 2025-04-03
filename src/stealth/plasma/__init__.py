#!/usr/bin/env python3
"""
Plasma Stealth module for stealth systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.plasma.plasma_system import PlasmaStealthSystem, PlasmaParameters
from src.stealth.plasma.plasma_generator import (
    PlasmaGenerator, 
    PlasmaControlSystem, 
    PlasmaGeneratorSpecs,
    PlasmaGenerationMethod,
    PlasmaPulsePattern
)
from src.stealth.plasma.plasma_simulation import PlasmaRadarSimulator
from src.stealth.plasma.power_management import (
    PlasmaStealthPowerManager,
    PowerProfile,
    PowerState,
    PowerOptimizer,
    PowerMonitor,
    create_default_power_profile
)

__all__ = [
    'PlasmaStealthSystem',
    'PlasmaParameters',
    'PlasmaGenerator',
    'PlasmaControlSystem',
    'PlasmaGeneratorSpecs',
    'PlasmaGenerationMethod',
    'PlasmaPulsePattern',
    'PlasmaRadarSimulator',
    'PlasmaStealthPowerManager',
    'PowerProfile',
    'PowerState',
    'PowerOptimizer',
    'PowerMonitor',
    'create_default_power_profile'
]
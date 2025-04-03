#!/usr/bin/env python3
"""
Conventional payload systems for UCAV platforms.

This module provides implementations of traditional payload systems
that have been enhanced with neuromorphic capabilities.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.payload.conventional.weapons import (
    ConventionalWeapon, AirToAirMissile, GuidedBomb, GunSystem, WeaponSpecs
)
from src.payload.conventional.sensors import (
    ConventionalSensor, RadarSystem, ElectroOpticalSystem, SensorSpecs
)
from src.payload.conventional.electronic_warfare import (
    ElectronicWarfareSystem, JammingSystem, SignalsIntelligenceSystem, ElectronicWarfareSpecs
)

__all__ = [
    'ConventionalWeapon',
    'AirToAirMissile',
    'GuidedBomb',
    'GunSystem',
    'WeaponSpecs',
    'ConventionalSensor',
    'RadarSystem',
    'ElectroOpticalSystem',
    'SensorSpecs',
    'ElectronicWarfareSystem',
    'JammingSystem',
    'SignalsIntelligenceSystem',
    'ElectronicWarfareSpecs'
]
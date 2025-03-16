"""
Conventional payload systems for UCAV platforms.

This module provides implementations of traditional payload systems
that have been enhanced with neuromorphic capabilities.
"""

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
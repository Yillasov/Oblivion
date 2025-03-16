"""
Payload module for Oblivion SDK.

This module provides classes and utilities for designing, integrating,
and optimizing payload systems for Unmanned Combat Aerial Vehicles (UCAVs).
"""

from src.payload.base import PayloadInterface, NeuromorphicPayload, PayloadSpecs
from src.payload.types import (
    PayloadCategory, PayloadMountType, WeaponType, SensorType,
    ElectronicWarfareType, CountermeasureType, PAYLOAD_TEMPLATES
)
from src.payload.integration import PayloadIntegrator

__all__ = [
    'PayloadInterface',
    'NeuromorphicPayload',
    'PayloadSpecs',
    'PayloadCategory',
    'PayloadMountType',
    'WeaponType',
    'SensorType',
    'ElectronicWarfareType',
    'CountermeasureType',
    'PAYLOAD_TEMPLATES',
    'PayloadIntegrator'
]
#!/usr/bin/env python3
"""
Payload module for Oblivion SDK.

This module provides classes and utilities for designing, integrating,
and optimizing payload systems for Unmanned Combat Aerial Vehicles (UCAVs).
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
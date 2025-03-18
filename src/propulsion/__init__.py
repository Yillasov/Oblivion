"""
Propulsion module for Oblivion SDK.

This module provides classes and utilities for designing, integrating,
and optimizing propulsion systems for Unmanned Combat Aerial Vehicles (UCAVs).
"""

from src.propulsion.base import (
    PropulsionInterface,
    NeuromorphicPropulsion,
    PropulsionSpecs,
    PropulsionType
)

__all__ = [
    'PropulsionInterface',
    'NeuromorphicPropulsion',
    'PropulsionSpecs',
    'PropulsionType'
]
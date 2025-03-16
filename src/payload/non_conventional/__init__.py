"""
Non-conventional payload systems for UCAV platforms.

This module provides implementations of advanced payload systems
that leverage neuromorphic computing for enhanced capabilities.
"""

from src.payload.non_conventional.directed_energy import (
    DirectedEnergyWeapon, HighEnergyLaser, DirectedEnergySpecs
)
from src.payload.non_conventional.drone_systems import (
    DroneDeploymentSystem, MicroDroneSwarm, DroneSystemSpecs
)
from src.payload.non_conventional.countermeasures import (
    AdaptiveCountermeasure, AdaptiveDecoy, CountermeasureSpecs
)

__all__ = [
    'DirectedEnergyWeapon',
    'HighEnergyLaser',
    'DirectedEnergySpecs',
    'DroneDeploymentSystem',
    'MicroDroneSwarm',
    'DroneSystemSpecs',
    'AdaptiveCountermeasure',
    'AdaptiveDecoy',
    'CountermeasureSpecs'
]
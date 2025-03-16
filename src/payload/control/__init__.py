"""
Neuromorphic payload control systems for UCAV platforms.

This module provides centralized control and coordination systems
for managing payload operations using neuromorphic computing.
"""

from src.payload.control.neuromorphic_controller import NeuromorphicPayloadController
from src.payload.control.payload_coordination import PayloadCoordinator, CoordinationStrategy

__all__ = [
    'NeuromorphicPayloadController',
    'PayloadCoordinator',
    'CoordinationStrategy'
]
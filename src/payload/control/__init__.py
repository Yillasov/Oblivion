#!/usr/bin/env python3
"""
Neuromorphic payload control systems for UCAV platforms.

This module provides centralized control and coordination systems
for managing payload operations using neuromorphic computing.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.payload.control.neuromorphic_controller import NeuromorphicPayloadController
from src.payload.control.payload_coordination import PayloadCoordinator, CoordinationStrategy

__all__ = [
    'NeuromorphicPayloadController',
    'PayloadCoordinator',
    'CoordinationStrategy'
]
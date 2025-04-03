#!/usr/bin/env python3
"""
Power supply systems for UCAV platforms.

This module provides implementations of advanced power supply systems
that leverage neuromorphic computing for enhanced capabilities.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.power.base import (
    PowerSupplyInterface, PowerSupplySpecs, PowerSupplyType, NeuromorphicPowerSupply
)

__all__ = [
    'PowerSupplyInterface',
    'PowerSupplySpecs',
    'PowerSupplyType',
    'NeuromorphicPowerSupply'
]
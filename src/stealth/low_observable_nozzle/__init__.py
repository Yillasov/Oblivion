#!/usr/bin/env python3
"""
Low-Observable Nozzle module for stealth propulsion integration.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.low_observable_nozzle.low_observable_nozzle import LowObservableNozzle, NozzleParameters

__all__ = [
    'LowObservableNozzle',
    'NozzleParameters'
]
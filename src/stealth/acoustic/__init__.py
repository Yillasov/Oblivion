#!/usr/bin/env python3
"""
Acoustic signature reduction technologies for stealth systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.acoustic.acoustic_reduction import AcousticReductionSystem, AcousticParameters

__all__ = [
    'AcousticReductionSystem',
    'AcousticParameters'
]
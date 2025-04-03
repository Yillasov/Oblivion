#!/usr/bin/env python3
"""
Materials-based stealth technologies for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.materials.ram import RAMSystem, RAMMaterial

__all__ = [
    'RAMSystem',
    'RAMMaterial'
]
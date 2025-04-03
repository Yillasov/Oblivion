#!/usr/bin/env python3
"""
Radar Absorbent Material (RAM) module.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.materials.ram.ram_material import RAMMaterial
from src.stealth.materials.ram.ram_system import RAMSystem, NeuromorphicStealth
from src.stealth.materials.ram.material_database import RAMMaterialDatabase

__all__ = ['RAMSystem', 'RAMMaterial', 'RAMMaterialDatabase', 'NeuromorphicStealth']
#!/usr/bin/env python3
"""
Electromagnetic shielding technologies for stealth systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.electromagnetic.emp_shielding import EMPShieldingSystem, EMPShieldingParameters

__all__ = [
    'EMPShieldingSystem',
    'EMPShieldingParameters'
]
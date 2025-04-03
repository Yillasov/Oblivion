#!/usr/bin/env python3
"""
Metamaterial-based stealth technologies for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.metamaterial.metamaterial_cloaking import MetamaterialCloaking, MetamaterialProperties

__all__ = [
    'MetamaterialCloaking',
    'MetamaterialProperties'
]
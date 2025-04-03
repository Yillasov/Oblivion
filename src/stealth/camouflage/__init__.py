#!/usr/bin/env python3
"""
Active Camouflage module for stealth systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.camouflage.active_camouflage import ActiveCamouflageSystem, CamouflageParameters
from src.stealth.camouflage.visual_signature import (
    VisualSignatureMatcher, 
    VisualSignature, 
    MatchingAlgorithm
)

__all__ = [
    'ActiveCamouflageSystem',
    'CamouflageParameters',
    'VisualSignatureMatcher',
    'VisualSignature',
    'MatchingAlgorithm'
]
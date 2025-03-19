"""
Active Camouflage module for stealth systems.
"""

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
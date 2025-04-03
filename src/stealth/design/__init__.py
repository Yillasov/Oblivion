#!/usr/bin/env python3
"""
Parametric design systems for stealth features.

This module provides tools for CAD integration and parametric design
of stealth features for various platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.design.parametric_design import (
    DesignOptimizationGoal,
    StealthFeatureType,
    ParametricStealthDesigner,
    CADIntegration
)

__all__ = [
    'DesignOptimizationGoal',
    'StealthFeatureType',
    'ParametricStealthDesigner',
    'CADIntegration'
]
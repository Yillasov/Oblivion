#!/usr/bin/env python3
"""
Manufacturing systems for UCAV payload integration.

This module provides tools for integrating payload designs with
manufacturing processes and quality control systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.manufacturing.integration import ManufacturingIntegration, ManufacturingSpec

__all__ = [
    'ManufacturingIntegration',
    'ManufacturingSpec'
]
#!/usr/bin/env python3
"""
Testing systems for UCAV payload validation.

This module provides tools for automated testing of payload functionality,
performance, and integration with other systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.testing.framework import TestingFramework, TestCase, TestResult

__all__ = [
    'TestingFramework',
    'TestCase',
    'TestResult'
]
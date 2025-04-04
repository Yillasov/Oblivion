#!/usr/bin/env python3
"""
Biomimetic Design Package

This package provides tools and frameworks for biomimetic design of UCAVs,
inspired by biological flying organisms and their adaptations.
"""

from src.biomimetic.design.principles import (
    BiologicalInspiration,
    BiomimeticPrinciple,
    BiologicalReference,
    BiomimeticDesignFramework
)
from src.biomimetic.design.reference_database import BiologicalReferenceDatabase
from src.biomimetic.design.parameter_mapping import BiomimeticParameterMapper, ParameterMapping
from src.biomimetic.design.biomimetic_parametric import BiomimeticParametricDesign

__all__ = [
    'BiologicalInspiration',
    'BiomimeticPrinciple',
    'BiologicalReference',
    'BiomimeticDesignFramework',
    'BiologicalReferenceDatabase',
    'BiomimeticParameterMapper',
    'ParameterMapping',
    'BiomimeticParametricDesign'
]
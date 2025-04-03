#!/usr/bin/env python3
"""
Stealth module for Oblivion SDK.

This module provides classes and utilities for designing, integrating,
and optimizing stealth systems for Unmanned Combat Aerial Vehicles (UCAVs).
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stealth.base.interfaces import (
    StealthInterface,
    NeuromorphicStealth,
    StealthSpecs,
    StealthType
)

from src.stealth.base.config import (
    StealthEffectivenessLevel,
    StealthPowerMode,
    StealthOperationalMode,
    StealthMaterialConfig,
    StealthSignatureConfig,
    StealthSystemConfig,
    StealthConfigTemplates
)

from src.stealth.materials.ram import RAMSystem, RAMMaterial
from src.stealth.plasma import PlasmaStealthSystem, PlasmaParameters
from src.stealth.camouflage.active_camouflage import ActiveCamouflageSystem, CamouflageParameters
from src.stealth.infrared import InfraredSuppressionSystem, IRSuppressionParameters
from src.stealth.acoustic import AcousticReductionSystem, AcousticParameters
from src.stealth.shape_shifting import ShapeShiftingSurfaces, ShapeShiftingParameters
from src.stealth.thermal_camouflage import AdaptiveThermalCamouflage, ThermalCamouflageParameters
from src.stealth.simulation.material_sim import MaterialPropertiesSimulator
from src.stealth.simulation.manufacturing_integration import (
    StealthManufacturingIntegration,
    StealthManufacturingSpec,
    StealthManufacturingStage
)

__all__ = [
    'StealthInterface',
    'NeuromorphicStealth',
    'StealthSpecs',
    'StealthType',
    'StealthEffectivenessLevel',
    'StealthPowerMode',
    'StealthOperationalMode',
    'StealthMaterialConfig',
    'StealthSignatureConfig',
    'StealthSystemConfig',
    'StealthConfigTemplates',
    'RAMSystem',
    'RAMMaterial',
    'PlasmaStealthSystem',
    'PlasmaParameters',
    'ActiveCamouflageSystem',
    'CamouflageParameters',
    'InfraredSuppressionSystem',
    'IRSuppressionParameters',
    'AcousticReductionSystem',
    'AcousticParameters',
    'ShapeShiftingSurfaces',
    'ShapeShiftingParameters',
    'AdaptiveThermalCamouflage',
    'ThermalCamouflageParameters',
    'MaterialPropertiesSimulator',
    'StealthManufacturingIntegration',
    'StealthManufacturingSpec',
    'StealthManufacturingStage'
]
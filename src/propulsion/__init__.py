#!/usr/bin/env python3
"""
Propulsion module for Oblivion SDK.

This module provides classes and utilities for designing, integrating,
and optimizing propulsion systems for Unmanned Combat Aerial Vehicles (UCAVs).
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import base interfaces and types
from src.propulsion.base import (
    PropulsionInterface,
    PropulsionSpecs,
    PropulsionType
)

# Import propulsion systems
from src.propulsion.ion_thruster import IonThrusterController, IonThrusterMode, IonThrusterSpecs
from src.propulsion.plasma_thruster import PlasmaThrusterManager, PlasmaMode, PlasmaThrusterSpecs
from src.propulsion.magnetic_levitation import MagneticLevitationController, MaglevMode, MaglevSpecs
from src.propulsion.hydrogen_fuel_cell import HydrogenFuelCellManager, FuelCellSpecs
from src.propulsion.hybrid_electric import HybridElectricController

# Import optimization and integration
from src.propulsion.optimization import PropulsionOptimizer, OptimizationConstraints
from src.propulsion.integration import PropulsionIntegrator, PropulsionIntegrationConfig
from src.propulsion.fuel_flow_control import FuelFlowController, FuelFlowProfile, FuelFlowMode
from src.propulsion.environmental_interaction import EnvironmentalInteractionSystem, EnvironmentalCondition
from src.propulsion.stealth_integration import PropulsionStealthIntegrator

# Import manufacturing
from src.propulsion.manufacturing_workflow_simple import (
    PropulsionManufacturingWorkflow,
    ManufacturingStage,
    ManufacturingConfig
)

__all__ = [
    'PropulsionInterface',
    'PropulsionSpecs',
    'PropulsionType',
    'IonThrusterController',
    'IonThrusterMode',
    'IonThrusterSpecs',
    'PlasmaThrusterManager',
    'PlasmaMode',
    'PlasmaThrusterSpecs',
    'MagneticLevitationController',
    'MaglevMode',
    'MaglevSpecs',
    'HydrogenFuelCellManager',
    'FuelCellSpecs',
    'HybridElectricController',
    'PropulsionOptimizer',
    'OptimizationConstraints',
    'PropulsionIntegrator',
    'PropulsionIntegrationConfig',
    'FuelFlowController',
    'FuelFlowProfile',
    'FuelFlowMode',
    'EnvironmentalInteractionSystem',
    'EnvironmentalCondition',
    'PropulsionManufacturingWorkflow',
    'ManufacturingStage',
    'ManufacturingConfig',
    'PropulsionStealthIntegrator'
]
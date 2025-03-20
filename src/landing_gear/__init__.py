"""
Landing gear module for Oblivion SDK.

This module provides classes and utilities for designing, integrating,
and optimizing landing gear systems for Unmanned Combat Aerial Vehicles (UCAVs).
"""

from src.landing_gear.base import LandingGearInterface, NeuromorphicLandingGear, LandingGearSpecs
from src.landing_gear.types import LandingGearType
from src.landing_gear.emergency import EmergencyHandler, FailureType, RecoveryAction
from src.landing_gear.neuromorphic_control import NeuromorphicLandingController, AdaptiveMode
from src.landing_gear.hardware_interface import NeuromorphicHardwareInterface, LandingGearNeuromorphicInterface
from src.landing_gear.learning import LandingOptimizer, LearningStrategy, LandingExperience
from src.landing_gear.implementations import (
    RetractableMorphingGear,
    ElectromagneticCatapultGear,
    VTOLRotorsGear,
    AirCushionGear,
    AdaptiveShockGear,
    create_landing_gear
)
from src.landing_gear.simulation import LandingGearSimulation, LandingEnvironment, TerrainType
from src.landing_gear.stress_testing import (
    LandingGearStressTester,
    StressTestType,
    FailureAnalysisResult
)
from src.landing_gear.integration import LandingGearIntegration
from src.landing_gear.messaging import LandingGearMessaging
from src.landing_gear.manufacturing import (
    LandingGearManufacturingProcess,
    LandingGearManufacturingSpec,
    LandingGearManufacturingIntegration
)

__all__ = [
    'LandingGearInterface',
    'NeuromorphicLandingGear',
    'LandingGearSpecs',
    'LandingGearType',
    'EmergencyHandler',
    'FailureType',
    'RecoveryAction',
    'NeuromorphicLandingController',
    'AdaptiveMode',
    'NeuromorphicHardwareInterface',
    'LandingGearNeuromorphicInterface',
    'LandingOptimizer',
    'LearningStrategy',
    'LandingExperience',
    'RetractableMorphingGear',
    'ElectromagneticCatapultGear',
    'VTOLRotorsGear',
    'AirCushionGear',
    'AdaptiveShockGear',
    'create_landing_gear',
    'LandingGearSimulation',
    'LandingEnvironment',
    'TerrainType',
    'LandingGearStressTester',
    'StressTestType',
    'FailureAnalysisResult',
    'LandingGearIntegration',
    'LandingGearMessaging',
    'LandingGearManufacturingProcess',
    'LandingGearManufacturingSpec',
    'LandingGearManufacturingIntegration'
]
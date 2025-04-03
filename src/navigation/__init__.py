#!/usr/bin/env python3
"""
Navigation systems for UCAV platforms.

This module provides implementations of advanced navigation systems
that leverage neuromorphic computing for enhanced capabilities.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.navigation.base import (
    NavigationSystem, NavigationSpecs, NavigationType
)
from src.navigation.integration import NavigationIntegrator
from src.navigation.interfaces import (
    NavigationState, NavigationServiceProvider,
    PropulsionInterface, SensorInterface,
    CommunicationInterface, MissionInterface
)
from src.navigation.common import (
    PositionProvider, OrientationProvider, 
    VelocityProvider, NavigationDataProvider
)
from src.navigation.error_handling import (
    NavigationError, SensorDataError, PositionEstimationError,
    OrientationEstimationError, NavigationSystemFailure,
    NavigationErrorHandler, UncertaintyQuantifier,
    safe_navigation_operation, navigation_error_handler
)
from src.navigation.system_management import NavigationSystemManager
from src.navigation.quantum_inertial import QuantumInertialNavigation
from src.navigation.quantum_algorithms import (
    QuantumParameters, QuantumAlgorithms, QuantumAlgorithmType
)
from src.navigation.neuromorphic_quantum import (
    NeuromorphicQuantumEstimator, EncodingMethod
)
from src.navigation.star_tracker import (
    StarTracker, StarTrackerSpecs, StarPatternAlgorithm,
    StarCatalogEntry
)
from src.navigation.celestial_database import CelestialDatabase
from src.navigation.attitude_determination import (
    AttitudeDetermination, AttitudeConfig, AttitudeMethod
)
from src.navigation.terrain_database import (
    TerrainDatabase, TerrainFeature, TerrainFeatureType
)
from src.navigation.predictive_terrain import (
    PredictiveTerrainModeling, TerrainPrediction, TerrainRiskLevel
)
from src.navigation.terrain_obstacle_avoidance import (
    TerrainObstacleAvoidance, TerrainObstacle
)
from src.navigation.multi_sensor_fusion_ai import (
    MultiSensorFusionAI, FusionAIConfig, FusionAIMode
)
from src.navigation.magnetic_field_mapping import (
    MagneticFieldMap, MagneticAnomaly, MagneticAnomalyType, MagneticFieldConfig
)
from src.navigation.gravitational_anomaly import (
    GravitationalAnomalySensor, GravityAnomaly, GravityAnomalyType, GravityAnomalyConfig
)
from src.navigation.pulsar_positioning import (
    PulsarPositioningSystem, PulsarData, PulsarType, PulsarPositioningConfig
)
from src.navigation.bio_odometry import (
    BioInspiredOdometry, BioOdometryConfig, BioInspirationSource
)
from src.navigation.celestial_arrays import (
    CelestialNavigationArrays, CelestialArrayConfig, CelestialArrayType
)
from src.navigation.atmospheric_guidance import (
    AtmosphericPressureGuidance, AtmosphericGuidanceConfig, PressurePatternType
)
from src.navigation.testing import (
    NavigationTestFramework, TestScenario, TestParameters
)

__all__ = [
    'NavigationSystem',
    'NavigationSpecs',
    'NavigationType',
    'NavigationIntegrator',
    'NavigationState',
    'NavigationServiceProvider',
    'PropulsionInterface',
    'SensorInterface',
    'CommunicationInterface',
    'MissionInterface',
    'PositionProvider',
    'OrientationProvider',
    'VelocityProvider',
    'NavigationDataProvider',
    'NavigationError',
    'SensorDataError',
    'PositionEstimationError',
    'OrientationEstimationError',
    'NavigationSystemFailure',
    'NavigationErrorHandler',
    'UncertaintyQuantifier',
    'safe_navigation_operation',
    'navigation_error_handler',
    'NavigationSystemManager',
    'QuantumInertialNavigation',
    'QuantumParameters',
    'QuantumAlgorithms',
    'QuantumAlgorithmType',
    'NeuromorphicQuantumEstimator',
    'EncodingMethod',
    'StarTracker',
    'StarTrackerSpecs',
    'StarPatternAlgorithm',
    'StarCatalogEntry',
    'CelestialDatabase',
    'AttitudeDetermination',
    'AttitudeConfig',
    'AttitudeMethod',
    'TerrainDatabase',
    'TerrainFeature',
    'TerrainFeatureType',
    'PredictiveTerrainModeling',
    'TerrainPrediction',
    'TerrainRiskLevel',
    'TerrainObstacleAvoidance',
    'TerrainObstacle',
    'MultiSensorFusionAI',
    'FusionAIConfig',
    'FusionAIMode',
    'MagneticFieldMap',
    'MagneticAnomaly',
    'MagneticAnomalyType',
    'MagneticFieldConfig',
    'GravitationalAnomalySensor',
    'GravityAnomaly',
    'GravityAnomalyType',
    'GravityAnomalyConfig',
    'PulsarPositioningSystem',
    'PulsarData',
    'PulsarType',
    'PulsarPositioningConfig',
    'BioInspiredOdometry',
    'BioOdometryConfig',
    'BioInspirationSource',
    'CelestialNavigationArrays',
    'CelestialArrayConfig',
    'CelestialArrayType',
    'AtmosphericPressureGuidance',
    'AtmosphericGuidanceConfig',
    'PressurePatternType',
    'NavigationTestFramework',
    'TestScenario',
    'TestParameters'
]
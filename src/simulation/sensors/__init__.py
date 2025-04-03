#!/usr/bin/env python3
"""
Sensor simulation modules for Oblivion SDK.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.sensors.sensor_framework import (
    SensorType,
    SensorConfig,
    Sensor,
    Radar,
    Altimeter,
    SensorManager,
    create_default_sensors
)

from src.simulation.sensors.stealth_detection import (
    SignatureType,
    StealthDetectionSensor,
    RadarStealthDetector,
    IRStealthDetector,
    AcousticStealthDetector,
    EMStealthDetector,
    MultiSignatureDetector,
    create_stealth_detection_sensors
)

__all__ = [
    'SensorType',
    'SensorConfig',
    'Sensor',
    'Radar',
    'Altimeter',
    'SensorManager',
    'create_default_sensors',
    'SignatureType',
    'StealthDetectionSensor',
    'RadarStealthDetector',
    'IRStealthDetector',
    'AcousticStealthDetector',
    'EMStealthDetector',
    'MultiSignatureDetector',
    'create_stealth_detection_sensors'
]
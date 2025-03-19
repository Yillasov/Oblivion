"""
Sensor simulation modules for Oblivion SDK.
"""

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
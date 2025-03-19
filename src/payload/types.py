"""
Type definitions for UCAV payload systems.
"""

from enum import Enum, auto
from typing import Dict, Any, List, Optional


class PayloadCategory(Enum):
    """Categories of payload systems."""
    WEAPON = auto()
    SENSOR = auto()
    ELECTRONIC_WARFARE = auto()
    COMMUNICATION = auto()
    COUNTERMEASURE = auto()
    SPECIAL = auto()


class PayloadMountType(Enum):
    """Types of payload mounting systems."""
    INTERNAL_BAY = auto()
    EXTERNAL_HARDPOINT = auto()
    WING_TIP = auto()
    FUSELAGE = auto()
    CUSTOM = auto()


class WeaponType(Enum):
    """Types of weapon payloads."""
    MISSILE = auto()
    BOMB = auto()
    GUN = auto()
    DIRECTED_ENERGY = auto()
    DRONE = auto()


class SensorType(Enum):
    """Types of sensor payloads."""
    # Basic sensor types
    RADAR = auto()
    ELECTRO_OPTICAL = auto()
    INFRARED = auto()
    LIDAR = auto()
    MULTI_SPECTRAL = auto()
    SIGNALS_INTELLIGENCE = auto()
    
    # Advanced sensor types
    QUANTUM_RADAR = auto()
    HYPERSPECTRAL = auto()
    SYNTHETIC_APERTURE_RADAR = auto()
    GRAPHENE_INFRARED = auto()
    BIO_MIMETIC = auto()
    MULTI_SPECTRAL_EO_IR = auto()
    DISTRIBUTED_APERTURE = auto()
    NEUROMORPHIC_VISION = auto()
    TERAHERTZ = auto()


class ElectronicWarfareType(Enum):
    """Types of electronic warfare payloads."""
    JAMMING = auto()
    SPOOFING = auto()
    SIGNALS_INTELLIGENCE = auto()
    CYBER = auto()


class CountermeasureType(Enum):
    """Types of countermeasure payloads."""
    CHAFF = auto()
    FLARE = auto()
    DECOY = auto()
    ACTIVE_JAMMING = auto()
    DIRECTED_ENERGY = auto()


# Payload configuration templates
PAYLOAD_TEMPLATES = {
    "air_to_air": {
        "primary": [WeaponType.MISSILE],
        "secondary": [SensorType.RADAR, CountermeasureType.CHAFF],
        "recommended_mounts": [PayloadMountType.INTERNAL_BAY, PayloadMountType.EXTERNAL_HARDPOINT]
    },
    "air_to_ground": {
        "primary": [WeaponType.BOMB, WeaponType.MISSILE],
        "secondary": [SensorType.ELECTRO_OPTICAL, SensorType.INFRARED],
        "recommended_mounts": [PayloadMountType.INTERNAL_BAY, PayloadMountType.EXTERNAL_HARDPOINT]
    },
    "reconnaissance": {
        "primary": [SensorType.MULTI_SPECTRAL, SensorType.SIGNALS_INTELLIGENCE],
        "secondary": [CountermeasureType.ACTIVE_JAMMING],
        "recommended_mounts": [PayloadMountType.INTERNAL_BAY, PayloadMountType.FUSELAGE]
    },
    "electronic_warfare": {
        "primary": [ElectronicWarfareType.JAMMING, ElectronicWarfareType.SPOOFING],
        "secondary": [SensorType.SIGNALS_INTELLIGENCE],
        "recommended_mounts": [PayloadMountType.INTERNAL_BAY, PayloadMountType.WING_TIP]
    },
    "advanced_strike": {
        "primary": [WeaponType.DIRECTED_ENERGY, WeaponType.DRONE],
        "secondary": [ElectronicWarfareType.CYBER, SensorType.MULTI_SPECTRAL],
        "recommended_mounts": [PayloadMountType.INTERNAL_BAY, PayloadMountType.FUSELAGE]
    }
}
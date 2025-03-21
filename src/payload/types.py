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


# Add LASER_DEFENSE to your CountermeasureType enum
class CountermeasureType(Enum):
    """Types of countermeasure systems."""
    DECOY = auto()
    JAMMER = auto()
    CHAFF = auto()
    FLARE = auto()
    DIRECTED_ENERGY_JAMMER = auto()
    NANO_CHAFF_CLOUD = auto()
    LASER_DEFENSE = auto()
    ACTIVE_JAMMING = auto()  # Add this line
    ADAPTIVE_FREQUENCY_HOPPER = auto()
    HOLOGRAPHIC_DECOY = auto()
    ELECTROMAGNETIC_PULSE = auto()
    SELF_DESTRUCTING_DRONE = auto()
    BIO_MIMETIC_FLARE = auto()
    ACOUSTIC_WAVE_DISRUPTOR = auto()
    CYBER_ATTACK_PAYLOAD = auto()


class JammingFrequencyBand(Enum):
    """Frequency bands for jamming operations."""
    VHF = auto()  # Very High Frequency (30-300 MHz)
    UHF = auto()  # Ultra High Frequency (300 MHz-3 GHz)
    L_BAND = auto()  # L-Band (1-2 GHz)
    S_BAND = auto()  # S-Band (2-4 GHz)
    C_BAND = auto()  # C-Band (4-8 GHz)
    X_BAND = auto()  # X-Band (8-12 GHz)
    KU_BAND = auto()  # Ku-Band (12-18 GHz)
    K_BAND = auto()  # K-Band (18-27 GHz)
    KA_BAND = auto()  # Ka-Band (27-40 GHz)
    MILLIMETER = auto()  # Millimeter Wave (40-300 GHz)
    ADAPTIVE = auto()  # Adaptive frequency selection
    FULL_SPECTRUM = auto()  # Full spectrum jamming


class ChaffType(Enum):
    """Types of chaff countermeasures."""
    STANDARD = auto()  # Standard radar-reflecting chaff
    NANO = auto()  # Nano-scale chaff particles
    SMART = auto()  # Programmable chaff with active elements
    PERSISTENT = auto()  # Long-duration chaff cloud
    BLOOMING = auto()  # Rapidly expanding chaff cloud
    MULTI_BAND = auto()  # Multi-band frequency response
    STEALTH = auto()  # Low observable chaff with selective reflection


class LaserDefenseType(Enum):
    """Types of laser-based defense systems."""
    DAZZLER = auto()  # Sensor/optics dazzling system
    TRACKER_DISRUPTOR = auto()  # Guidance system disruptor
    HIGH_ENERGY = auto()  # High energy laser for physical damage
    MULTI_BEAM = auto()  # Multiple beam system
    PULSE = auto()  # Pulsed laser system
    CONTINUOUS = auto()  # Continuous wave laser
    ADAPTIVE_OPTICS = auto()  # System with adaptive optics


class DecoySignatureType(Enum):
    """Types of decoy signatures."""
    RADAR = auto()  # Radar cross-section simulation
    INFRARED = auto()  # Thermal signature simulation
    VISUAL = auto()  # Visual signature simulation
    ACOUSTIC = auto()  # Acoustic signature simulation
    MULTI_SPECTRAL = auto()  # Combined signature types
    HOLOGRAPHIC = auto()  # Advanced holographic projection
    ADAPTIVE = auto()  # Signature that adapts to threats


class EMPStrength(Enum):
    """Electromagnetic pulse strength levels."""
    LOW = auto()  # Low-power, short-range disruption
    MEDIUM = auto()  # Medium-power, tactical range
    HIGH = auto()  # High-power, wide area effect
    DIRECTIONAL = auto()  # Focused, directional EMP
    SUSTAINED = auto()  # Sustained emission
    PULSED = auto()  # Pulsed emission
    ADAPTIVE = auto()  # Adaptive power based on target


class CyberAttackVector(Enum):
    """Types of cyber attack vectors."""
    COMMUNICATIONS = auto()  # Communication systems disruption
    NAVIGATION = auto()  # Navigation systems attack
    SENSOR = auto()  # Sensor systems spoofing
    IDENTIFICATION = auto()  # IFF systems manipulation
    COMMAND_CONTROL = auto()  # Command and control systems
    DATA_EXFILTRATION = auto()  # Data theft operations
    SYSTEM_SHUTDOWN = auto()  # Remote system shutdown


class AcousticDisruptionMode(Enum):
    """Modes of acoustic wave disruption."""
    BROADBAND = auto()  # Broadband acoustic disruption
    TARGETED = auto()  # Frequency-targeted disruption
    RESONANT = auto()  # Resonant frequency disruption
    PULSED = auto()  # Pulsed acoustic emission
    CONTINUOUS = auto()  # Continuous acoustic emission
    ADAPTIVE = auto()  # Adaptive frequency selection


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
"""
Configuration data structures for stealth systems in the Oblivion SDK.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum, auto

from src.stealth.base.interfaces import StealthType


class StealthEffectivenessLevel(Enum):
    """Effectiveness levels for stealth systems."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class StealthPowerMode(Enum):
    """Power modes for stealth systems."""
    STANDBY = "standby"
    ECO = "eco"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    MAXIMUM = "maximum"


class StealthOperationalMode(Enum):
    """Operational modes for stealth systems."""
    INACTIVE = "inactive"
    PASSIVE = "passive"
    ACTIVE = "active"
    ADAPTIVE = "adaptive"
    EMERGENCY = "emergency"


@dataclass
class StealthMaterialConfig:
    """Configuration for stealth material properties."""
    material_type: str
    thickness_mm: float
    coverage_percentage: float
    frequency_range_ghz: Dict[str, float] = field(default_factory=lambda: {"min": 0.5, "max": 18.0})
    conductivity: Optional[float] = None
    permittivity: Optional[float] = None
    thermal_properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class StealthSignatureConfig:
    """Configuration for signature reduction properties."""
    radar_cross_section_reduction: float  # Percentage reduction
    infrared_signature_reduction: float   # Percentage reduction
    acoustic_signature_reduction: float   # Percentage reduction
    visual_signature_reduction: float     # Percentage reduction
    electromagnetic_signature_reduction: float  # Percentage reduction


@dataclass
class StealthSystemConfig:
    """Main configuration for stealth systems."""
    # Basic configuration
    stealth_type: StealthType
    name: str
    description: str = ""
    
    # Performance configuration
    effectiveness_level: StealthEffectivenessLevel = StealthEffectivenessLevel.MEDIUM
    power_mode: StealthPowerMode = StealthPowerMode.BALANCED
    operational_mode: StealthOperationalMode = StealthOperationalMode.PASSIVE
    
    # Physical properties
    weight_kg: float = 0.0
    volume_cubic_m: float = 0.0
    power_requirements_kw: float = 0.0
    
    # Operational parameters
    activation_time_seconds: float = 1.0
    cooldown_time_seconds: float = 30.0
    operational_duration_minutes: float = 60.0
    
    # Signature reduction configuration
    signature_config: StealthSignatureConfig = field(default_factory=lambda: StealthSignatureConfig(
        radar_cross_section_reduction=0.0,
        infrared_signature_reduction=0.0,
        acoustic_signature_reduction=0.0,
        visual_signature_reduction=0.0,
        electromagnetic_signature_reduction=0.0
    ))
    
    # Material configuration (for material-based stealth systems)
    material_config: Optional[StealthMaterialConfig] = None
    
    # Environmental constraints
    temperature_range_c: Dict[str, float] = field(default_factory=lambda: {"min": -40.0, "max": 85.0})
    humidity_range_percent: Dict[str, float] = field(default_factory=lambda: {"min": 0.0, "max": 100.0})
    altitude_range_m: Dict[str, float] = field(default_factory=lambda: {"min": 0.0, "max": 20000.0})
    
    # Advanced configuration
    neuromorphic_enabled: bool = True
    adaptive_parameters: Dict[str, Any] = field(default_factory=dict)
    integration_points: Dict[str, List[str]] = field(default_factory=dict)


class StealthConfigTemplates:
    """Predefined stealth configuration templates."""
    
    @staticmethod
    def get_template_list() -> Dict[str, List[str]]:
        """
        Get list of available templates by stealth type.
        
        Returns:
            Dict[str, List[str]]: Stealth types and their templates
        """
        return {
            "radar_absorbent_material": ["standard", "high_performance", "lightweight"],
            "plasma_stealth": ["standard", "high_power", "extended_duration"],
            "active_camouflage": ["standard", "rapid_adaptation", "multi_spectrum"],
            "metamaterial_cloaking": ["standard", "broadband", "directional"],
            "infrared_suppression": ["standard", "complete", "adaptive"],
            "acoustic_reduction": ["standard", "deep_spectrum", "broadband"],
            "electromagnetic_shielding": ["standard", "high_intensity", "selective"],
            "shape_shifting": ["standard", "rapid_morph", "multi_form"],
            "thermal_camouflage": ["standard", "adaptive", "complete_masking"],
            "low_observable_nozzle": ["standard", "high_performance", "multi_mode"]
        }
    
    @staticmethod
    def get_template(stealth_type: str, template_name: str) -> Optional[StealthSystemConfig]:
        """
        Get configuration template.
        
        Args:
            stealth_type: Stealth type
            template_name: Template name
            
        Returns:
            Optional[StealthSystemConfig]: Template configuration or None if not found
        """
        # Convert string to enum
        try:
            stealth_type_enum = StealthType[stealth_type.upper()]
        except KeyError:
            return None
            
        # Define templates
        templates = {
            # RAM templates
            "radar_absorbent_material": {
                "standard": StealthSystemConfig(
                    stealth_type=StealthType.RADAR_ABSORBENT_MATERIAL,
                    name="Standard RAM",
                    description="Standard radar absorbent material configuration",
                    effectiveness_level=StealthEffectivenessLevel.MEDIUM,
                    weight_kg=120.0,
                    power_requirements_kw=0.0,  # Passive system
                    signature_config=StealthSignatureConfig(
                        radar_cross_section_reduction=80.0,
                        infrared_signature_reduction=10.0,
                        acoustic_signature_reduction=5.0,
                        visual_signature_reduction=0.0,
                        electromagnetic_signature_reduction=60.0
                    ),
                    material_config=StealthMaterialConfig(
                        material_type="ferrite_composite",
                        thickness_mm=3.0,
                        coverage_percentage=95.0
                    )
                ),
                "high_performance": StealthSystemConfig(
                    stealth_type=StealthType.RADAR_ABSORBENT_MATERIAL,
                    name="High Performance RAM",
                    description="Advanced radar absorbent material with superior performance",
                    effectiveness_level=StealthEffectivenessLevel.VERY_HIGH,
                    weight_kg=180.0,
                    power_requirements_kw=0.0,  # Passive system
                    signature_config=StealthSignatureConfig(
                        radar_cross_section_reduction=95.0,
                        infrared_signature_reduction=15.0,
                        acoustic_signature_reduction=5.0,
                        visual_signature_reduction=0.0,
                        electromagnetic_signature_reduction=85.0
                    ),
                    material_config=StealthMaterialConfig(
                        material_type="advanced_composite",
                        thickness_mm=5.0,
                        coverage_percentage=98.0,
                        frequency_range_ghz={"min": 0.1, "max": 40.0}
                    )
                )
            },
            
            # Plasma stealth templates
            "plasma_stealth": {
                "standard": StealthSystemConfig(
                    stealth_type=StealthType.PLASMA_STEALTH,
                    name="Standard Plasma Stealth",
                    description="Standard plasma stealth system",
                    effectiveness_level=StealthEffectivenessLevel.HIGH,
                    power_mode=StealthPowerMode.PERFORMANCE,
                    operational_mode=StealthOperationalMode.ACTIVE,
                    weight_kg=250.0,
                    power_requirements_kw=75.0,
                    activation_time_seconds=3.0,
                    cooldown_time_seconds=60.0,
                    operational_duration_minutes=30.0,
                    signature_config=StealthSignatureConfig(
                        radar_cross_section_reduction=90.0,
                        infrared_signature_reduction=30.0,
                        acoustic_signature_reduction=0.0,
                        visual_signature_reduction=0.0,
                        electromagnetic_signature_reduction=95.0
                    )
                )
            }
        }
        
        # Return the requested template
        if stealth_type in templates and template_name in templates[stealth_type]:
            return templates[stealth_type][template_name]
        
        return None
#!/usr/bin/env python3
"""
Radar Absorbent Material (RAM) class definition.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

@dataclass
class RAMMaterial:
    """Radar Absorbent Material properties."""
    name: str
    density: float  # kg/m³
    thickness: float  # mm
    frequency_response: Dict[str, float]  # Frequency (GHz) -> Attenuation (dB)
    temperature_range: Tuple[float, float]  # Operating temperature range (°C)
    weather_resistance: float  # 0-1 scale
    durability: float  # 0-1 scale
    cost_factor: float  # Relative cost factor
    material_type: str
    thickness_mm: float
    frequency_range_ghz: Dict[str, float]
    conductivity: Optional[float] = None
    permittivity: Optional[float] = None
    absorption_rate: float = 0.0  # Percentage of absorbed radiation
    reflection_coefficient: float = 0.0
    thermal_properties: Dict[str, float] = field(default_factory=dict)
    weight_per_area_kg_m2: float = 0.0
    cost_per_area: float = 0.0
    durability_rating: float = 0.0  # 0-1 scale
    environmental_resistance: Dict[str, float] = field(default_factory=dict)
    manufacturing_complexity: float = 0.0  # 0-1 scale
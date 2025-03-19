"""
Radar-Absorbent Materials (RAM) module for stealth systems.
"""

from src.stealth.materials.ram.ram_system import RAMSystem, RAMMaterial
from src.stealth.materials.ram.material_database import RAMMaterialDatabase
from src.stealth.materials.ram.ram_simulation import RAMSimulator
from src.stealth.materials.ram.manufacturing import RAMManufacturingGenerator, RAMManufacturingSpec

__all__ = [
    'RAMSystem',
    'RAMMaterial',
    'RAMMaterialDatabase',
    'RAMSimulator',
    'RAMManufacturingGenerator',
    'RAMManufacturingSpec'
]
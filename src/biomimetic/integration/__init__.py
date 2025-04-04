"""
Biomimetic System Integration Framework.

Provides cross-module validation and configuration management for biomimetic designs.
"""

from src.biomimetic.integration.framework import BiomimeticIntegrationFramework
from src.biomimetic.integration.validator import BiomimeticValidator
from src.biomimetic.integration.config_manager import BiomimeticConfigManager

__all__ = [
    'BiomimeticIntegrationFramework',
    'BiomimeticValidator',
    'BiomimeticConfigManager'
]
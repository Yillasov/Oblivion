#!/usr/bin/env python3
"""
Base hardware optimization module.

Provides the foundation for hardware-specific optimizations.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class HardwareOptimizer(ABC):
    """Base class for hardware-specific optimizations."""
    
    @abstractmethod
    def optimize_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a neural network configuration for specific hardware."""
        pass
    
    @abstractmethod
    def optimize_resource_allocation(self, resource_request: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation for specific hardware."""
        pass
    
    @abstractmethod
    def get_optimization_recommendations(self) -> List[str]:
        """Get hardware-specific optimization recommendations."""
        pass


class OptimizationRegistry:
    """Registry for hardware optimizers."""
    
    _optimizers: Dict[str, HardwareOptimizer] = {}
    
    @classmethod
    def register(cls, hardware_type: str, optimizer: HardwareOptimizer) -> None:
        """Register an optimizer for a hardware type."""
        cls._optimizers[hardware_type.lower()] = optimizer
    
    @classmethod
    def get_optimizer(cls, hardware_type: str) -> Optional[HardwareOptimizer]:
        """Get the optimizer for a hardware type."""
        return cls._optimizers.get(hardware_type.lower())
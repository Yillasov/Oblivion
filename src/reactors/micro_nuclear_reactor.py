"""
Micro Nuclear Reactor system.

This module implements basic functionality for micro nuclear reactors,
including power generation, safety checks, and efficiency optimization.
"""

from typing import Dict, Any
import time
import math

class MicroNuclearReactor:
    """Micro nuclear reactor system."""
    
    def __init__(self, reactor_id: str, max_power: float = 50.0, efficiency: float = 0.9):
        """
        Initialize micro nuclear reactor.
        
        Args:
            reactor_id: Unique identifier
            max_power: Maximum power output in kW
            efficiency: Operational efficiency (0-1)
        """
        self.reactor_id = reactor_id
        self.max_power = max_power
        self.efficiency = efficiency
        self.current_power = 0.0
        self.active = False
        self.status = {
            "temperature": 300.0,  # K
            "radiation_level": 0.01,  # mSv/h
            "health": 1.0,
            "last_error": None
        }
        
    def activate(self) -> bool:
        """Activate the reactor."""
        if self.active:
            return True
            
        self.active = True
        self.current_power = self.max_power * self.efficiency
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the reactor."""
        if not self.active:
            return True
            
        self.active = False
        self.current_power = 0.0
        return True
    
    def check_safety(self) -> bool:
        """
        Perform safety checks.
        
        Returns:
            Safety status
        """
        if self.status["temperature"] > 350.0 or self.status["radiation_level"] > 0.05:
            self.status["health"] = 0.5
            return False
        
        self.status["health"] = 1.0
        return True
    
    def optimize_efficiency(self) -> float:
        """
        Optimize reactor efficiency.
        
        Returns:
            New efficiency value
        """
        if self.status["temperature"] < 300.0:
            self.efficiency = min(1.0, self.efficiency + 0.05)
        elif self.status["temperature"] > 350.0:
            self.efficiency = max(0.8, self.efficiency - 0.05)
        
        return self.efficiency
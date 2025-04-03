"""
Power distribution system for UCAV platforms.

This module provides classes and utilities for distributing power
across different systems and optimizing power usage.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import numpy as np

from src.power.resource_management import PowerResourceManager, PowerPriority
from src.power.base import PowerSupplyType, NeuromorphicPowerSupply
from src.core.utils.logging_framework import get_logger

logger = get_logger("power_distribution")


@dataclass
class PowerDistributionConfig:
    """Configuration for power distribution."""
    max_power_draw: float = 1000.0  # Maximum power draw in kW
    reserve_percentage: float = 10.0  # Power reserve percentage
    efficiency_threshold: float = 0.8  # Minimum efficiency threshold
    temperature_threshold: float = 70.0  # Maximum temperature threshold in Celsius
    load_balancing: bool = True  # Enable load balancing
    dynamic_allocation: bool = True  # Enable dynamic allocation
    priority_override: bool = True  # Enable priority override for critical systems


class PowerDistributor:
    """Distributes power across UCAV systems."""
    
    def __init__(self, 
                resource_manager: PowerResourceManager,
                config: Optional[PowerDistributionConfig] = None):
        """
        Initialize power distributor.
        
        Args:
            resource_manager: Power resource manager
            config: Power distribution configuration
        """
        self.resource_manager = resource_manager
        self.config = config or PowerDistributionConfig()
        self.distribution_map: Dict[str, Dict[str, float]] = {}
        self.system_priorities: Dict[str, PowerPriority] = {}
        self.last_distribution_time = 0.0
        self.distribution_history: List[Dict[str, Any]] = []
        
        logger.info("Power distributor initialized")
    
    def register_system(self, 
                       system_id: str, 
                       power_requirements: Dict[str, float],
                       priority: PowerPriority = PowerPriority.MEDIUM) -> bool:
        """
        Register a system for power distribution.
        
        Args:
            system_id: System identifier
            power_requirements: Power requirements by resource ID
            priority: System priority
            
        Returns:
            Success status
        """
        # Register with resource manager
        if not self.resource_manager.register_power_consumer(system_id, power_requirements):
            return False
        
        # Store system priority
        self.system_priorities[system_id] = priority
        
        logger.info(f"Registered system '{system_id}' for power distribution with priority {priority}")
        return True
    
    def distribute_power(self, flight_conditions: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Distribute power based on current conditions.
        
        Args:
            flight_conditions: Current flight conditions
            
        Returns:
            Power distribution map
        """
        current_time = time.time()
        
        try:
            # Adjust resource manager distribution strategy based on conditions
            if flight_conditions:
                self._adjust_distribution_strategy(flight_conditions)
            
            # Allocate power using resource manager
            self.distribution_map = self.resource_manager.allocate_power()
            
            # Apply any additional distribution logic
            if self.config.load_balancing:
                self._balance_load()
            
            # Record distribution
            self._record_distribution(current_time, flight_conditions)
            
            self.last_distribution_time = current_time
            
            logger.info("Power distribution completed successfully")
            return self.distribution_map
            
        except Exception as e:
            logger.error(f"Error distributing power: {str(e)}")
            return {}
    
    def _adjust_distribution_strategy(self, flight_conditions: Dict[str, float]) -> None:
        """
        Adjust distribution strategy based on flight conditions.
        
        Args:
            flight_conditions: Current flight conditions
        """
        # Example condition-based strategy selection
        if "emergency" in flight_conditions and flight_conditions["emergency"]:
            # Use priority-based in emergencies
            self.resource_manager.distribution_strategy = "priority_based"
            logger.info("Switched to priority-based distribution due to emergency")
            
        elif "altitude" in flight_conditions and flight_conditions["altitude"] > 10000:
            # At high altitude, efficiency becomes more important
            self.resource_manager.distribution_strategy = "efficiency_based"
            logger.info("Switched to efficiency-based distribution at high altitude")
            
        elif "stealth_mode" in flight_conditions and flight_conditions["stealth_mode"]:
            # In stealth mode, minimize power usage
            self.resource_manager.distribution_strategy = "efficiency_based"
            logger.info("Switched to efficiency-based distribution in stealth mode")
            
        else:
            # Default to proportional distribution
            self.resource_manager.distribution_strategy = "proportional"
    
    def _balance_load(self) -> None:
        """Balance load across power resources."""
        # Implementation simplified for brevity
        pass
    
    def _record_distribution(self, 
                           timestamp: float, 
                           flight_conditions: Optional[Dict[str, float]] = None) -> None:
        """
        Record power distribution.
        
        Args:
            timestamp: Current timestamp
            flight_conditions: Current flight conditions
        """
        # Create distribution record
        record = {
            "timestamp": timestamp,
            "strategy": self.resource_manager.distribution_strategy,
            "distribution": {
                system_id: {
                    "total": sum(allocations.values()),
                    "by_resource": allocations,
                    "priority": self.system_priorities.get(system_id, PowerPriority.MEDIUM).name
                }
                for system_id, allocations in self.distribution_map.items()
            },
            "conditions": flight_conditions or {}
        }
        
        # Add to history
        self.distribution_history.append(record)
        
        # Limit history size
        if len(self.distribution_history) > 1000:
            self.distribution_history = self.distribution_history[-1000:]
    
    def get_system_power(self, system_id: str) -> Dict[str, Any]:
        """
        Get power allocation for a specific system.
        
        Args:
            system_id: System identifier
            
        Returns:
            Power allocation information
        """
        if system_id not in self.distribution_map:
            return {"error": "system_not_found"}
        
        allocations = self.distribution_map[system_id]
        total_allocation = sum(allocations.values())
        
        return {
            "total_allocation": total_allocation,
            "by_resource": allocations,
            "priority": self.system_priorities.get(system_id, PowerPriority.MEDIUM).name
        }
    
    def optimize_distribution(self) -> Dict[str, Any]:
        """
        Optimize power distribution.
        
        Returns:
            Optimization results
        """
        # Use resource manager's optimization
        return self.resource_manager.optimize_power_usage()
    
    def predict_distribution(self, time_horizon: float = 60.0) -> Dict[str, Any]:
        """
        Predict future power distribution.
        
        Args:
            time_horizon: Time horizon in seconds
            
        Returns:
            Prediction results
        """
        # Use resource manager's prediction
        return self.resource_manager.predict_power_needs(time_horizon)
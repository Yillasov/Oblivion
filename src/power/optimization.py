"""
Power supply selection and optimization algorithms.

This module provides algorithms for selecting optimal power sources
and optimizing power distribution across systems.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import math

from src.power.resource_management import PowerPriority, PowerResource
from src.core.utils.logging_framework import get_logger

logger = get_logger("power_optimization")


class PowerOptimizationAlgorithm:
    """Power supply selection and optimization algorithms."""
    
    @staticmethod
    def select_optimal_power_sources(
        available_sources: Dict[str, PowerResource],
        power_demand: float,
        optimization_criteria: str = "efficiency"
    ) -> Dict[str, float]:
        """
        Select optimal power sources based on criteria.
        
        Args:
            available_sources: Available power sources
            power_demand: Total power demand in kW
            optimization_criteria: Criteria for optimization (efficiency, endurance, performance)
            
        Returns:
            Optimal source allocation {source_id: allocation_kw}
        """
        if not available_sources:
            return {}
            
        # Sort sources based on optimization criteria
        if optimization_criteria == "efficiency":
            sorted_sources = sorted(
                available_sources.items(),
                key=lambda x: (x[1].priority.value, x[1].efficiency),
                reverse=True
            )
        elif optimization_criteria == "endurance":
            sorted_sources = sorted(
                available_sources.items(),
                key=lambda x: (
                    1 if x[1].type.name in ["SOLAR", "FUEL_CELL"] else 0,
                    x[1].priority.value
                ),
                reverse=True
            )
        else:  # performance
            sorted_sources = sorted(
                available_sources.items(),
                key=lambda x: (x[1].priority.value, x[1].max_output),
                reverse=True
            )
            
        # Allocate power based on sorted order
        allocation = {}
        remaining_demand = power_demand
        
        for source_id, source in sorted_sources:
            if remaining_demand <= 0:
                break
                
            # Calculate how much this source can provide
            # Use max_output and current_output to determine available power
            available = min(source.max_output - source.current_output, remaining_demand)
            
            if available > 0:
                allocation[source_id] = available
                remaining_demand -= available
                
        return allocation
    
    @staticmethod
    def optimize_power_distribution(
        consumers: Dict[str, Dict[str, float]],
        available_power: float,
        priorities: Dict[str, PowerPriority]
    ) -> Dict[str, float]:
        """
        Optimize power distribution across consumers.
        
        Args:
            consumers: Consumer power requirements {consumer_id: {resource_type: requirement}}
            available_power: Total available power in kW
            priorities: Consumer priorities {consumer_id: priority}
            
        Returns:
            Optimized allocation {consumer_id: allocation}
        """
        # Calculate total requirements and normalize by priority
        total_required = 0
        weighted_requirements = {}
        
        for consumer_id, requirements in consumers.items():
            req_sum = sum(requirements.values())
            total_required += req_sum
            
            # Get priority weight (higher priority = higher weight)
            priority = priorities.get(consumer_id, PowerPriority.MEDIUM)
            priority_weight = 4.0 if priority == PowerPriority.CRITICAL else \
                             2.0 if priority == PowerPriority.HIGH else \
                             1.0 if priority == PowerPriority.MEDIUM else \
                             0.5 if priority == PowerPriority.LOW else 0.25
                             
            weighted_requirements[consumer_id] = req_sum * priority_weight
            
        # If we have enough power, give everyone what they need
        if total_required <= available_power:
            return {consumer_id: sum(reqs.values()) for consumer_id, reqs in consumers.items()}
            
        # Otherwise, distribute based on weighted requirements
        total_weighted = sum(weighted_requirements.values())
        allocation = {}
        
        for consumer_id, weighted_req in weighted_requirements.items():
            # Allocate proportionally to weighted requirement
            allocation_ratio = weighted_req / total_weighted
            allocation[consumer_id] = allocation_ratio * available_power
            
        return allocation
    
    @staticmethod
    def calculate_optimal_power_mode(
        current_power: float,
        target_duration: float,
        available_energy: float,
        power_modes: Dict[str, float]
    ) -> str:
        """
        Calculate optimal power mode to achieve target duration.
        
        Args:
            current_power: Current power consumption in kW
            target_duration: Target operation duration in minutes
            available_energy: Available energy in kWh
            power_modes: Available power modes {mode_name: power_kw}
            
        Returns:
            Optimal power mode name
        """
        # Calculate energy needed for target duration
        energy_needed = current_power * (target_duration / 60.0)  # kWh
        
        if energy_needed <= available_energy:
            # Current mode is fine
            for mode, power in power_modes.items():
                if math.isclose(power, current_power, rel_tol=0.05):
                    return mode
        
        # Find mode that allows us to reach target duration
        for mode_name, mode_power in sorted(power_modes.items(), key=lambda x: x[1]):
            mode_energy_needed = mode_power * (target_duration / 60.0)
            if mode_energy_needed <= available_energy:
                return mode_name
                
        # If no mode works, return the lowest power mode
        return min(power_modes.items(), key=lambda x: x[1])[0]
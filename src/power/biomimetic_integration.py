"""
Biomimetic power integration framework for UCAV platforms.

This module extends the standard power integration with biomimetic-specific
capabilities such as adaptive power distribution and energy harvesting.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.power.integration import PowerIntegrator
from src.power.base import NeuromorphicPowerSupply, PowerSupplyType
from src.biomimetic.design.principles import BiomimeticPrinciple
from src.core.utils.logging_framework import get_logger

logger = get_logger("biomimetic_power")

class BiomimeticPowerIntegrator(PowerIntegrator):
    """Enhanced power integrator with biomimetic capabilities."""
    
    def __init__(self, hardware_interface=None, config=None, biological_reference: str = "peregrine_falcon"):
        """
        Initialize the biomimetic power integrator.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
            config: Configuration for power integration
            biological_reference: Biological reference model
        """
        super().__init__(hardware_interface, config)
        self.biological_reference = biological_reference
        self.biomimetic_systems: Dict[str, Dict[str, Any]] = {}
        self.energy_harvesting_enabled = False
        self.adaptive_distribution_enabled = True
        
        # Biomimetic power states
        self.power_states = {
            "high_performance": {"scaling": 1.0, "priority_systems": ["propulsion", "navigation"]},
            "balanced": {"scaling": 0.8, "priority_systems": ["propulsion", "sensors"]},
            "efficiency": {"scaling": 0.6, "priority_systems": ["sensors"]},
            "stealth": {"scaling": 0.4, "priority_systems": []},
            "emergency": {"scaling": 0.3, "priority_systems": ["critical_systems"]}
        }
        
        self.current_power_state = "balanced"
        logger.info(f"Initialized biomimetic power integrator with reference: {biological_reference}")
    
    def register_biomimetic_system(self, system_id: str, 
                                  power_requirements: Dict[str, float],
                                  adaptability_factor: float = 0.5) -> bool:
        """
        Register a biomimetic system with the power integrator.
        
        Args:
            system_id: System identifier
            power_requirements: Power requirements by mode
            adaptability_factor: How adaptable the system is to power changes (0-1)
            
        Returns:
            Success status
        """
        if system_id in self.biomimetic_systems:
            logger.warning(f"Biomimetic system '{system_id}' already registered")
            return False
        
        self.biomimetic_systems[system_id] = {
            "power_requirements": power_requirements,
            "adaptability_factor": adaptability_factor,
            "current_allocation": 0.0,
            "priority": 0.5
        }
        
        logger.info(f"Registered biomimetic system: {system_id}")
        return True
    
    def set_power_state(self, state: str) -> bool:
        """
        Set the current power state for biomimetic systems.
        
        Args:
            state: Power state name
            
        Returns:
            Success status
        """
        if state not in self.power_states:
            logger.error(f"Unknown power state: {state}")
            return False
        
        self.current_power_state = state
        
        # Apply power state to all systems
        self._redistribute_power()
        
        logger.info(f"Set biomimetic power state to: {state}")
        return True
    
    def enable_energy_harvesting(self, enabled: bool = True) -> None:
        """Enable or disable energy harvesting capabilities."""
        self.energy_harvesting_enabled = enabled
        
        # Configure power systems for energy harvesting
        for system_id, system in self.power_systems.items():
            if hasattr(system, 'set_harvesting_mode'):
                system.set_harvesting_mode(enabled)
        
        logger.info(f"Energy harvesting {'enabled' if enabled else 'disabled'}")
    
    def _redistribute_power(self) -> None:
        """Redistribute power based on current state and biomimetic principles."""
        if not self.biomimetic_systems:
            return
            
        state_config = self.power_states[self.current_power_state]
        scaling = state_config["scaling"]
        priority_systems = state_config["priority_systems"]
        
        # Calculate total available power
        total_available = sum(
            system.get_specifications().max_output * scaling
            for system in self.power_systems.values()
            if system.get_specifications().supply_type != PowerSupplyType.BACKUP
        )
        
        # Calculate total required power
        total_required = sum(
            system["power_requirements"].get("nominal", 0.0)
            for system in self.biomimetic_systems.values()
        )
        
        # Adjust allocations based on priority and adaptability
        for system_id, system_data in self.biomimetic_systems.items():
            base_requirement = system_data["power_requirements"].get("nominal", 0.0)
            
            # Determine priority factor
            priority_factor = 1.0
            if system_id in priority_systems:
                priority_factor = 1.5
            elif total_required > total_available:
                # Reduce non-priority systems when power is limited
                priority_factor = 0.7
            
            # Calculate allocation with adaptability factor
            adaptability = system_data["adaptability_factor"]
            if total_required > total_available:
                # Power is limited, adapt based on adaptability factor
                allocation_ratio = min(1.0, total_available / total_required)
                allocation = base_requirement * allocation_ratio * priority_factor
                
                # More adaptable systems can reduce power further if needed
                if allocation_ratio < 0.8:
                    adaptation_factor = 1.0 - ((1.0 - allocation_ratio) * adaptability)
                    allocation *= adaptation_factor
            else:
                # Sufficient power available
                allocation = base_requirement * priority_factor
            
            # Update allocation
            system_data["current_allocation"] = allocation
            
            # Apply to connected systems through neuromorphic system
            self.neuromorphic_system.process_data({
                "command": "set_power_allocation",
                "system_id": system_id,
                "allocation": allocation
            })
        
        logger.debug(f"Redistributed power for state: {self.current_power_state}")
    
    def update(self, dt: float) -> None:
        """
        Update power integration with time step.
        
        Args:
            dt: Time step in seconds
        """
        super().update(dt)
        
        # Handle energy harvesting if enabled
        if self.energy_harvesting_enabled:
            self._process_energy_harvesting(dt)
        
        # Adaptive power distribution based on system demands
        if self.adaptive_distribution_enabled:
            self._adapt_to_system_demands()
    
    def _process_energy_harvesting(self, dt: float) -> None:
        """Process energy harvesting from environment."""
        # Simplified energy harvesting simulation
        harvested_energy = 0.0
        
        # Get environmental data from neuromorphic system
        env_data = self.neuromorphic_system.process_data({"query": "environment"})
        
        if env_data and "solar_intensity" in env_data:
            # Solar harvesting
            solar_intensity = env_data["solar_intensity"]
            harvested_energy += solar_intensity * 0.05 * dt  # 5% efficiency
        
        if env_data and "airflow_energy" in env_data:
            # Airflow energy harvesting (biomimetic)
            airflow_energy = env_data["airflow_energy"]
            harvested_energy += airflow_energy * 0.02 * dt  # 2% efficiency
        
        # Distribute harvested energy to power systems
        if harvested_energy > 0:
            for system_id, system in self.power_systems.items():
                if hasattr(system, 'add_harvested_energy'):
                    system.add_harvested_energy(harvested_energy / len(self.power_systems))
    
    def _adapt_to_system_demands(self) -> None:
        """Adapt power distribution based on real-time system demands."""
        # Get current system demands
        system_demands = {}
        for system_id in self.biomimetic_systems:
            # Query current power demand from the system
            demand_data = self.neuromorphic_system.process_data({
                "query": "power_demand",
                "system_id": system_id
            })
            
            if demand_data and "current_demand" in demand_data:
                system_demands[system_id] = demand_data["current_demand"]
        
        # Check if redistribution is needed
        if system_demands:
            needs_redistribution = False
            for system_id, demand in system_demands.items():
                current = self.biomimetic_systems[system_id]["current_allocation"]
                # If demand is significantly different from current allocation
                if abs(demand - current) / current > 0.2:  # 20% threshold
                    needs_redistribution = True
                    break
            
            if needs_redistribution:
                self._redistribute_power()
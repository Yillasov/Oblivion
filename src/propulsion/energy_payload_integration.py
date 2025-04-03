"""
Integration system for connecting energy harvesting with payload systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.propulsion.energy_harvesting import EnergyHarvestingSystem, HarvesterSpecs, HarvestingType
from src.payload.base import PayloadInterface
from src.payload.integration import PayloadIntegrator


@dataclass
class EnergyPayloadConfig:
    """Configuration for energy-payload integration."""
    min_power_threshold: float  # Minimum power required to activate integration (kW)
    max_power_transfer: float  # Maximum power that can be transferred to payloads (kW)
    priority_payloads: List[str]  # List of payload IDs that get priority for harvested energy
    efficiency: float  # Energy transfer efficiency (0-1)
    storage_capacity: float  # Energy storage capacity in kWh


class PowerDistributionMode(Enum):
    """Power distribution modes for harvested energy."""
    DISABLED = 0
    PRIORITY_ONLY = 1
    PROPORTIONAL = 2
    EQUAL = 3
    DEMAND_BASED = 4


class EnergyPayloadIntegrator:
    """System for integrating energy harvesting with payload systems."""
    
    def __init__(self, 
                 harvesting_system: EnergyHarvestingSystem,
                 payload_integrator: Optional[PayloadIntegrator] = None,
                 config: Optional[EnergyPayloadConfig] = None):
        """Initialize energy-payload integration system."""
        self.harvesting_system = harvesting_system
        self.payload_integrator = payload_integrator
        self.config = config or EnergyPayloadConfig(
            min_power_threshold=0.1,  # 100W
            max_power_transfer=5.0,  # 5kW
            priority_payloads=[],
            efficiency=0.85,
            storage_capacity=10.0  # 10kWh
        )
        self.payloads: Dict[str, PayloadInterface] = {}
        self.payload_power_requirements: Dict[str, float] = {}
        self.distribution_mode = PowerDistributionMode.DISABLED
        self.stored_energy = 0.0  # kWh
        self.integration_active = False
        self.distribution_history: List[Dict[str, Any]] = []
        
    def register_payload(self, payload_id: str, payload: PayloadInterface, 
                        power_requirement: float) -> bool:
        """
        Register a payload for energy integration.
        
        Args:
            payload_id: Unique identifier for the payload
            payload: Payload interface
            power_requirement: Power requirement in kW
            
        Returns:
            Success status
        """
        if payload_id in self.payloads:
            return False
            
        self.payloads[payload_id] = payload
        self.payload_power_requirements[payload_id] = power_requirement
        
        # Add to priority list if it's a low-power payload
        if power_requirement < 0.5:  # Less than 500W
            self.config.priority_payloads.append(payload_id)
            
        return True
        
    def set_distribution_mode(self, mode: PowerDistributionMode) -> None:
        """Set the power distribution mode."""
        self.distribution_mode = mode
        
    def activate_integration(self) -> bool:
        """Activate energy-payload integration."""
        if not self.payloads or not self.harvesting_system.harvesters:
            return False
            
        self.integration_active = True
        return True
        
    def deactivate_integration(self) -> None:
        """Deactivate energy-payload integration."""
        self.integration_active = False
        
    def update(self, env_condition: Any, dt: float) -> Dict[str, Any]:
        """
        Update energy-payload integration.
        
        Args:
            env_condition: Environmental conditions
            dt: Time step in seconds
            
        Returns:
            Integration status
        """
        if not self.integration_active:
            return {"status": "inactive"}
            
        # Get current harvested energy
        harvester_outputs = self.harvesting_system.update(env_condition, dt)
        total_harvested = self.harvesting_system.get_total_output()  # kW
        
        # Add to storage
        energy_to_store = total_harvested * dt / 3600.0  # Convert kW to kWh
        self.stored_energy += energy_to_store * self.config.efficiency
        self.stored_energy = min(self.stored_energy, self.config.storage_capacity)
        
        # Check if we have enough energy to distribute
        available_power = total_harvested  # kW
        if available_power < self.config.min_power_threshold:
            # Not enough power being generated, use stored energy if available
            if self.stored_energy > 0:
                # Convert some stored energy to power
                available_power = min(self.config.max_power_transfer, 
                                     self.stored_energy * 0.1)  # Use up to 10% of stored energy
                self.stored_energy -= available_power * dt / 3600.0
            else:
                return {
                    "status": "insufficient_power",
                    "harvested_power": total_harvested,
                    "stored_energy": self.stored_energy
                }
        
        # Distribute power to payloads
        distribution = self._distribute_power(available_power)
        
        # Record distribution
        self.distribution_history.append({
            "timestamp": np.datetime64('now'),
            "harvested_power": total_harvested,
            "distributed_power": sum(distribution.values()),
            "stored_energy": self.stored_energy,
            "distribution": distribution.copy()
        })
        
        # Limit history size
        if len(self.distribution_history) > 100:
            self.distribution_history.pop(0)
            
        return {
            "status": "active",
            "harvested_power": total_harvested,
            "distributed_power": sum(distribution.values()),
            "stored_energy": self.stored_energy,
            "distribution": distribution,
            "harvester_outputs": harvester_outputs
        }
        
    def _distribute_power(self, available_power: float) -> Dict[str, float]:
        """
        Distribute available power to payloads based on current mode.
        
        Args:
            available_power: Available power in kW
            
        Returns:
            Dictionary mapping payload IDs to allocated power
        """
        distribution = {pid: 0.0 for pid in self.payloads}
        
        if self.distribution_mode == PowerDistributionMode.DISABLED:
            return distribution
            
        # Limit available power to maximum transfer
        available_power = min(available_power, self.config.max_power_transfer)
        
        if self.distribution_mode == PowerDistributionMode.PRIORITY_ONLY:
            # Distribute only to priority payloads
            priority_payloads = {pid: self.payload_power_requirements[pid] 
                               for pid in self.config.priority_payloads 
                               if pid in self.payloads}
            
            total_required = sum(priority_payloads.values())
            if total_required <= available_power:
                # We can power all priority payloads
                distribution.update(priority_payloads)
            else:
                # Not enough power, distribute proportionally
                factor = available_power / total_required
                for pid, req in priority_payloads.items():
                    distribution[pid] = req * factor
                    
        elif self.distribution_mode == PowerDistributionMode.PROPORTIONAL:
            # Distribute proportionally to power requirements
            total_required = sum(self.payload_power_requirements.values())
            if total_required <= available_power:
                # We can power all payloads
                distribution.update(self.payload_power_requirements)
            else:
                # Not enough power, distribute proportionally
                factor = available_power / total_required
                for pid, req in self.payload_power_requirements.items():
                    distribution[pid] = req * factor
                    
        elif self.distribution_mode == PowerDistributionMode.EQUAL:
            # Distribute equally among all payloads
            per_payload = available_power / len(self.payloads)
            for pid in self.payloads:
                distribution[pid] = min(per_payload, self.payload_power_requirements[pid])
                
        elif self.distribution_mode == PowerDistributionMode.DEMAND_BASED:
            # Simplified demand-based distribution
            # In a real system, this would use actual current demand from payloads
            active_payloads = {pid: self.payload_power_requirements[pid] 
                             for pid in self.payloads}
            
            total_required = sum(active_payloads.values())
            if total_required <= available_power:
                # We can power all active payloads
                distribution.update(active_payloads)
            else:
                # Not enough power, distribute to highest priority first
                remaining = available_power
                for pid in self.config.priority_payloads:
                    if pid in active_payloads and remaining > 0:
                        req = active_payloads[pid]
                        allocated = min(req, remaining)
                        distribution[pid] = allocated
                        remaining -= allocated
                
                # Distribute remaining power proportionally to non-priority payloads
                non_priority = {pid: req for pid, req in active_payloads.items() 
                              if pid not in self.config.priority_payloads}
                if non_priority and remaining > 0:
                    total_non_priority = sum(non_priority.values())
                    factor = remaining / total_non_priority
                    for pid, req in non_priority.items():
                        distribution[pid] = req * factor
                        
        return distribution
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current status of energy-payload integration."""
        return {
            "active": self.integration_active,
            "distribution_mode": self.distribution_mode.name,
            "stored_energy": self.stored_energy,
            "storage_capacity": self.config.storage_capacity,
            "registered_payloads": len(self.payloads),
            "priority_payloads": self.config.priority_payloads,
            "last_distribution": self.distribution_history[-1] if self.distribution_history else None
        }
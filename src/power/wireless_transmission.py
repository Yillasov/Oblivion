"""
Wireless Power Transmission system.

This module implements wireless power transmission capabilities for the UCAV platform,
enabling power transfer between systems without physical connections.
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum, auto
import time
import math

from src.core.utils.logging_framework import get_logger
from src.power.resource_management import PowerPriority, PowerResource, PowerSupplyType
from src.power.power_requirements import PowerRequirementCalculator

logger = get_logger("wireless_power")


class WirelessTransmissionMode(Enum):
    """Wireless power transmission modes."""
    MICROWAVE = auto()  # Microwave-based transmission
    LASER = auto()      # Laser-based transmission
    RESONANT = auto()   # Resonant inductive coupling
    CAPACITIVE = auto() # Capacitive coupling


class WirelessPowerTransmitter:
    """Wireless power transmitter system."""
    
    def __init__(self, 
                 transmitter_id: str,
                 max_power: float = 10.0,  # kW
                 efficiency: float = 0.75,
                 mode: WirelessTransmissionMode = WirelessTransmissionMode.MICROWAVE):
        """
        Initialize wireless power transmitter.
        
        Args:
            transmitter_id: Unique identifier
            max_power: Maximum power output in kW
            efficiency: Transmission efficiency (0-1)
            mode: Transmission mode
        """
        self.transmitter_id = transmitter_id
        self.max_power = max_power
        self.efficiency = efficiency
        self.mode = mode
        self.current_power = 0.0
        self.active = False
        self.target_receivers: Dict[str, float] = {}  # receiver_id: power_allocation
        self.status = {
            "temperature": 25.0,
            "health": 1.0,
            "last_error": None
        }
        
        logger.info(f"Wireless power transmitter '{transmitter_id}' initialized with {mode.name} mode")
    
    def activate(self) -> bool:
        """Activate the transmitter."""
        if self.active:
            return True
            
        self.active = True
        logger.info(f"Transmitter '{self.transmitter_id}' activated")
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the transmitter."""
        if not self.active:
            return True
            
        self.active = False
        self.current_power = 0.0
        self.target_receivers = {}
        logger.info(f"Transmitter '{self.transmitter_id}' deactivated")
        return True
    
    def set_power_level(self, power_level: float) -> bool:
        """
        Set transmitter power level.
        
        Args:
            power_level: Power level in kW
            
        Returns:
            Success status
        """
        if not self.active:
            logger.warning(f"Cannot set power level: transmitter '{self.transmitter_id}' not active")
            return False
            
        self.current_power = min(power_level, self.max_power)
        logger.debug(f"Transmitter '{self.transmitter_id}' power set to {self.current_power} kW")
        return True
    
    def add_receiver(self, receiver_id: str, power_allocation: float) -> bool:
        """
        Add a receiver to target with specified power allocation.
        
        Args:
            receiver_id: Receiver identifier
            power_allocation: Power allocation in kW
            
        Returns:
            Success status
        """
        if not self.active:
            logger.warning(f"Cannot add receiver: transmitter '{self.transmitter_id}' not active")
            return False
            
        self.target_receivers[receiver_id] = power_allocation
        logger.info(f"Added receiver '{receiver_id}' with {power_allocation} kW allocation")
        return True
    
    def remove_receiver(self, receiver_id: str) -> bool:
        """Remove a receiver from targets."""
        if receiver_id in self.target_receivers:
            del self.target_receivers[receiver_id]
            logger.info(f"Removed receiver '{receiver_id}' from transmitter '{self.transmitter_id}'")
            return True
        return False
    
    def get_effective_power(self, distance: float, alignment: float = 1.0) -> Dict[str, float]:
        """
        Calculate effective power at each receiver based on distance and alignment.
        
        Args:
            distance: Distance to receiver in meters
            alignment: Alignment factor (0-1)
            
        Returns:
            Dictionary of receiver_id: effective_power
        """
        if not self.active or not self.target_receivers:
            return {}
            
        effective_power = {}
        total_allocation = sum(self.target_receivers.values())
        
        # Calculate transmission losses based on mode
        for receiver_id, allocation in self.target_receivers.items():
            # Calculate power ratio
            power_ratio = allocation / total_allocation if total_allocation > 0 else 0
            base_power = self.current_power * power_ratio
            
            # Apply mode-specific distance losses
            if self.mode == WirelessTransmissionMode.MICROWAVE:
                # Microwave follows inverse square law
                distance_factor = 1.0 / (1.0 + (distance / 100.0) ** 2)
            elif self.mode == WirelessTransmissionMode.LASER:
                # Laser has less divergence
                distance_factor = 1.0 / (1.0 + (distance / 1000.0) ** 1.5)
            elif self.mode == WirelessTransmissionMode.RESONANT:
                # Resonant coupling works well at short distances
                distance_factor = 1.0 / (1.0 + (distance / 10.0) ** 3)
            else:  # Capacitive
                # Capacitive coupling only works at very short distances
                distance_factor = 1.0 / (1.0 + (distance / 1.0) ** 4)
                
            # Apply alignment factor
            alignment_factor = alignment ** 2
            
            # Calculate effective power
            effective = base_power * self.efficiency * distance_factor * alignment_factor
            effective_power[receiver_id] = max(0.0, effective)
            
        return effective_power


class WirelessPowerReceiver:
    """Wireless power receiver system."""
    
    def __init__(self, 
                 receiver_id: str,
                 max_power: float = 5.0,  # kW
                 efficiency: float = 0.85,
                 compatible_modes: List[WirelessTransmissionMode] = []):
        """
        Initialize wireless power receiver.
        
        Args:
            receiver_id: Unique identifier
            max_power: Maximum power input in kW
            efficiency: Receiver efficiency (0-1)
            compatible_modes: Compatible transmission modes
        """
        self.receiver_id = receiver_id
        self.max_power = max_power
        self.efficiency = efficiency
        self.compatible_modes = compatible_modes or list(WirelessTransmissionMode)
        self.current_power = 0.0
        self.active = False
        self.connected_transmitters: Dict[str, float] = {}  # transmitter_id: received_power
        self.status = {
            "temperature": 25.0,
            "health": 1.0,
            "last_error": None
        }
        
        logger.info(f"Wireless power receiver '{receiver_id}' initialized")
    
    def activate(self) -> bool:
        """Activate the receiver."""
        if self.active:
            return True
            
        self.active = True
        logger.info(f"Receiver '{self.receiver_id}' activated")
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the receiver."""
        if not self.active:
            return True
            
        self.active = False
        self.current_power = 0.0
        self.connected_transmitters = {}
        logger.info(f"Receiver '{self.receiver_id}' deactivated")
        return True
    
    def receive_power(self, transmitter_id: str, power: float, mode: WirelessTransmissionMode) -> float:
        """
        Receive power from a transmitter.
        
        Args:
            transmitter_id: Transmitter identifier
            power: Incoming power in kW
            mode: Transmission mode
            
        Returns:
            Actual power received in kW
        """
        if not self.active:
            return 0.0
            
        if mode not in self.compatible_modes:
            logger.warning(f"Receiver '{self.receiver_id}' not compatible with {mode.name} mode")
            return 0.0
            
        # Limit to max power
        actual_power = min(power, self.max_power)
        
        # Apply receiver efficiency
        received_power = actual_power * self.efficiency
        
        # Update connected transmitters
        self.connected_transmitters[transmitter_id] = received_power
        
        # Update current power
        self.current_power = sum(self.connected_transmitters.values())
        
        return received_power
    
    def get_available_power(self) -> float:
        """Get available power from the receiver."""
        return self.current_power if self.active else 0.0


class WirelessPowerManager:
    """Manager for wireless power transmission systems."""
    
    def __init__(self):
        """Initialize wireless power manager."""
        self.transmitters: Dict[str, WirelessPowerTransmitter] = {}
        self.receivers: Dict[str, WirelessPowerReceiver] = {}
        self.power_links: Dict[str, Dict[str, Any]] = {}  # link_id: link_info
        self.active = False
        
        logger.info("Wireless power manager initialized")
    
    def register_transmitter(self, transmitter: WirelessPowerTransmitter) -> bool:
        """Register a transmitter with the manager."""
        if transmitter.transmitter_id in self.transmitters:
            logger.warning(f"Transmitter '{transmitter.transmitter_id}' already registered")
            return False
            
        self.transmitters[transmitter.transmitter_id] = transmitter
        logger.info(f"Registered transmitter '{transmitter.transmitter_id}'")
        return True
    
    def register_receiver(self, receiver: WirelessPowerReceiver) -> bool:
        """Register a receiver with the manager."""
        if receiver.receiver_id in self.receivers:
            logger.warning(f"Receiver '{receiver.receiver_id}' already registered")
            return False
            
        self.receivers[receiver.receiver_id] = receiver
        logger.info(f"Registered receiver '{receiver.receiver_id}'")
        return True
    
    def create_power_link(self, 
                         transmitter_id: str, 
                         receiver_id: str, 
                         power_allocation: float,
                         distance: float = 10.0,
                         alignment: float = 1.0) -> Optional[str]:
        """
        Create a power link between transmitter and receiver.
        
        Args:
            transmitter_id: Transmitter identifier
            receiver_id: Receiver identifier
            power_allocation: Power allocation in kW
            distance: Distance in meters
            alignment: Alignment factor (0-1)
            
        Returns:
            Link identifier if successful, None otherwise
        """
        if not self.active:
            logger.warning("Cannot create link: wireless power manager not active")
            return None
            
        if transmitter_id not in self.transmitters:
            logger.warning(f"Transmitter '{transmitter_id}' not registered")
            return None
            
        if receiver_id not in self.receivers:
            logger.warning(f"Receiver '{receiver_id}' not registered")
            return None
            
        transmitter = self.transmitters[transmitter_id]
        receiver = self.receivers[receiver_id]
        
        # Check compatibility
        if transmitter.mode not in receiver.compatible_modes:
            logger.warning(f"Incompatible transmission mode: {transmitter.mode.name}")
            return None
            
        # Create link ID
        link_id = f"{transmitter_id}_{receiver_id}_{int(time.time())}"
        
        # Add receiver to transmitter
        if not transmitter.add_receiver(receiver_id, power_allocation):
            return None
            
        # Store link information
        self.power_links[link_id] = {
            "transmitter_id": transmitter_id,
            "receiver_id": receiver_id,
            "power_allocation": power_allocation,
            "distance": distance,
            "alignment": alignment,
            "created_time": time.time(),
            "active": True
        }
        
        logger.info(f"Created power link '{link_id}' from '{transmitter_id}' to '{receiver_id}'")
        return link_id
    
    def remove_power_link(self, link_id: str) -> bool:
        """Remove a power link."""
        if link_id not in self.power_links:
            return False
            
        link_info = self.power_links[link_id]
        transmitter_id = link_info["transmitter_id"]
        receiver_id = link_info["receiver_id"]
        
        # Remove receiver from transmitter
        if transmitter_id in self.transmitters:
            self.transmitters[transmitter_id].remove_receiver(receiver_id)
            
        # Remove link
        del self.power_links[link_id]
        
        logger.info(f"Removed power link '{link_id}'")
        return True
    
    def activate(self) -> bool:
        """Activate the wireless power manager."""
        if self.active:
            return True
            
        self.active = True
        logger.info("Wireless power manager activated")
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the wireless power manager."""
        if not self.active:
            return True
            
        # Deactivate all transmitters
        for transmitter in self.transmitters.values():
            transmitter.deactivate()
            
        # Deactivate all receivers
        for receiver in self.receivers.values():
            receiver.deactivate()
            
        # Clear all links
        self.power_links.clear()
        
        self.active = False
        logger.info("Wireless power manager deactivated")
        return True
    
    def update(self) -> Dict[str, Any]:
        """
        Update wireless power transmission.
        
        Returns:
            Status information
        """
        if not self.active:
            return {"active": False}
            
        # Process each link
        link_status = {}
        for link_id, link_info in self.power_links.items():
            if not link_info["active"]:
                continue
                
            transmitter_id = link_info["transmitter_id"]
            receiver_id = link_info["receiver_id"]
            
            if transmitter_id not in self.transmitters or receiver_id not in self.receivers:
                link_info["active"] = False
                continue
                
            transmitter = self.transmitters[transmitter_id]
            receiver = self.receivers[receiver_id]
            
            # Calculate effective power
            effective_power = transmitter.get_effective_power(
                link_info["distance"], 
                link_info["alignment"]
            )
            
            # Receiver gets power
            received_power = 0.0
            if receiver_id in effective_power:
                received_power = receiver.receive_power(
                    transmitter_id,
                    effective_power[receiver_id],
                    transmitter.mode
                )
            
            # Update link status
            link_status[link_id] = {
                "transmitter": transmitter_id,
                "receiver": receiver_id,
                "allocated_power": link_info["power_allocation"],
                "effective_power": effective_power.get(receiver_id, 0.0),
                "received_power": received_power,
                "efficiency": (received_power / effective_power.get(receiver_id, 1.0)) if effective_power.get(receiver_id, 0.0) > 0 else 0.0,
                "distance": link_info["distance"],
                "alignment": link_info["alignment"],
                "mode": transmitter.mode.name
            }
        
        return {
            "active": self.active,
            "timestamp": time.time(),
            "links": link_status,
            "total_transmitted": sum(t.current_power for t in self.transmitters.values()),
            "total_received": sum(r.current_power for r in self.receivers.values())
        }
    
    def optimize_transmission(self) -> Dict[str, Any]:
        """
        Optimize wireless power transmission.
        
        Returns:
            Optimization results
        """
        if not self.active or not self.power_links:
            return {"optimized": False}
            
        optimization_results = {
            "optimized": True,
            "adjustments": {}
        }
        
        # For each transmitter, optimize its receivers
        for transmitter_id, transmitter in self.transmitters.items():
            if not transmitter.active or not transmitter.target_receivers:
                continue
                
            # Get links for this transmitter
            transmitter_links = {
                link_id: link_info for link_id, link_info in self.power_links.items()
                if link_info["transmitter_id"] == transmitter_id and link_info["active"]
            }
            
            if not transmitter_links:
                continue
                
            # Calculate optimal alignment for each link
            for link_id, link_info in transmitter_links.items():
                receiver_id = link_info["receiver_id"]
                
                # Simulate different alignments
                best_alignment = link_info["alignment"]
                best_efficiency = 0.0
                
                for test_alignment in [0.7, 0.8, 0.9, 1.0]:
                    # Calculate power with this alignment
                    test_power = transmitter.get_effective_power(
                        link_info["distance"], 
                        test_alignment
                    ).get(receiver_id, 0.0)
                    
                    # Calculate efficiency
                    allocation = link_info["power_allocation"]
                    efficiency = test_power / allocation if allocation > 0 else 0.0
                    
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_alignment = test_alignment
                
                # Update alignment if better found
                if abs(best_alignment - link_info["alignment"]) > 0.01:
                    old_alignment = link_info["alignment"]
                    link_info["alignment"] = best_alignment
                    
                    optimization_results["adjustments"][link_id] = {
                        "parameter": "alignment",
                        "old_value": old_alignment,
                        "new_value": best_alignment,
                        "improvement": f"{((best_efficiency - 1.0) * 100):.1f}%"
                    }
        
        return optimization_results


class WirelessPowerIntegrator:
    """Integrates wireless power with other systems."""
    
    def __init__(self, wireless_manager: WirelessPowerManager):
        """
        Initialize wireless power integrator.
        
        Args:
            wireless_manager: Wireless power manager
        """
        self.wireless_manager = wireless_manager
        self.power_consumers: Dict[str, Dict[str, Any]] = {}
        self.power_sources: Dict[str, Dict[str, Any]] = {}
        self.integration_active = False
        
        logger.info("Wireless power integrator initialized")
    
    def register_power_consumer(self, 
                              consumer_id: str, 
                              power_requirement: float,
                              priority: PowerPriority = PowerPriority.MEDIUM) -> bool:
        """
        Register a power consumer.
        
        Args:
            consumer_id: Consumer identifier
            power_requirement: Power requirement in kW
            priority: Consumer priority
            
        Returns:
            Success status
        """
        if consumer_id in self.power_consumers:
            logger.warning(f"Consumer '{consumer_id}' already registered")
            return False
            
        # Create receiver for this consumer
        receiver = WirelessPowerReceiver(
            receiver_id=f"rcv_{consumer_id}",
            max_power=power_requirement * 1.2,  # Add 20% margin
            efficiency=0.85
        )
        
        # Register receiver with manager
        if not self.wireless_manager.register_receiver(receiver):
            return False
            
        # Store consumer information
        self.power_consumers[consumer_id] = {
            "receiver_id": receiver.receiver_id,
            "power_requirement": power_requirement,
            "priority": priority,
            "links": []
        }
        
        logger.info(f"Registered power consumer '{consumer_id}'")
        return True
    
    def register_power_source(self, 
                            source_id: str, 
                            available_power: float,
                            mode: WirelessTransmissionMode = WirelessTransmissionMode.MICROWAVE) -> bool:
        """
        Register a power source.
        
        Args:
            source_id: Source identifier
            available_power: Available power in kW
            mode: Transmission mode
            
        Returns:
            Success status
        """
        if source_id in self.power_sources:
            logger.warning(f"Source '{source_id}' already registered")
            return False
            
        # Create transmitter for this source
        transmitter = WirelessPowerTransmitter(
            transmitter_id=f"xmt_{source_id}",
            max_power=available_power,
            mode=mode
        )
        
        # Register transmitter with manager
        if not self.wireless_manager.register_transmitter(transmitter):
            return False
            
        # Store source information
        self.power_sources[source_id] = {
            "transmitter_id": transmitter.transmitter_id,
            "available_power": available_power,
            "mode": mode,
            "links": []
        }
        
        logger.info(f"Registered power source '{source_id}'")
        return True
    
    def activate(self) -> bool:
        """Activate wireless power integration."""
        if self.integration_active:
            return True
            
        # Activate wireless manager
        if not self.wireless_manager.activate():
            return False
            
        # Activate all transmitters and receivers
        for source in self.power_sources.values():
            transmitter_id = source["transmitter_id"]
            if transmitter_id in self.wireless_manager.transmitters:
                self.wireless_manager.transmitters[transmitter_id].activate()
                
        for consumer in self.power_consumers.values():
            receiver_id = consumer["receiver_id"]
            if receiver_id in self.wireless_manager.receivers:
                self.wireless_manager.receivers[receiver_id].activate()
        
        self.integration_active = True
        logger.info("Wireless power integration activated")
        return True
    
    def deactivate(self) -> bool:
        """Deactivate wireless power integration."""
        if not self.integration_active:
            return True
            
        # Deactivate wireless manager
        self.wireless_manager.deactivate()
        
        self.integration_active = False
        logger.info("Wireless power integration deactivated")
        return True
    
    def create_power_connections(self) -> Dict[str, Any]:
        """
        Create optimal power connections between sources and consumers.
        
        Returns:
            Connection results
        """
        if not self.integration_active:
            return {"success": False, "error": "Integration not active"}
            
        # Sort consumers by priority
        sorted_consumers = sorted(
            self.power_consumers.items(),
            key=lambda x: x[1]["priority"].value,
            reverse=True
        )
        
        # Sort sources by available power
        sorted_sources = sorted(
            self.power_sources.items(),
            key=lambda x: x[1]["available_power"],
            reverse=True
        )
        
        if not sorted_consumers or not sorted_sources:
            return {"success": False, "error": "No consumers or sources available"}
            
        # Create connections
        connections = []
        
        for consumer_id, consumer in sorted_consumers:
            receiver_id = consumer["receiver_id"]
            requirement = consumer["power_requirement"]
            
            # Find best source for this consumer
            for source_id, source in sorted_sources:
                transmitter_id = source["transmitter_id"]
                
                # Check if source has enough power
                transmitter = self.wireless_manager.transmitters.get(transmitter_id)
                if not transmitter or not transmitter.active:
                    continue
                    
                available = transmitter.max_power - transmitter.current_power
                if available < requirement * 0.1:  # Need at least 10% of requirement
                    continue
                    
                # Create link
                allocation = min(requirement * 1.5, available)  # Allocate with 50% margin
                link_id = self.wireless_manager.create_power_link(
                    transmitter_id,
                    receiver_id,
                    allocation,
                    distance=50.0,  # Default distance
                    alignment=0.9   # Default alignment
                )
                
                if link_id:
                    connections.append({
                        "link_id": link_id,
                        "consumer_id": consumer_id,
                        "source_id": source_id,
                        "allocation": allocation
                    })
                    
                    # Update consumer and source
                    consumer["links"].append(link_id)
                    source["links"].append(link_id)
                    break
        
        return {
            "success": True,
            "connections": connections,
            "total_connections": len(connections)
        }
    
    def get_power_status(self) -> Dict[str, Any]:
        """
        Get power status for all consumers and sources.
        
        Returns:
            Power status information
        """
        if not self.integration_active:
            return {"active": False}
            
        # Update wireless manager
        wireless_status = self.wireless_manager.update()
        
        # Compile consumer status
        consumer_status = {}
        for consumer_id, consumer in self.power_consumers.items():
            receiver_id = consumer["receiver_id"]
            receiver = self.wireless_manager.receivers.get(receiver_id)
            
            if not receiver:
                continue
                
            # Calculate power satisfaction
            received = receiver.get_available_power()
            required = consumer["power_requirement"]
            satisfaction = (received / required) * 100 if required > 0 else 0
            
            consumer_status[consumer_id] = {
                "received_power": received,
                "required_power": required,
                "satisfaction": satisfaction,
                "active": receiver.active,
                "links": consumer["links"]
            }
        
        # Compile source status
        source_status = {}
        for source_id, source in self.power_sources.items():
            transmitter_id = source["transmitter_id"]
            transmitter = self.wireless_manager.transmitters.get(transmitter_id)
            
            if not transmitter:
                continue
                
            # Calculate utilization
            current = transmitter.current_power
            available = transmitter.max_power
            utilization = (current / available) * 100 if available > 0 else 0
            
            source_status[source_id] = {
                "current_power": current,
                "available_power": available,
                "utilization": utilization,
                "active": transmitter.active,
                "mode": transmitter.mode.name,
                "links": source["links"]
            }
        
        return {
            "active": self.integration_active,
            "timestamp": time.time(),
            "consumers": consumer_status,
            "sources": source_status,
            "wireless": wireless_status
        }
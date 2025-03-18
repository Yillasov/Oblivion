"""
Standardized Communication Protocols

Provides unified communication interfaces for different neuromorphic hardware types.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import time
import json
from enum import Enum

from src.core.utils.logging_framework import get_logger
from src.core.hardware.exceptions import HardwareCommunicationError

logger = get_logger("hardware_protocols")


class ProtocolType(Enum):
    """Communication protocol types."""
    SPIKE = "spike"           # Spike-based communication
    RATE = "rate"             # Rate-based communication
    DIRECT = "direct"         # Direct value communication
    MULTICAST = "multicast"   # One-to-many communication


class CommunicationProtocol:
    """Base class for hardware communication protocols."""
    
    def __init__(self, hardware_type: str):
        """Initialize communication protocol."""
        self.hardware_type = hardware_type
        self.protocol_name = f"{hardware_type}_protocol"
        
    def encode_message(self, message: Dict[str, Any]) -> bytes:
        """Encode message for transmission to hardware."""
        try:
            return json.dumps(message).encode('utf-8')
        except Exception as e:
            logger.error(f"Message encoding error: {str(e)}")
            raise HardwareCommunicationError(f"Failed to encode message: {str(e)}")
    
    def decode_message(self, data: bytes) -> Dict[str, Any]:
        """Decode message received from hardware."""
        try:
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Message decoding error: {str(e)}")
            raise HardwareCommunicationError(f"Failed to decode message: {str(e)}")
    
    def send(self, target_id: int, message: Dict[str, Any]) -> bool:
        """Send message to hardware (to be implemented by subclasses)."""
        raise NotImplementedError("Communication protocol must implement send method")
    
    def receive(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """Receive message from hardware (to be implemented by subclasses)."""
        raise NotImplementedError("Communication protocol must implement receive method")


class LoihiProtocol(CommunicationProtocol):
    """Loihi-specific communication protocol."""
    
    def __init__(self):
        """Initialize Loihi communication protocol."""
        super().__init__("loihi")
        self.packet_size = 64  # Loihi uses 64-byte packets
    
    def send(self, target_id: int, message: Dict[str, Any]) -> bool:
        """Send message to Loihi hardware."""
        try:
            # Convert message to spike format for Loihi
            if "spikes" in message:
                # Loihi expects specific spike format
                encoded = self._format_spike_data(message["spikes"], target_id)
            else:
                # Standard message format
                encoded = self.encode_message(message)
                
            # In a real implementation, this would use Loihi's NxSDK
            logger.debug(f"Sending to Loihi target {target_id}: {len(encoded)} bytes")
            return True
        except Exception as e:
            logger.error(f"Loihi send error: {str(e)}")
            return False
    
    def receive(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """Receive message from Loihi hardware."""
        try:
            # In a real implementation, this would use Loihi's NxSDK
            # Simulated response for demonstration
            return {"status": "ok", "timestamp": time.time()}
        except Exception as e:
            logger.error(f"Loihi receive error: {str(e)}")
            return None
    
    def _format_spike_data(self, spikes: Dict[int, List[float]], target_id: int) -> bytes:
        """Format spike data for Loihi hardware."""
        # Loihi-specific spike encoding
        formatted = {"target": target_id, "spike_data": spikes}
        return self.encode_message(formatted)


class SpiNNakerProtocol(CommunicationProtocol):
    """SpiNNaker-specific communication protocol."""
    
    def __init__(self):
        """Initialize SpiNNaker communication protocol."""
        super().__init__("spinnaker")
        self.uses_multicast = True
    
    def send(self, target_id: int, message: Dict[str, Any]) -> bool:
        """Send message to SpiNNaker hardware."""
        try:
            # SpiNNaker uses multicast routing
            if self.uses_multicast and "targets" in message:
                # Send to multiple targets
                targets = message["targets"]
                logger.debug(f"Sending multicast to {len(targets)} SpiNNaker targets")
            else:
                # Single target
                encoded = self.encode_message(message)
                logger.debug(f"Sending to SpiNNaker target {target_id}: {len(encoded)} bytes")
            return True
        except Exception as e:
            logger.error(f"SpiNNaker send error: {str(e)}")
            return False
    
    def receive(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """Receive message from SpiNNaker hardware."""
        try:
            # In a real implementation, this would use SpiNNaker's API
            # Simulated response for demonstration
            return {"status": "ok", "timestamp": time.time()}
        except Exception as e:
            logger.error(f"SpiNNaker receive error: {str(e)}")
            return None


class TrueNorthProtocol(CommunicationProtocol):
    """TrueNorth-specific communication protocol."""
    
    def __init__(self):
        """Initialize TrueNorth communication protocol."""
        super().__init__("truenorth")
    
    def send(self, target_id: int, message: Dict[str, Any]) -> bool:
        """Send message to TrueNorth hardware."""
        try:
            # TrueNorth has specific data format requirements
            encoded = self.encode_message(message)
            logger.debug(f"Sending to TrueNorth target {target_id}: {len(encoded)} bytes")
            return True
        except Exception as e:
            logger.error(f"TrueNorth send error: {str(e)}")
            return False
    
    def receive(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """Receive message from TrueNorth hardware."""
        try:
            # In a real implementation, this would use TrueNorth's API
            # Simulated response for demonstration
            return {"status": "ok", "timestamp": time.time()}
        except Exception as e:
            logger.error(f"TrueNorth receive error: {str(e)}")
            return None


class ProtocolFactory:
    """Factory for creating communication protocol instances."""
    
    @staticmethod
    def create_protocol(hardware_type: str) -> CommunicationProtocol:
        """
        Create a communication protocol for the specified hardware type.
        
        Args:
            hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth')
            
        Returns:
            CommunicationProtocol: Protocol instance
            
        Raises:
            ValueError: If hardware type is unsupported
        """
        hardware_type = hardware_type.lower()
        
        if hardware_type == "loihi":
            return LoihiProtocol()
        elif hardware_type == "spinnaker":
            return SpiNNakerProtocol()
        elif hardware_type == "truenorth":
            return TrueNorthProtocol()
        else:
            logger.warning(f"Unsupported hardware type: {hardware_type}, using generic protocol")
            return CommunicationProtocol(hardware_type)
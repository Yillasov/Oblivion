"""
Hardware interface definitions and factory for neuromorphic systems.
"""

from typing import Optional, Protocol
from dataclasses import dataclass
from src.core.integration.neuromorphic_system import NeuromorphicInterface as SystemInterface

class NeuromorphicInterface(SystemInterface):
    """Protocol defining the interface for neuromorphic hardware."""
    
    def initialize(self) -> None:
        """Initialize the hardware connection."""
        ...
        
    def cleanup(self) -> None:
        """Clean up hardware resources."""
        ...

@dataclass
class LoihiInterface(NeuromorphicInterface):
    """Intel Loihi neuromorphic hardware interface."""
    address: Optional[str] = None
    
    def initialize(self) -> None:
        pass
        
    def cleanup(self) -> None:
        pass

@dataclass
class TrueNorthInterface(NeuromorphicInterface):
    """IBM TrueNorth neuromorphic hardware interface."""
    address: Optional[str] = None
    
    def initialize(self) -> None:
        pass
        
    def cleanup(self) -> None:
        pass

def create_hardware_interface(address: Optional[str] = None) -> Optional[SystemInterface]:
    """
    Create appropriate hardware interface based on address format.
    
    Args:
        address: Hardware address string
        
    Returns:
        NeuromorphicInterface implementation or None
    """
    if not address:
        return None
        
    if address.startswith('loihi:'):
        return LoihiInterface(address=address.replace('loihi:', ''))
    elif address.startswith('truenorth:'):
        return TrueNorthInterface(address=address.replace('truenorth:', ''))
    
    # Default to Loihi interface
    return LoihiInterface(address=address)
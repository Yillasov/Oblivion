from typing import Dict, Any, Type
from .base import AirframeBase
from .types import (
    MorphingWingDrone, BiomimeticDrone, HypersonicDrone,
    SpaceCapableDrone, UnderwaterLaunchedDrone, SwarmConfiguredDrone,
    StealthDrone, DirectedEnergyDrone, VTOLHighSpeedDrone, ModularPayloadDrone
)

class AirframeFactory:
    """Factory for creating airframe instances."""
    
    _airframe_types = {
        "morphing_wing": MorphingWingDrone,
        "biomimetic": BiomimeticDrone,
        "hypersonic": HypersonicDrone,
        "space_capable": SpaceCapableDrone,
        "underwater_launched": UnderwaterLaunchedDrone,
        "swarm_configured": SwarmConfiguredDrone,
        "stealth": StealthDrone,
        "directed_energy": DirectedEnergyDrone,
        "vtol_high_speed": VTOLHighSpeedDrone,
        "modular_payload": ModularPayloadDrone
    }
    
    @classmethod
    def create(cls, airframe_type: str, config: Dict[str, Any]) -> AirframeBase:
        """
        Create an airframe instance of the specified type.
        
        Args:
            airframe_type: Type of airframe to create
            config: Configuration for the airframe
            
        Returns:
            An instance of the specified airframe type
        
        Raises:
            ValueError: If the airframe type is not supported
        """
        if airframe_type not in cls._airframe_types:
            raise ValueError(f"Unsupported airframe type: {airframe_type}")
        
        return cls._airframe_types[airframe_type](config)
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get a list of available airframe types."""
        return list(cls._airframe_types.keys())
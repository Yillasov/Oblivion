"""
Integration of payload systems with UCAV geometry and aerodynamics.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional
import numpy as np
from src.simulation.aerodynamics.ucav_model import UCAVGeometry
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.payload.base import PayloadInterface, PayloadSpecs
from src.payload.types import PayloadMountType


class PayloadIntegrator:
    """Integrates payload systems with UCAV geometry and analyzes impacts."""
    
    def __init__(self, hardware_interface=None):
        """
        Initialize the payload integrator.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
        """
        self.system = NeuromorphicSystem(hardware_interface)
        self.payloads = []
        self.ucav_geometry = None
    
    def set_ucav_geometry(self, geometry: UCAVGeometry) -> None:
        """
        Set the UCAV geometry for payload integration.
        
        Args:
            geometry: UCAV geometry model
        """
        self.ucav_geometry = geometry
    
    def add_payload(self, payload: PayloadInterface, mount_type: PayloadMountType, 
                   mount_location: Dict[str, float]) -> bool:
        """
        Add a payload to the UCAV.
        
        Args:
            payload: Payload to add
            mount_type: Type of mounting system
            mount_location: Location coordinates for mounting
            
        Returns:
            Success status
        """
        if not self.ucav_geometry:
            raise ValueError("UCAV geometry must be set before adding payloads")
        
        # Check if payload can be integrated
        if self._validate_payload_integration(payload, mount_type, mount_location):
            self.payloads.append({
                "payload": payload,
                "mount_type": mount_type,
                "mount_location": mount_location
            })
            return True
        return False
    
    def _validate_payload_integration(self, payload: PayloadInterface, 
                                    mount_type: PayloadMountType,
                                    mount_location: Dict[str, float]) -> bool:
        """
        Validate if a payload can be integrated with the UCAV.
        
        Args:
            payload: Payload to validate
            mount_type: Type of mounting system
            mount_location: Location coordinates for mounting
            
        Returns:
            Validation result
        """
        specs = payload.get_specifications()
        
        # Use neuromorphic processing to validate integration
        self.system.initialize()
        validation_result = self.system.process_data({
            'geometry': self.ucav_geometry.__dict__,
            'payload_specs': vars(specs),
            'mount_type': mount_type.name,
            'mount_location': mount_location,
            'computation': 'payload_integration_validation'
        })
        self.system.cleanup()
        
        return validation_result.get('valid', False)
    
    def calculate_performance_impact(self) -> Dict[str, Any]:
        """
        Calculate the impact of all payloads on UCAV performance.
        
        Returns:
            Dict containing performance impacts
        """
        if not self.ucav_geometry or not self.payloads:
            return {"error": "UCAV geometry not set or no payloads added"}
        
        # Collect payload data
        payload_data = []
        for p in self.payloads:
            payload_data.append({
                "specs": vars(p["payload"].get_specifications()),
                "mount_type": p["mount_type"].name,
                "mount_location": p["mount_location"],
                "impact": p["payload"].calculate_impact()
            })
        
        # Use neuromorphic processing to calculate combined impact
        self.system.initialize()
        impact_result = self.system.process_data({
            'geometry': self.ucav_geometry.__dict__,
            'payloads': payload_data,
            'computation': 'payload_performance_impact'
        })
        self.system.cleanup()
        
        return impact_result
    
    def optimize_payload_configuration(self, mission_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the payload configuration for given mission parameters.
        
        Args:
            mission_parameters: Parameters defining the mission
            
        Returns:
            Optimized configuration
        """
        if not self.ucav_geometry or not self.payloads:
            return {"error": "UCAV geometry not set or no payloads added"}
        
        # Use neuromorphic processing for optimization
        self.system.initialize()
        optimization_result = self.system.process_data({
            'geometry': self.ucav_geometry.__dict__,
            'payloads': [vars(p["payload"].get_specifications()) for p in self.payloads],
            'mission': mission_parameters,
            'computation': 'payload_configuration_optimization'
        })
        self.system.cleanup()
        
        return optimization_result
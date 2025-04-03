#!/usr/bin/env python3
"""
Stealth Detection Integration

Integrates stealth effectiveness with sensor detection systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, Any, List, Optional

from src.stealth.effectiveness.stealth_effectiveness import StealthEffectivenessEvaluator
from src.simulation.sensors.stealth_detection import SignatureType, MultiSignatureDetector


class StealthDetectionIntegrator:
    """Integrates stealth effectiveness with sensor detection systems."""
    
    def __init__(self, effectiveness_evaluator: Optional[StealthEffectivenessEvaluator] = None):
        """Initialize stealth detection integrator."""
        self.evaluator = effectiveness_evaluator or StealthEffectivenessEvaluator()
        
    def apply_stealth_to_target(self, 
                               target_data: Dict[str, Any], 
                               platform_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply stealth effectiveness to target data.
        
        Args:
            target_data: Target data to modify
            platform_state: Current platform state
            
        Returns:
            Modified target data with stealth effects applied
        """
        # Get reduction factors for each signature type
        radar_reduction = self.evaluator.get_signature_reduction_factor(
            SignatureType.RADAR, platform_state)
        ir_reduction = self.evaluator.get_signature_reduction_factor(
            SignatureType.INFRARED, platform_state)
        acoustic_reduction = self.evaluator.get_signature_reduction_factor(
            SignatureType.ACOUSTIC, platform_state)
        em_reduction = self.evaluator.get_signature_reduction_factor(
            SignatureType.ELECTROMAGNETIC, platform_state)
        
        # Apply reductions to target signatures
        if 'rcs' in target_data:
            target_data['rcs'] = target_data['rcs'] * radar_reduction
            
        if 'ir_signature' in target_data:
            target_data['ir_signature'] = target_data['ir_signature'] * ir_reduction
            
        if 'acoustic_signature' in target_data:
            target_data['acoustic_signature'] = target_data['acoustic_signature'] * acoustic_reduction
            
        if 'em_signature' in target_data:
            target_data['em_signature'] = target_data['em_signature'] * em_reduction
            
        return target_data
    
    def process_environment(self, environment: Dict[str, Any], platform_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process environment data to apply stealth effects to all targets.
        
        Args:
            environment: Environment data with targets
            platform_state: Current platform state
            
        Returns:
            Modified environment with stealth effects applied to targets
        """
        # Get targets from environment
        targets = environment.get('targets', [])
        
        # Apply stealth to each target
        for i, target in enumerate(targets):
            targets[i] = self.apply_stealth_to_target(target, platform_state)
            
        # Update environment with modified targets
        environment['targets'] = targets
        
        return environment
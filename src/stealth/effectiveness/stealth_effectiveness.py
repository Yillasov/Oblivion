#!/usr/bin/env python3
"""
Stealth Effectiveness Module

Evaluates the effectiveness of stealth systems against different sensor types.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

from src.stealth.base.interfaces import StealthInterface, StealthType
from src.stealth.base.config import StealthEffectivenessLevel
from src.simulation.sensors.stealth_detection import SignatureType


@dataclass
class EffectivenessRating:
    """Rating of stealth system effectiveness against a sensor type."""
    signature_type: SignatureType
    effectiveness: float  # 0.0 to 1.0 scale
    confidence: float  # 0.0 to 1.0 scale


class StealthEffectivenessEvaluator:
    """Evaluates stealth system effectiveness against different sensor types."""
    
    def __init__(self):
        """Initialize stealth effectiveness evaluator."""
        self.stealth_systems: Dict[str, StealthInterface] = {}
        self.effectiveness_cache: Dict[str, Dict[SignatureType, EffectivenessRating]] = {}
        
    def register_stealth_system(self, system_id: str, system: StealthInterface) -> None:
        """Register a stealth system for evaluation."""
        self.stealth_systems[system_id] = system
        self.effectiveness_cache[system_id] = {}
        
    def evaluate_effectiveness(self, 
                              system_id: str, 
                              signature_type: SignatureType,
                              platform_state: Optional[Dict[str, Any]] = None) -> EffectivenessRating:
        """
        Evaluate stealth system effectiveness against a specific signature type.
        
        Args:
            system_id: ID of the stealth system
            signature_type: Type of signature to evaluate against
            platform_state: Current platform state (optional)
            
        Returns:
            EffectivenessRating for the system against the signature type
        """
        if system_id not in self.stealth_systems:
            return EffectivenessRating(
                signature_type=signature_type,
                effectiveness=0.0,
                confidence=1.0
            )
            
        # Get stealth system
        system = self.stealth_systems[system_id]
        
        # Check if system is active
        if not system.get_status().get("active", False):
            return EffectivenessRating(
                signature_type=signature_type,
                effectiveness=0.0,
                confidence=1.0
            )
            
        # Get system type and specs
        system_specs = system.get_specifications()
        stealth_type = system_specs.stealth_type
        
        # Calculate base effectiveness based on stealth type and signature type
        base_effectiveness = self._calculate_base_effectiveness(stealth_type, signature_type)
        
        # Apply system-specific modifiers
        system_effectiveness = self._apply_system_modifiers(
            base_effectiveness, 
            system, 
            signature_type,
            platform_state
        )
        
        # Create effectiveness rating
        rating = EffectivenessRating(
            signature_type=signature_type,
            effectiveness=system_effectiveness,
            confidence=0.9  # High confidence in evaluation
        )
        
        # Cache the rating
        self.effectiveness_cache[system_id][signature_type] = rating
        
        return rating
    
    def evaluate_combined_effectiveness(self, 
                                       signature_type: SignatureType,
                                       platform_state: Optional[Dict[str, Any]] = None) -> EffectivenessRating:
        """
        Evaluate combined effectiveness of all active stealth systems.
        
        Args:
            signature_type: Type of signature to evaluate against
            platform_state: Current platform state (optional)
            
        Returns:
            Combined EffectivenessRating against the signature type
        """
        # Get effectiveness ratings for all active systems
        ratings = []
        for system_id in self.stealth_systems:
            rating = self.evaluate_effectiveness(system_id, signature_type, platform_state)
            if rating.effectiveness > 0:
                ratings.append(rating)
                
        if not ratings:
            return EffectivenessRating(
                signature_type=signature_type,
                effectiveness=0.0,
                confidence=1.0
            )
            
        # Calculate combined effectiveness (diminishing returns model)
        combined_effectiveness = 0.0
        for i, rating in enumerate(sorted(ratings, key=lambda r: r.effectiveness, reverse=True)):
            # First system contributes fully, subsequent systems contribute less
            contribution = rating.effectiveness * (0.8 ** i)
            combined_effectiveness = combined_effectiveness + contribution * (1.0 - combined_effectiveness)
            
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in ratings) / len(ratings)
        
        return EffectivenessRating(
            signature_type=signature_type,
            effectiveness=min(0.95, combined_effectiveness),  # Cap at 95% effectiveness
            confidence=avg_confidence
        )
    
    def get_signature_reduction_factor(self, 
                                      signature_type: SignatureType,
                                      platform_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Get the signature reduction factor for a specific signature type.
        
        Args:
            signature_type: Type of signature
            platform_state: Current platform state (optional)
            
        Returns:
            Reduction factor (0.0 to 1.0, where 0.0 is complete reduction)
        """
        rating = self.evaluate_combined_effectiveness(signature_type, platform_state)
        return 1.0 - rating.effectiveness
    
    def _calculate_base_effectiveness(self, 
                                     stealth_type: StealthType, 
                                     signature_type: SignatureType) -> float:
        """Calculate base effectiveness based on stealth type and signature type."""
        # Mapping of stealth types to signature types with effectiveness values
        effectiveness_map = {
            StealthType.RADAR_ABSORBING: {
                SignatureType.RADAR: 0.8,
                SignatureType.ELECTROMAGNETIC: 0.6,
                SignatureType.INFRARED: 0.1,
                SignatureType.ACOUSTIC: 0.0
            },
            StealthType.RADAR_ABSORBENT_MATERIAL: {
                SignatureType.RADAR: 0.7,
                SignatureType.ELECTROMAGNETIC: 0.5,
                SignatureType.INFRARED: 0.0,
                SignatureType.ACOUSTIC: 0.0
            },
            StealthType.PLASMA_STEALTH: {
                SignatureType.RADAR: 0.9,
                SignatureType.ELECTROMAGNETIC: 0.7,
                SignatureType.INFRARED: 0.3,
                SignatureType.ACOUSTIC: 0.0
            },
            StealthType.ACTIVE_CAMOUFLAGE: {
                SignatureType.RADAR: 0.2,
                SignatureType.ELECTROMAGNETIC: 0.3,
                SignatureType.INFRARED: 0.8,
                SignatureType.ACOUSTIC: 0.0
            },
            StealthType.METAMATERIAL_CLOAKING: {
                SignatureType.RADAR: 0.9,
                SignatureType.ELECTROMAGNETIC: 0.8,
                SignatureType.INFRARED: 0.5,
                SignatureType.ACOUSTIC: 0.2
            },
            StealthType.ACOUSTIC_REDUCTION: {
                SignatureType.RADAR: 0.0,
                SignatureType.ELECTROMAGNETIC: 0.0,
                SignatureType.INFRARED: 0.0,
                SignatureType.ACOUSTIC: 0.7
            },
            StealthType.ACOUSTIC_DAMPENING: {
                SignatureType.RADAR: 0.0,
                SignatureType.ELECTROMAGNETIC: 0.0,
                SignatureType.INFRARED: 0.0,
                SignatureType.ACOUSTIC: 0.8
            },
            StealthType.INFRARED_SUPPRESSION: {
                SignatureType.RADAR: 0.0,
                SignatureType.ELECTROMAGNETIC: 0.1,
                SignatureType.INFRARED: 0.8,
                SignatureType.ACOUSTIC: 0.0
            },
            StealthType.THERMAL_CAMOUFLAGE: {
                SignatureType.RADAR: 0.0,
                SignatureType.ELECTROMAGNETIC: 0.1,
                SignatureType.INFRARED: 0.9,
                SignatureType.ACOUSTIC: 0.0
            },
            StealthType.EMP_SHIELDING: {
                SignatureType.RADAR: 0.3,
                SignatureType.ELECTROMAGNETIC: 0.9,
                SignatureType.INFRARED: 0.0,
                SignatureType.ACOUSTIC: 0.0
            },
            StealthType.ELECTROMAGNETIC_SHIELDING: {
                SignatureType.RADAR: 0.4,
                SignatureType.ELECTROMAGNETIC: 0.8,
                SignatureType.INFRARED: 0.0,
                SignatureType.ACOUSTIC: 0.0
            },
            StealthType.LOW_OBSERVABLE_NOZZLE: {
                SignatureType.RADAR: 0.3,
                SignatureType.ELECTROMAGNETIC: 0.2,
                SignatureType.INFRARED: 0.7,
                SignatureType.ACOUSTIC: 0.5
            }
        }
        
        # Get effectiveness for this stealth type against this signature type
        return effectiveness_map.get(stealth_type, {}).get(signature_type, 0.0)
    
    def _apply_system_modifiers(self, 
                               base_effectiveness: float, 
                               system: StealthInterface,
                               signature_type: SignatureType,
                               platform_state: Optional[Dict[str, Any]] = None) -> float:
        """Apply system-specific modifiers to base effectiveness."""
        # Get system status
        status = system.get_status()
        
        # Get system specifications
        specs = system.get_specifications()
        
        # Apply integrity modifier
        integrity = status.get("integrity", 1.0)
        effectiveness = base_effectiveness * integrity
        
        # Apply power mode modifier
        power_level = status.get("power_level", 0.5)
        effectiveness = effectiveness * (0.7 + (0.3 * power_level))
        
        # Apply system-specific modifiers based on signature type
        if signature_type == SignatureType.RADAR:
            # RCS-specific modifiers
            rcs_reduction = specs.radar_cross_section
            effectiveness = effectiveness * min(1.0, 1.0 - rcs_reduction)
            
        elif signature_type == SignatureType.INFRARED:
            # IR-specific modifiers
            ir_reduction = specs.infrared_signature
            effectiveness = effectiveness * min(1.0, 1.0 - ir_reduction)
            
        elif signature_type == SignatureType.ACOUSTIC:
            # Acoustic-specific modifiers
            acoustic_reduction = specs.acoustic_signature
            effectiveness = effectiveness * min(1.0, 1.0 - acoustic_reduction)
            
        # Apply platform state modifiers if available
        if platform_state:
            # Speed affects effectiveness
            speed = platform_state.get("speed", 0.0)
            if speed > 500:  # High speed reduces effectiveness
                effectiveness = effectiveness * 0.9
                
            # Altitude affects effectiveness
            altitude = platform_state.get("altitude", 0.0)
            if altitude > 10000:  # High altitude improves effectiveness
                effectiveness = min(1.0, effectiveness * 1.1)
                
        return min(1.0, max(0.0, effectiveness))
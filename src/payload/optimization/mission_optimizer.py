#!/usr/bin/env python3
"""
Mission-specific payload optimization for UCAV platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from src.payload.optimization.payload_optimizer import PayloadOptimizer, OptimizationConstraints, OptimizationResult


@dataclass
class MissionProfile:
    """Profile for a specific mission type."""
    name: str
    description: str
    target_types: List[str]
    priority_level: int  # 1-10, with 10 being highest priority
    environmental_requirements: Dict[str, Any]
    payload_preferences: Dict[str, float]  # Payload ID to preference weight mapping


class MissionOptimizer:
    """
    Optimizes payload configurations for specific mission profiles.
    """
    
    def __init__(self, payload_optimizer: PayloadOptimizer):
        """
        Initialize the mission optimizer.
        
        Args:
            payload_optimizer: Payload optimizer instance
        """
        self.optimizer = payload_optimizer
        self.mission_profiles: Dict[str, MissionProfile] = {}
    
    def register_mission_profile(self, profile_id: str, profile: MissionProfile) -> bool:
        """
        Register a mission profile.
        
        Args:
            profile_id: Unique identifier for the profile
            profile: Mission profile
            
        Returns:
            Success status
        """
        if profile_id in self.mission_profiles:
            return False
        
        self.mission_profiles[profile_id] = profile
        return True
    
    def optimize_for_mission(self, profile_id: str, 
                            platform_constraints: Dict[str, Any]) -> OptimizationResult:
        """
        Optimize payload configuration for a specific mission profile.
        
        Args:
            profile_id: Identifier of the mission profile
            platform_constraints: Platform-specific constraints
            
        Returns:
            Optimization results
        """
        if profile_id not in self.mission_profiles:
            return OptimizationResult(
                recommended_payloads=[],
                payload_settings={},
                estimated_effectiveness=0.0,
                power_usage=0.0,
                weight_total=0.0,
                optimization_score=0.0
            )
        
        profile = self.mission_profiles[profile_id]
        
        # Create optimization constraints from mission profile and platform constraints
        constraints = OptimizationConstraints(
            max_weight=platform_constraints.get("max_weight", 1000.0),
            max_power=platform_constraints.get("max_power", 10000.0),
            priority_targets=profile.target_types,
            mission_type=profile.name,
            environmental_factors=profile.environmental_requirements
        )
        
        # Run optimization
        return self.optimizer.optimize_configuration(constraints)
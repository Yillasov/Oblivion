#!/usr/bin/env python3
"""
Payload Optimization System for UCAV platforms.

This module provides optimization capabilities for payload configurations
using neuromorphic computing to maximize mission effectiveness.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from src.payload.base import PayloadInterface, PayloadSpecs
from src.core.integration.neuromorphic_system import NeuromorphicSystem


@dataclass
class OptimizationConstraints:
    """Constraints for payload optimization."""
    max_weight: float  # Maximum total weight in kg
    max_power: float  # Maximum power consumption in watts
    priority_targets: List[str]  # Priority target types
    mission_type: str  # Type of mission
    environmental_factors: Dict[str, Any]  # Environmental factors


@dataclass
class OptimizationResult:
    """Results of payload optimization."""
    recommended_payloads: List[str]  # List of recommended payload IDs
    payload_settings: Dict[str, Dict[str, Any]]  # Settings for each payload
    estimated_effectiveness: float  # Estimated mission effectiveness (0-1)
    power_usage: float  # Estimated power usage in watts
    weight_total: float  # Total weight in kg
    optimization_score: float  # Overall optimization score (0-1)


class PayloadOptimizer:
    """
    Optimizes payload configurations for different mission scenarios.
    """
    
    def __init__(self, hardware_interface=None):
        """
        Initialize the payload optimizer.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
        """
        self.system = NeuromorphicSystem(hardware_interface)
        self.payloads: Dict[str, PayloadInterface] = {}
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the optimization system."""
        if self.system.initialize():
            self.initialized = True
            return True
        return False
    
    def register_payload(self, payload_id: str, payload: PayloadInterface) -> bool:
        """
        Register a payload with the optimizer.
        
        Args:
            payload_id: Unique identifier for the payload
            payload: Payload instance to register
            
        Returns:
            Success status
        """
        if payload_id in self.payloads:
            return False
        
        self.payloads[payload_id] = payload
        return True
    
    def optimize_configuration(self, constraints: OptimizationConstraints) -> OptimizationResult:
        """
        Optimize payload configuration based on given constraints.
        
        Args:
            constraints: Optimization constraints
            
        Returns:
            Optimization results
        """
        if not self.initialized or not self.payloads:
            return OptimizationResult(
                recommended_payloads=[],
                payload_settings={},
                estimated_effectiveness=0.0,
                power_usage=0.0,
                weight_total=0.0,
                optimization_score=0.0
            )
        
        # Collect payload specifications
        payload_specs = {}
        for pid, payload in self.payloads.items():
            specs = payload.get_specifications()
            if specs:
                payload_specs[pid] = specs
        
        # Use neuromorphic processing for optimization
        optimization_data = self._run_neuromorphic_optimization(payload_specs, constraints)
        
        # Process optimization results
        recommended_payloads = optimization_data.get("recommended_payloads", [])
        payload_settings = optimization_data.get("payload_settings", {})
        
        # Calculate totals
        power_usage = sum(
            payload_specs[pid].power_requirements 
            for pid in recommended_payloads 
            if pid in payload_specs
        )
        
        weight_total = sum(
            payload_specs[pid].weight 
            for pid in recommended_payloads 
            if pid in payload_specs
        )
        
        return OptimizationResult(
            recommended_payloads=recommended_payloads,
            payload_settings=payload_settings,
            estimated_effectiveness=optimization_data.get("effectiveness", 0.0),
            power_usage=power_usage,
            weight_total=weight_total,
            optimization_score=optimization_data.get("optimization_score", 0.0)
        )
    
    def optimize_for_target(self, target_data: Dict[str, Any], 
                           available_payloads: List[str]) -> Dict[str, Any]:
        """
        Optimize payload selection for a specific target.
        
        Args:
            target_data: Data about the target
            available_payloads: List of available payload IDs
            
        Returns:
            Optimization results
        """
        if not self.initialized:
            return {"error": "Not initialized"}
        
        # Filter to only available payloads
        payload_specs = {}
        for pid in available_payloads:
            if pid in self.payloads:
                specs = self.payloads[pid].get_specifications()
                if specs:
                    payload_specs[pid] = specs
        
        if not payload_specs:
            return {"error": "No valid payloads available"}
        
        # Use neuromorphic processing for target-specific optimization
        return self._run_target_optimization(payload_specs, target_data)
    
    def _run_neuromorphic_optimization(self, payload_specs: Dict[str, PayloadSpecs],
                                      constraints: OptimizationConstraints) -> Dict[str, Any]:
        """
        Run neuromorphic optimization for payload configuration.
        
        Args:
            payload_specs: Specifications for available payloads
            constraints: Optimization constraints
            
        Returns:
            Optimization results
        """
        if not self.initialized:
            return {"error": "Not initialized"}
        
        # Process data using neuromorphic hardware
        optimization_result = self.system.process_data({
            'payloads': payload_specs,
            'constraints': {
                'max_weight': constraints.max_weight,
                'max_power': constraints.max_power,
                'priority_targets': constraints.priority_targets,
                'mission_type': constraints.mission_type,
                'environmental_factors': constraints.environmental_factors
            },
            'computation': 'payload_configuration_optimization'
        })
        
        # If processing failed, return basic result
        if not optimization_result or "error" in optimization_result:
            return {
                "recommended_payloads": [],
                "payload_settings": {},
                "effectiveness": 0.0,
                "optimization_score": 0.0
            }
        
        return optimization_result
    
    def _run_target_optimization(self, payload_specs: Dict[str, PayloadSpecs],
                               target_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run neuromorphic optimization for target-specific payload selection.
        
        Args:
            payload_specs: Specifications for available payloads
            target_data: Data about the target
            
        Returns:
            Optimization results
        """
        if not self.initialized:
            return {"error": "Not initialized"}
        
        # Process data using neuromorphic hardware
        optimization_result = self.system.process_data({
            'payloads': payload_specs,
            'target': target_data,
            'computation': 'target_specific_optimization'
        })
        
        # If processing failed, return basic result
        if not optimization_result or "error" in optimization_result:
            return {
                "recommended_payloads": [],
                "effectiveness": 0.0
            }
        
        return optimization_result
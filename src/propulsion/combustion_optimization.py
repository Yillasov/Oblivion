"""
Combustion Optimization Algorithms for UCAV propulsion systems.
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

from src.propulsion.base import PropulsionInterface
from src.propulsion.optimization import OptimizationConstraints


@dataclass
class CombustionParameters:
    """Parameters for combustion optimization."""
    fuel_air_ratio: float  # Optimal fuel-air ratio
    ignition_timing: float  # Ignition timing in degrees or ms
    chamber_pressure: float  # Optimal chamber pressure in kPa
    inlet_temperature: float  # Inlet temperature in K
    exhaust_temperature: float  # Exhaust temperature in K
    combustion_efficiency: float  # Efficiency of combustion (0-1)


class CombustionOptimizer:
    """Simple combustion optimization system."""
    
    def __init__(self):
        """Initialize combustion optimizer."""
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.combustion_params: Dict[str, CombustionParameters] = {}
        self.optimization_history: Dict[str, List[Dict[str, float]]] = {}
        
    def register_system(self, system_id: str, system: PropulsionInterface) -> bool:
        """Register a propulsion system for combustion optimization."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        self.optimization_history[system_id] = []
        
        # Initialize with default parameters
        self.combustion_params[system_id] = CombustionParameters(
            fuel_air_ratio=0.067,  # Stoichiometric for typical hydrocarbon
            ignition_timing=0.0,
            chamber_pressure=1000.0,
            inlet_temperature=300.0,
            exhaust_temperature=900.0,
            combustion_efficiency=0.9
        )
        
        return True
        
    def optimize_combustion(self, 
                          system_id: str, 
                          flight_conditions: Dict[str, float],
                          constraints: OptimizationConstraints) -> Dict[str, Any]:
        """
        Optimize combustion parameters for given flight conditions.
        
        Args:
            system_id: ID of the propulsion system
            flight_conditions: Current flight conditions
            constraints: Optimization constraints
            
        Returns:
            Optimized combustion parameters and performance metrics
        """
        if system_id not in self.propulsion_systems:
            return {"success": False, "error": "System not found"}
            
        system = self.propulsion_systems[system_id]
        
        # Extract relevant flight conditions
        altitude = flight_conditions.get("altitude", 0.0)
        mach = flight_conditions.get("mach", 0.0)
        ambient_temp = flight_conditions.get("temperature", 288.15)
        
        # Adjust inlet temperature based on altitude and speed
        inlet_temp = ambient_temp * (1 + 0.2 * mach**2)
        
        # Optimize fuel-air ratio based on altitude
        # At higher altitudes, lean mixture for better efficiency
        base_far = 0.067  # Stoichiometric ratio
        altitude_factor = max(0.8, 1.0 - altitude / 40000)
        optimal_far = base_far * altitude_factor
        
        # Adjust ignition timing based on speed
        # Advance timing at higher speeds
        base_timing = 0.0
        if mach > 0.5:
            timing_adjustment = 5.0 * (mach - 0.5)
            optimal_timing = base_timing + timing_adjustment
        else:
            optimal_timing = base_timing
            
        # Calculate optimal chamber pressure
        # Higher at higher speeds for better compression
        base_pressure = 1000.0  # kPa
        pressure_factor = 1.0 + 0.5 * mach
        optimal_pressure = base_pressure * pressure_factor
        
        # Calculate combustion efficiency
        # Efficiency drops at very high altitudes
        base_efficiency = 0.9
        if altitude > 20000:
            efficiency_drop = 0.1 * (altitude - 20000) / 20000
            combustion_efficiency = base_efficiency - efficiency_drop
        else:
            combustion_efficiency = base_efficiency
            
        # Calculate exhaust temperature
        exhaust_temp = inlet_temp + 600.0 * combustion_efficiency
        
        # Create optimized parameters
        optimized_params = CombustionParameters(
            fuel_air_ratio=optimal_far,
            ignition_timing=optimal_timing,
            chamber_pressure=optimal_pressure,
            inlet_temperature=inlet_temp,
            exhaust_temperature=exhaust_temp,
            combustion_efficiency=combustion_efficiency
        )
        
        # Update system parameters
        self.combustion_params[system_id] = optimized_params
        
        # Calculate expected performance improvement
        thrust_improvement = self._calculate_thrust_improvement(
            system_id, optimized_params, flight_conditions
        )
        
        efficiency_improvement = self._calculate_efficiency_improvement(
            system_id, optimized_params, flight_conditions
        )
        
        # Record optimization results
        result = {
            "success": True,
            "parameters": {
                "fuel_air_ratio": optimal_far,
                "ignition_timing": optimal_timing,
                "chamber_pressure": optimal_pressure,
                "inlet_temperature": inlet_temp,
                "exhaust_temperature": exhaust_temp,
                "combustion_efficiency": combustion_efficiency
            },
            "improvement": {
                "thrust": thrust_improvement,
                "efficiency": efficiency_improvement
            }
        }
        
        self.optimization_history[system_id].append(result)
        return result
    
    def _calculate_thrust_improvement(self, 
                                    system_id: str, 
                                    params: CombustionParameters,
                                    flight_conditions: Dict[str, float]) -> float:
        """Calculate expected thrust improvement from optimized parameters."""
        # Simple model: improvement based on combustion efficiency and pressure
        base_improvement = (params.combustion_efficiency - 0.85) * 10.0
        pressure_factor = (params.chamber_pressure / 1000.0 - 1.0) * 5.0
        
        return base_improvement + pressure_factor
    
    def _calculate_efficiency_improvement(self, 
                                        system_id: str, 
                                        params: CombustionParameters,
                                        flight_conditions: Dict[str, float]) -> float:
        """Calculate expected efficiency improvement from optimized parameters."""
        # Simple model: improvement based on fuel-air ratio optimization
        stoichiometric = 0.067
        deviation = abs(params.fuel_air_ratio - stoichiometric)
        
        # Slight lean mixture (deviation up to 0.01) improves efficiency
        if params.fuel_air_ratio < stoichiometric and deviation <= 0.01:
            return deviation * 100.0  # Up to 1% improvement
        else:
            return -deviation * 50.0  # Efficiency loss for non-optimal mixture
    
    def apply_parameters(self, system_id: str) -> bool:
        """Apply optimized combustion parameters to the propulsion system."""
        if system_id not in self.propulsion_systems or system_id not in self.combustion_params:
            return False
            
        system = self.propulsion_systems[system_id]
        params = self.combustion_params[system_id]
        
        # Convert parameters to system-specific power state
        power_state = {
            "fuel_air_ratio": params.fuel_air_ratio,
            "ignition_timing": params.ignition_timing,
            "chamber_pressure": params.chamber_pressure
        }
        
        # Apply to system
        system.set_power_state(power_state)
        return True
"""
Power Distribution Optimization System for UCAV platforms.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.propulsion.hybrid_electric import HybridElectricController
from src.propulsion.solar_electric import SolarElectricSystem
from src.propulsion.hydrogen_fuel_cell import HydrogenFuelCellManager


@dataclass
class PowerSource:
    """Represents a power source in the system."""
    id: str
    type: str  # "electric", "combustion", "solar", "fuel_cell"
    max_output: float  # Maximum power output in kW
    current_output: float = 0.0  # Current power output in kW
    priority: int = 1  # Priority level (higher = more important)
    efficiency: float = 0.9  # Efficiency factor (0-1)


class PowerDistributionOptimizer:
    """Optimizes power distribution across multiple propulsion systems."""
    
    def __init__(self):
        """Initialize the power distribution optimizer."""
        self.power_sources: Dict[str, PowerSource] = {}
        self.hybrid_controller: Optional[HybridElectricController] = None
        self.solar_system: Optional[SolarElectricSystem] = None
        self.fuel_cell_manager: Optional[HydrogenFuelCellManager] = None
        self.total_power_demand = 0.0
        self.optimization_history: List[Dict[str, Any]] = []
        
    def register_hybrid_controller(self, controller: HybridElectricController) -> None:
        """Register a hybrid electric controller."""
        self.hybrid_controller = controller
        
        # Register electric and combustion power sources
        self.power_sources["electric_main"] = PowerSource(
            id="electric_main",
            type="electric",
            max_output=100.0,  # Default value, will be updated during optimization
            priority=2,
            efficiency=0.95
        )
        
        self.power_sources["combustion_main"] = PowerSource(
            id="combustion_main",
            type="combustion",
            max_output=200.0,  # Default value, will be updated during optimization
            priority=1,
            efficiency=0.35
        )
        
    def register_solar_system(self, solar_system: SolarElectricSystem) -> None:
        """Register a solar electric system."""
        self.solar_system = solar_system
        
        self.power_sources["solar"] = PowerSource(
            id="solar",
            type="solar",
            max_output=solar_system.solar_specs.max_output,
            priority=3,  # Highest priority as it's free energy
            efficiency=solar_system.solar_specs.efficiency
        )
        
    def register_fuel_cell(self, fuel_cell: HydrogenFuelCellManager) -> None:
        """Register a hydrogen fuel cell system."""
        self.fuel_cell_manager = fuel_cell
        
        self.power_sources["fuel_cell"] = PowerSource(
            id="fuel_cell",
            type="fuel_cell",
            max_output=fuel_cell.specs.power_output,
            priority=2,  # Same priority as electric
            efficiency=fuel_cell.specs.efficiency
        )
        
    def update_power_sources(self, flight_conditions: Dict[str, float]) -> None:
        """Update power source information based on current conditions."""
        if self.solar_system:
            solar_output = self.solar_system.calculate_solar_output(flight_conditions)
            self.power_sources["solar"].current_output = solar_output
            self.power_sources["solar"].max_output = solar_output  # Max is current for solar
            
        if self.fuel_cell_manager and self.fuel_cell_manager.operational:
            self.power_sources["fuel_cell"].current_output = self.fuel_cell_manager.current_power
            # Adjust max output based on hydrogen level
            self.power_sources["fuel_cell"].max_output = (
                self.fuel_cell_manager.specs.power_output * 
                self.fuel_cell_manager.hydrogen_level
            )
            
        if self.hybrid_controller:
            # Update electric and combustion based on battery and fuel levels
            self.power_sources["electric_main"].max_output = (
                100.0 * self.hybrid_controller.battery_level
            )
            self.power_sources["combustion_main"].max_output = (
                200.0 * self.hybrid_controller.fuel_level
            )
    
    def optimize_distribution(self, 
                            power_demand: float, 
                            flight_conditions: Dict[str, float],
                            optimization_mode: str = "efficiency") -> Dict[str, Any]:
        """
        Optimize power distribution across all sources.
        
        Args:
            power_demand: Total power demand in kW
            flight_conditions: Current flight conditions
            optimization_mode: "efficiency", "endurance", or "performance"
            
        Returns:
            Optimized power distribution
        """
        self.total_power_demand = power_demand
        self.update_power_sources(flight_conditions)
        
        # Sort power sources by priority (higher first) and then by efficiency if in efficiency mode
        if optimization_mode == "efficiency":
            sorted_sources = sorted(
                self.power_sources.values(),
                key=lambda x: (x.priority, x.efficiency),
                reverse=True
            )
        elif optimization_mode == "endurance":
            # For endurance, prioritize renewable sources
            sorted_sources = sorted(
                self.power_sources.values(),
                key=lambda x: (1 if x.type in ["solar", "fuel_cell"] else 0, x.priority),
                reverse=True
            )
        else:  # performance mode
            # For performance, prioritize by max output
            sorted_sources = sorted(
                self.power_sources.values(),
                key=lambda x: (x.priority, x.max_output),
                reverse=True
            )
        
        # Allocate power based on priority
        remaining_demand = power_demand
        allocation = {}
        
        for source in sorted_sources:
            if remaining_demand <= 0:
                allocation[source.id] = 0
                continue
                
            # Allocate up to max output or remaining demand
            allocated = min(source.max_output, remaining_demand)
            allocation[source.id] = allocated
            remaining_demand -= allocated
        
        # Calculate total allocated and efficiency
        total_allocated = sum(allocation.values())
        weighted_efficiency = sum(
            allocation[source.id] * source.efficiency 
            for source in self.power_sources.values() 
            if allocation.get(source.id, 0) > 0
        ) / total_allocated if total_allocated > 0 else 0
        
        # Record optimization result
        result = {
            "allocation": allocation,
            "total_demand": power_demand,
            "total_allocated": total_allocated,
            "demand_satisfied": total_allocated >= power_demand * 0.99,
            "weighted_efficiency": weighted_efficiency,
            "optimization_mode": optimization_mode
        }
        
        self.optimization_history.append(result)
        return result
    
    def apply_distribution(self, distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the optimized distribution to all systems."""
        allocation = distribution["allocation"]
        
        # Apply to hybrid controller
        if self.hybrid_controller:
            electric_power = allocation.get("electric_main", 0)
            combustion_power = allocation.get("combustion_main", 0)
            total_hybrid_power = electric_power + combustion_power
            
            if total_hybrid_power > 0:
                # Calculate electric ratio
                electric_ratio = electric_power / total_hybrid_power
                
                # Find closest mode based on electric ratio
                closest_mode = "balanced"
                min_diff = float('inf')
                
                for mode, config in self.hybrid_controller.modes.items():
                    diff = abs(config["electric_ratio"] - electric_ratio)
                    if diff < min_diff:
                        min_diff = diff
                        closest_mode = mode
                
                # Set mode and update power distribution
                self.hybrid_controller.set_mode(closest_mode)
        
        # Apply to solar system
        if self.solar_system and "solar" in allocation:
            # Solar is passive, so we don't need to apply anything
            pass
        
        # Apply to fuel cell
        if self.fuel_cell_manager and "fuel_cell" in allocation:
            fuel_cell_power = allocation.get("fuel_cell", 0)
            
            # Start or update fuel cell
            if fuel_cell_power > 0 and not self.fuel_cell_manager.operational:
                self.fuel_cell_manager.start_system()
            elif fuel_cell_power == 0 and self.fuel_cell_manager.operational:
                self.fuel_cell_manager.shutdown()
        
        return {
            "success": True,
            "applied_distribution": allocation
        }
    
    def get_recommended_mode(self, flight_conditions: Dict[str, float]) -> str:
        """Get recommended optimization mode based on flight conditions."""
        altitude = flight_conditions.get("altitude", 0)
        speed = flight_conditions.get("speed", 0)
        remaining_distance = flight_conditions.get("remaining_distance", 0)
        
        # Simple rule-based recommendation
        if remaining_distance > 1000:
            return "endurance"  # Prioritize endurance for long distances
        elif speed > 250 or altitude > 8000:
            return "performance"  # Prioritize performance for high speed/altitude
        else:
            return "efficiency"  # Default to efficiency
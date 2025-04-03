import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.core.utils.logging_framework import get_logger

logger = get_logger("cost_modeling")

class CostCategory(Enum):
    LABOR = "labor"
    MATERIALS = "materials"
    EQUIPMENT = "equipment"
    OVERHEAD = "overhead"
    QUALITY_CONTROL = "quality_control"

@dataclass
class ResourceCost:
    hourly_rate: float
    setup_cost: float = 0.0
    maintenance_cost: float = 0.0

class CostModel:
    def __init__(self):
        self.resource_costs: Dict[str, ResourceCost] = {
            "assembly_station": ResourceCost(hourly_rate=150.0, setup_cost=200.0),
            "composite_layup": ResourceCost(hourly_rate=200.0, setup_cost=300.0),
            "cnc_machine": ResourceCost(hourly_rate=180.0, setup_cost=250.0),
            "quality_control": ResourceCost(hourly_rate=120.0),
        }
        
        self.material_costs: Dict[str, float] = {
            "carbon_fiber": 500.0,  # per kg
            "aluminum": 150.0,      # per kg
            "composite_resin": 80.0 # per liter
        }
        
        self.task_costs: Dict[str, Dict[str, float]] = {}
    
    def calculate_task_cost(self, task_id: str, 
                          resources_used: Dict[str, int],
                          duration: float,
                          materials_used: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate total cost for a task."""
        costs = {category.value: 0.0 for category in CostCategory}
        
        # Calculate resource costs
        for resource, quantity in resources_used.items():
            if resource in self.resource_costs:
                resource_cost = self.resource_costs[resource]
                costs[CostCategory.EQUIPMENT.value] += resource_cost.setup_cost * quantity
                costs[CostCategory.LABOR.value] += (
                    resource_cost.hourly_rate * duration * quantity
                )
        
        # Calculate material costs
        if materials_used:
            for material, amount in materials_used.items():
                if material in self.material_costs:
                    costs[CostCategory.MATERIALS.value] += (
                        self.material_costs[material] * amount
                    )
        
        # Add overhead cost (20% of total direct costs)
        direct_costs = sum(v for k, v in costs.items() if k != CostCategory.OVERHEAD.value)
        costs[CostCategory.OVERHEAD.value] = direct_costs * 0.20
        
        # Calculate total
        costs["total"] = sum(costs.values())
        
        # Store task costs
        self.task_costs[task_id] = costs
        
        return costs
    
    def get_task_cost_summary(self, task_id: str) -> Dict[str, float]:
        """Get cost summary for a specific task."""
        return self.task_costs.get(task_id, {})
    
    def get_total_production_cost(self) -> Dict[str, float]:
        """Get total production costs across all tasks."""
        total_costs = {category.value: 0.0 for category in CostCategory}
        total_costs["total"] = 0.0
        
        for task_costs in self.task_costs.values():
            for category, cost in task_costs.items():
                total_costs[category] += cost
        
        return total_costs
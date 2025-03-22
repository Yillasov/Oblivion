from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import heapq

from src.core.utils.logging_framework import get_logger
from typing import Dict, List, Any
from src.manufacturing.workflow.production_automator import ProductionAutomator

logger = get_logger("production_scheduler")

class ProductionPriority(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1

@dataclass
class ProductionTask:
    task_id: str
    description: str
    priority: ProductionPriority
    estimated_duration: float  # hours
    dependencies: List[str]
    resources_required: Dict[str, int]
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None

class ProductionScheduler:
    def __init__(self):
        self.tasks: Dict[str, ProductionTask] = {}
        self.resource_pool: Dict[str, int] = {
            "cnc_machine": 2,
            "composite_layup": 3,
            "assembly_station": 4,
            "quality_control": 2
        }
        self.production_automator = ProductionAutomator()
        self.schedule: List[ProductionTask] = []
    
    def add_task(self, task: ProductionTask) -> bool:
        """Add a production task to the scheduler."""
        if task.task_id in self.tasks:
            return False
            
        if not self._validate_resources(task.resources_required):
            return False
            
        self.tasks[task.task_id] = task
        self._update_schedule()
        return True
    
    def _validate_resources(self, required_resources: Dict[str, int]) -> bool:
        """Validate if required resources are available."""
        for resource, amount in required_resources.items():
            if resource not in self.resource_pool:
                return False
            if amount > self.resource_pool[resource]:
                return False
        return True
    
    def _update_schedule(self) -> None:
        """Update the production schedule based on priorities and dependencies."""
        # Reset schedule
        self.schedule = []
        unscheduled = list(self.tasks.values())
        
        # Sort by priority and dependencies
        while unscheduled:
            available = [
                task for task in unscheduled
                if all(dep not in self.tasks or 
                      self.tasks[dep].completion_time is not None 
                      for dep in task.dependencies)
            ]
            
            if not available:
                break
                
            # Select highest priority task
            next_task = max(available, key=lambda x: x.priority.value)
            
            # Calculate start time
            start_time = self._calculate_start_time(next_task)
            next_task.start_time = start_time
            next_task.completion_time = start_time + timedelta(hours=next_task.estimated_duration)
            
            self.schedule.append(next_task)
            unscheduled.remove(next_task)
    
    def _calculate_start_time(self, task: ProductionTask) -> datetime:
        """Calculate the earliest possible start time for a task."""
        current_time = datetime.now()
        
        # Consider dependencies
        dep_completion_times = [
            self.tasks[dep].completion_time 
            for dep in task.dependencies 
            if dep in self.tasks and self.tasks[dep].completion_time is not None
        ]
        
        if dep_completion_times:
            # Filter out None values and find the maximum completion time
            latest_completion = max(
                time for time in dep_completion_times if time is not None
            )
            current_time = max(current_time, latest_completion)
            
        return current_time
    
    def get_schedule(self) -> List[Dict[str, Any]]:
        """Get the current production schedule."""
        return [
            {
                "task_id": task.task_id,
                "description": task.description,
                "priority": task.priority.value,
                "start_time": task.start_time.isoformat() if task.start_time else None,
                "completion_time": task.completion_time.isoformat() if task.completion_time else None,
                "duration": task.estimated_duration,
                "resources": task.resources_required
            }
            for task in self.schedule
        ]
    
    def optimize_schedule(self) -> None:
        """Optimize the current schedule for resource utilization."""
        # Simple optimization: try to parallelize non-dependent tasks
        for i, task in enumerate(self.schedule):
            if i == 0:
                continue
                
            # Try to move task earlier if resources allow
            for j in range(i-1, -1, -1):
                earlier_task = self.schedule[j]
                if (not set(task.dependencies) & set(earlier_task.dependencies) and
                    self._can_run_parallel(task, earlier_task)):
                    task.start_time = earlier_task.start_time
                    break
    
    def _can_run_parallel(self, task1: ProductionTask, task2: ProductionTask) -> bool:
        """Check if two tasks can run in parallel based on resource constraints."""
        combined_resources = {}
        for resource, amount in task1.resources_required.items():
            combined_resources[resource] = amount
            
        for resource, amount in task2.resources_required.items():
            if resource in combined_resources:
                combined_resources[resource] += amount
            else:
                combined_resources[resource] = amount
                
        return self._validate_resources(combined_resources)
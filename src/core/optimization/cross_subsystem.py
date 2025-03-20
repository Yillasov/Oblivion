"""
Cross-subsystem optimization framework for coordinating resource usage and performance.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import threading
import time

logger = logging.getLogger(__name__)

class OptimizationPriority:
    """Priority levels for optimization tasks."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class SubsystemOptimizer:
    """Base class for subsystem-specific optimizers."""
    
    def __init__(self, subsystem_id: str):
        """Initialize subsystem optimizer."""
        self.subsystem_id = subsystem_id
        self.enabled = True
        
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize subsystem based on context.
        
        Args:
            context: Optimization context with system-wide information
            
        Returns:
            Optimization results
        """
        raise NotImplementedError("Subclasses must implement optimize()")
    
    def get_resource_requirements(self) -> Dict[str, float]:
        """Get resource requirements for this subsystem."""
        return {}


class CrossSubsystemOptimizer:
    """
    Coordinates optimization across multiple subsystems.
    """
    
    def __init__(self, optimization_interval: float = 5.0):
        """
        Initialize cross-subsystem optimizer.
        
        Args:
            optimization_interval: Seconds between optimization runs
        """
        self.optimizers: Dict[str, SubsystemOptimizer] = {}
        self.optimization_interval = optimization_interval
        self.running = False
        self.optimization_thread = None
        self.context: Dict[str, Any] = {
            "resources": {},
            "priorities": {},
            "performance": {},
            "constraints": {}
        }
        self.lock = threading.RLock()
        self.callbacks: Dict[str, List[Callable]] = {}
        
    def register_optimizer(self, optimizer: SubsystemOptimizer) -> None:
        """Register a subsystem optimizer."""
        with self.lock:
            self.optimizers[optimizer.subsystem_id] = optimizer
            logger.info(f"Registered optimizer for subsystem: {optimizer.subsystem_id}")
    
    def set_resource_constraints(self, constraints: Dict[str, float]) -> None:
        """Set global resource constraints."""
        with self.lock:
            self.context["constraints"] = constraints
    
    def update_context(self, key: str, data: Any) -> None:
        """Update optimization context."""
        with self.lock:
            self.context[key] = data
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for optimization events."""
        with self.lock:
            if event not in self.callbacks:
                self.callbacks[event] = []
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, data: Any) -> None:
        """Trigger callbacks for an event."""
        callbacks = self.callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback for event {event}: {str(e)}")
    
    def start(self) -> bool:
        """Start optimization thread."""
        if self.running:
            return False
            
        self.running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        logger.info("Cross-subsystem optimization started")
        return True
    
    def stop(self) -> None:
        """Stop optimization thread."""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=2.0)
        logger.info("Cross-subsystem optimization stopped")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self.running:
            try:
                self._run_optimization_cycle()
            except Exception as e:
                logger.error(f"Error in optimization cycle: {str(e)}")
            
            time.sleep(self.optimization_interval)
    
    def _run_optimization_cycle(self) -> None:
        """Run a single optimization cycle."""
        with self.lock:
            # Collect resource requirements
            requirements = {}
            for subsystem_id, optimizer in self.optimizers.items():
                if optimizer.enabled:
                    requirements[subsystem_id] = optimizer.get_resource_requirements()
            
            # Allocate resources based on priorities
            allocations = self._allocate_resources(requirements)
            self.context["allocations"] = allocations
            
            # Run optimizers in priority order
            results = {}
            for priority in range(OptimizationPriority.CRITICAL, -1, -1):
                for subsystem_id, optimizer in self.optimizers.items():
                    if not optimizer.enabled:
                        continue
                        
                    subsystem_priority = self.context.get("priorities", {}).get(
                        subsystem_id, OptimizationPriority.MEDIUM)
                    
                    if subsystem_priority == priority:
                        # Run optimization for this subsystem
                        results[subsystem_id] = optimizer.optimize(self.context)
            
            # Update performance metrics
            self.context["performance"] = {
                subsystem_id: result.get("performance", {})
                for subsystem_id, result in results.items()
            }
            
            # Trigger callbacks
            self._trigger_callbacks("optimization_complete", {
                "results": results,
                "allocations": allocations
            })
    
    def _allocate_resources(self, requirements: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Allocate resources based on requirements and priorities.
        
        Args:
            requirements: Resource requirements by subsystem
            
        Returns:
            Resource allocations by subsystem
        """
        allocations = {subsystem_id: {} for subsystem_id in requirements}
        constraints = self.context.get("constraints", {})
        priorities = self.context.get("priorities", {})
        
        # Simple allocation strategy - prioritize critical subsystems
        for resource_type, limit in constraints.items():
            # Calculate total requested resources
            total_requested = sum(
                req.get(resource_type, 0) 
                for req in requirements.values()
            )
            
            # If total requested is within limits, allocate as requested
            if total_requested <= limit:
                for subsystem_id, req in requirements.items():
                    allocations[subsystem_id][resource_type] = req.get(resource_type, 0)
            else:
                # Allocate based on priority
                remaining = limit
                
                # First pass: allocate to critical subsystems
                for subsystem_id, req in requirements.items():
                    if priorities.get(subsystem_id, OptimizationPriority.MEDIUM) == OptimizationPriority.CRITICAL:
                        allocation = min(req.get(resource_type, 0), remaining)
                        allocations[subsystem_id][resource_type] = allocation
                        remaining -= allocation
                
                # Second pass: allocate to high priority subsystems
                if remaining > 0:
                    for subsystem_id, req in requirements.items():
                        if priorities.get(subsystem_id, OptimizationPriority.MEDIUM) == OptimizationPriority.HIGH:
                            allocation = min(req.get(resource_type, 0), remaining)
                            allocations[subsystem_id][resource_type] = allocation
                            remaining -= allocation
                
                # Third pass: allocate proportionally to remaining subsystems
                if remaining > 0:
                    lower_priority_subsystems = [
                        subsystem_id for subsystem_id in requirements
                        if priorities.get(subsystem_id, OptimizationPriority.MEDIUM) < OptimizationPriority.HIGH
                    ]
                    
                    if lower_priority_subsystems:
                        for subsystem_id in lower_priority_subsystems:
                            req = requirements[subsystem_id].get(resource_type, 0)
                            allocation = (req / total_requested) * remaining
                            allocations[subsystem_id][resource_type] = allocation
        
        return allocations
#!/usr/bin/env python3
"""
Real-time Adaptive Optimization System

Provides a framework for adapting hardware and system configurations
based on real-time performance metrics.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
import time
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.core.utils.logging_framework import get_logger

logger = get_logger("adaptive_optimizer")

class OptimizationTarget(Enum):
    """Optimization targets for the adaptive system."""
    PERFORMANCE = "performance"
    POWER_EFFICIENCY = "power_efficiency"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RELIABILITY = "reliability"
    THERMAL = "thermal"
    BALANCED = "balanced"


@dataclass
class AdaptiveOptimizationConfig:
    """Configuration for adaptive optimization."""
    learning_rate: float = 0.05  # Learning rate for parameter updates
    exploration_rate: float = 0.1  # Rate of exploration vs exploitation
    memory_size: int = 20  # Size of memory buffer for past optimizations
    adaptation_threshold: float = 0.02  # Minimum performance change to trigger adaptation
    update_interval: float = 1.0  # Seconds between optimization updates
    max_iterations: int = 100  # Maximum optimization iterations
    target: OptimizationTarget = OptimizationTarget.BALANCED  # Optimization target
    metrics_weights: Dict[str, float] = field(default_factory=lambda: {
        "performance": 1.0,
        "power_efficiency": 1.0,
        "latency": 1.0,
        "throughput": 1.0,
        "reliability": 1.0,
        "thermal": 1.0
    })


class AdaptiveRealtimeOptimizer:
    """
    Adaptive optimization system that learns and improves in real-time
    based on performance metrics.
    """
    
    def __init__(self, config: AdaptiveOptimizationConfig):
        """Initialize adaptive optimizer."""
        self.config = config
        self.systems: Dict[str, Any] = {}
        self.optimization_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.learned_parameters: Dict[str, Dict[str, float]] = {}
        self.performance_history: Dict[str, List[Dict[str, float]]] = {}
        self.last_update_time: Dict[str, float] = {}
        self.metric_providers: Dict[str, Callable[[], Dict[str, float]]] = {}
        self.parameter_updaters: Dict[str, Callable[[Dict[str, float]], None]] = {}
        
    def register_system(self, 
                        system_id: str, 
                        system: Any, 
                        initial_parameters: Dict[str, float],
                        metric_provider: Callable[[], Dict[str, float]],
                        parameter_updater: Callable[[Dict[str, float]], None]) -> bool:
        """
        Register a system for adaptive optimization.
        
        Args:
            system_id: Unique identifier for the system
            system: The system object
            initial_parameters: Initial optimization parameters
            metric_provider: Function that returns current performance metrics
            parameter_updater: Function that applies updated parameters to the system
            
        Returns:
            bool: Success status
        """
        if system_id in self.systems:
            return False
            
        self.systems[system_id] = system
        self.optimization_memory[system_id] = []
        self.performance_history[system_id] = []
        self.last_update_time[system_id] = time.time()
        self.metric_providers[system_id] = metric_provider
        self.parameter_updaters[system_id] = parameter_updater
        
        # Initialize learned parameters
        self.learned_parameters[system_id] = initial_parameters.copy()
        
        return True
        
    def update(self, system_id: str) -> Dict[str, Any]:
        """
        Update optimization for a system based on real-time metrics.
        
        Args:
            system_id: System identifier
            
        Returns:
            Dict[str, Any]: Optimization result
        """
        if system_id not in self.systems:
            return {"success": False, "error": "System not found"}
            
        # Check if it's time to update
        current_time = time.time()
        if current_time - self.last_update_time.get(system_id, 0) < self.config.update_interval:
            return {"success": True, "status": "skipped", "reason": "Update interval not reached"}
            
        self.last_update_time[system_id] = current_time
        
        # Get current metrics
        try:
            current_metrics = self.metric_providers[system_id]()
        except Exception as e:
            logger.error(f"Failed to get metrics for {system_id}: {str(e)}")
            return {"success": False, "error": f"Failed to get metrics: {str(e)}"}
        
        # Store metrics in history
        self.performance_history[system_id].append(current_metrics)
        if len(self.performance_history[system_id]) > self.config.memory_size:
            self.performance_history[system_id].pop(0)
        
        # Calculate performance score
        score = self._calculate_performance_score(system_id, current_metrics)
        
        # Determine if adaptation is needed
        needs_adaptation = self._check_adaptation_needed(system_id, score)
        
        if not needs_adaptation:
            return {
                "success": True, 
                "status": "unchanged", 
                "score": score,
                "metrics": current_metrics,
                "parameters": self.learned_parameters[system_id]
            }
        
        # Update parameters based on performance
        updated_params = self._update_parameters(system_id, current_metrics, score)
        
        # Apply updated parameters
        try:
            self.parameter_updaters[system_id](updated_params)
        except Exception as e:
            logger.error(f"Failed to update parameters for {system_id}: {str(e)}")
            return {"success": False, "error": f"Failed to update parameters: {str(e)}"}
        
        # Store optimization result in memory
        self._update_memory(system_id, current_metrics, updated_params, score)
        
        return {
            "success": True,
            "status": "updated",
            "score": score,
            "metrics": current_metrics,
            "parameters": updated_params,
            "improvement": self._calculate_improvement(system_id)
        }
    
    def _calculate_performance_score(self, system_id: str, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score from metrics."""
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in self.config.metrics_weights:
                weight = self.config.metrics_weights[metric]
                score += value * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
            
        return score / total_weight
    
    def _check_adaptation_needed(self, system_id: str, current_score: float) -> bool:
        """Check if adaptation is needed based on performance history."""
        history = self.performance_history[system_id]
        
        # Always adapt if we have limited history
        if len(history) < 2:
            return True
            
        # Calculate previous score
        prev_metrics = history[-2]
        prev_score = self._calculate_performance_score(system_id, prev_metrics)
        
        # Check if change exceeds threshold
        score_change = abs(current_score - prev_score)
        return score_change > self.config.adaptation_threshold
    
    def _update_parameters(self, 
                          system_id: str, 
                          metrics: Dict[str, float],
                          score: float) -> Dict[str, float]:
        """Update learned parameters based on performance metrics."""
        params = self.learned_parameters[system_id].copy()
        
        # Get performance history
        history = self.performance_history[system_id]
        
        # Calculate performance change
        if len(history) > 1:
            prev_metrics = history[-2]
            prev_score = self._calculate_performance_score(system_id, prev_metrics)
            improvement = score - prev_score
        else:
            improvement = 0.0
        
        # Apply learning rate
        lr = self.config.learning_rate
        
        # Decide whether to explore or exploit
        if np.random.random() < self.config.exploration_rate:
            # Exploration: make random adjustments
            for param in params:
                # Random adjustment within ±10% of current value
                adjustment = np.random.uniform(-0.1, 0.1) * params[param]
                params[param] += adjustment
        else:
            # Exploitation: adjust based on performance improvement
            if improvement != 0:
                # For each parameter, adjust in the direction that improved performance
                for param in params:
                    # Get parameter history
                    param_history = [
                        entry["parameters"].get(param, params[param]) 
                        for entry in self.optimization_memory.get(system_id, [])[-3:]
                    ]
                    
                    if len(param_history) > 1:
                        # Calculate parameter change direction
                        param_change = param_history[-1] - param_history[0]
                        
                        # If parameter change correlates with improvement, continue in that direction
                        if (param_change * improvement) > 0:
                            # Continue in same direction
                            params[param] += lr * param_change
                        else:
                            # Reverse direction
                            params[param] -= lr * 0.5 * param_change
        
        # Apply constraints to parameters (ensure they stay within reasonable bounds)
        for param in params:
            # Get original value from first registration
            original = list(self.learned_parameters[system_id].values())[0] if self.learned_parameters[system_id] else 1.0
            
            # Limit to 50% change from original in either direction
            params[param] = max(original * 0.5, min(original * 1.5, params[param]))
        
        # Update learned parameters
        self.learned_parameters[system_id] = params
        
        return params
    
    def _update_memory(self, 
                      system_id: str, 
                      metrics: Dict[str, float],
                      parameters: Dict[str, float],
                      score: float) -> None:
        """Update optimization memory with new result."""
        memory = self.optimization_memory[system_id]
        
        # Create memory entry
        entry = {
            "timestamp": time.time(),
            "metrics": metrics.copy(),
            "parameters": parameters.copy(),
            "score": score
        }
        
        # Add to memory
        memory.append(entry)
        
        # Limit memory size
        if len(memory) > self.config.memory_size:
            memory.pop(0)
    
    def _calculate_improvement(self, system_id: str) -> Dict[str, float]:
        """Calculate improvement metrics over time."""
        history = self.performance_history[system_id]
        
        if len(history) < 2:
            return {"overall": 0.0}
            
        # Calculate improvements for each metric
        improvements = {}
        first_metrics = history[0]
        last_metrics = history[-1]
        
        for metric in last_metrics:
            if metric in first_metrics:
                improvements[metric] = last_metrics[metric] - first_metrics[metric]
        
        # Calculate overall improvement
        first_score = self._calculate_performance_score(system_id, first_metrics)
        last_score = self._calculate_performance_score(system_id, last_metrics)
        improvements["overall"] = last_score - first_score
        
        return improvements
    
    def get_optimization_stats(self, system_id: str) -> Dict[str, Any]:
        """Get optimization statistics for a system."""
        if system_id not in self.systems:
            return {"success": False, "error": "System not found"}
            
        history = self.performance_history[system_id]
        
        if not history:
            return {"success": True, "stats": {}, "parameters": {}}
            
        # Calculate statistics
        metrics_stats = {}
        
        # For each metric, calculate statistics
        for metric in history[0].keys():
            values = [entry.get(metric, 0.0) for entry in history]
            metrics_stats[metric] = {
                "current": values[-1] if values else 0.0,
                "average": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "improvement": values[-1] - values[0] if len(values) > 1 else 0.0
            }
        
        # Calculate overall score statistics
        scores = [self._calculate_performance_score(system_id, entry) for entry in history]
        
        return {
            "success": True,
            "stats": {
                "metrics": metrics_stats,
                "overall_score": {
                    "current": scores[-1] if scores else 0.0,
                    "average": sum(scores) / len(scores) if scores else 0.0,
                    "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0.0
                },
                "optimization_count": len(history)
            },
            "parameters": self.learned_parameters[system_id],
            "update_frequency": 1.0 / self.config.update_interval if self.config.update_interval > 0 else 0.0
        }
    
    def reset(self, system_id: str, keep_learning: bool = False) -> bool:
        """
        Reset optimization history for a system.
        
        Args:
            system_id: System identifier
            keep_learning: Whether to keep learned parameters
            
        Returns:
            bool: Success status
        """
        if system_id not in self.systems:
            return False
            
        self.optimization_memory[system_id] = []
        self.performance_history[system_id] = []
        
        if not keep_learning:
            # Reset learned parameters to initial values
            initial_params = next(iter(self.optimization_memory.get(system_id, [{}])), {}).get("parameters", {})
            if initial_params:
                self.learned_parameters[system_id] = initial_params.copy()
        
        return True
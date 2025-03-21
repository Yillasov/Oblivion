"""
Evolutionary techniques for adaptive targeting in dynamic environments.
Enables real-time adaptation to changing battlefield conditions.
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Any, Optional
import logging

from src.targeting.genetic_targeting import GeneticTargetingOptimizer

logger = logging.getLogger(__name__)

class AdaptiveTargetingEvolution:
    """
    Implements evolutionary techniques for targeting in dynamic environments.
    Adapts targeting strategies in real-time based on environmental changes.
    """
    
    def __init__(self, base_optimizer: GeneticTargetingOptimizer, 
                adaptation_rate: float = 0.2,
                environment_memory: int = 5):
        self.base_optimizer = base_optimizer
        self.adaptation_rate = adaptation_rate
        self.environment_memory = environment_memory
        self.environment_history: List[Dict[str, Any]] = []
        self.strategy_history: List[Dict[str, float]] = []
        self.performance_history: List[float] = []
        self.current_strategy: Optional[Dict[str, float]] = None
        
    def detect_environment_change(self, new_environment: Dict[str, Any]) -> float:
        """
        Detect how much the environment has changed.
        
        Args:
            new_environment: Current environment data
            
        Returns:
            Change magnitude (0-1)
        """
        if not self.environment_history:
            return 1.0  # First environment, maximum change
            
        # Compare with most recent environment
        last_env = self.environment_history[-1]
        
        # Calculate change metrics
        changes = []
        
        # Compare threat levels
        if "threat_level" in new_environment and "threat_level" in last_env:
            changes.append(abs(new_environment["threat_level"] - last_env["threat_level"]))
            
        # Compare target mobility
        if "target_mobility" in new_environment and "target_mobility" in last_env:
            changes.append(abs(new_environment["target_mobility"] - last_env["target_mobility"]))
            
        # Compare environmental conditions
        if "visibility" in new_environment and "visibility" in last_env:
            changes.append(abs(new_environment["visibility"] - last_env["visibility"]))
            
        # Return average change
        return min(1.0, sum(changes) / max(1, len(changes)))
        
    def adapt_to_environment(self, environment: Dict[str, Any]) -> Dict[str, float]:
        """
        Adapt targeting strategy to current environment.
        
        Args:
            environment: Current environment data
            
        Returns:
            Adapted targeting strategy
        """
        # Store environment
        self.environment_history.append(environment.copy())
        if len(self.environment_history) > self.environment_memory:
            self.environment_history.pop(0)
            
        # Detect environment change
        change_magnitude = self.detect_environment_change(environment)
        
        # If significant change or no current strategy, generate new strategy
        if change_magnitude > self.adaptation_rate or self.current_strategy is None:
            logger.info(f"Environment change detected ({change_magnitude:.2f}). Adapting strategy.")
            
            # Generate new strategy using genetic algorithm
            self.current_strategy = self.base_optimizer.optimize_targeting(environment)
            
        else:
            # Minor adaptation to current strategy
            self.current_strategy = self._refine_strategy(self.current_strategy, environment, change_magnitude)
            
        # Store strategy
        self.strategy_history.append(self.current_strategy.copy())
        if len(self.strategy_history) > self.environment_memory:
            self.strategy_history.pop(0)
            
        return self.current_strategy
    
    def _refine_strategy(self, strategy: Dict[str, float], 
                        environment: Dict[str, Any],
                        change_magnitude: float) -> Dict[str, float]:
        """Refine existing strategy with small adaptations."""
        refined = strategy.copy()
        
        # Adjust parameters based on environment
        for param in refined:
            # Small random adjustment proportional to change magnitude
            adjustment = random.uniform(-0.1, 0.1) * change_magnitude
            refined[param] = max(0.1, min(1.0, refined[param] + adjustment))
            
        # Specific adaptations based on environment factors
        if "visibility" in environment:
            visibility = environment["visibility"]
            # In low visibility, increase accuracy weight
            if visibility < 0.5 and "accuracy" in refined:
                refined["accuracy"] = min(1.0, refined["accuracy"] + 0.1 * (1 - visibility))
                
        if "target_mobility" in environment:
            mobility = environment["target_mobility"]
            # For highly mobile targets, increase speed weight
            if mobility > 0.7 and "speed" in refined:
                refined["speed"] = min(1.0, refined["speed"] + 0.1 * mobility)
                
        return refined
        
    def record_performance(self, performance_score: float) -> None:
        """
        Record performance of current strategy.
        
        Args:
            performance_score: Performance metric (higher is better)
        """
        self.performance_history.append(performance_score)
        if len(self.performance_history) > self.environment_memory:
            self.performance_history.pop(0)
            
    def get_best_strategy_for_environment(self, environment: Dict[str, Any]) -> Dict[str, float]:
        """
        Get best historical strategy for similar environment.
        
        Args:
            environment: Target environment
            
        Returns:
            Best matching strategy or None
        """
        if not self.environment_history:
            return self.base_optimizer.optimize_targeting(environment)
            
        # Find most similar environment
        similarities = []
        for i, past_env in enumerate(self.environment_history):
            # Skip if no performance data
            if i >= len(self.performance_history):
                continue
                
            # Calculate similarity
            sim_score = self._calculate_environment_similarity(environment, past_env)
            similarities.append((i, sim_score, self.performance_history[i]))
            
        # Sort by similarity and performance
        similarities.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Return strategy for most similar environment with good performance
        if similarities:
            best_idx = similarities[0][0]
            if best_idx < len(self.strategy_history):
                return self.strategy_history[best_idx]
                
        # Fallback to new optimization
        return self.base_optimizer.optimize_targeting(environment)
    
    def _calculate_environment_similarity(self, env1: Dict[str, Any], 
                                         env2: Dict[str, Any]) -> float:
        """Calculate similarity between environments (0-1)."""
        common_keys = set(env1.keys()) & set(env2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            if isinstance(env1[key], (int, float)) and isinstance(env2[key], (int, float)):
                similarities.append(1.0 - min(1.0, abs(env1[key] - env2[key])))
                
        return sum(similarities) / max(1, len(similarities))


# Example usage
if __name__ == "__main__":
    # Create base genetic optimizer
    base_optimizer = GeneticTargetingOptimizer(population_size=30, generations=50)
    
    # Create adaptive targeting evolution
    adaptive_targeting = AdaptiveTargetingEvolution(base_optimizer)
    
    # Example dynamic environment
    environments = [
        {"threat_level": 0.3, "target_mobility": 0.2, "visibility": 0.9, "target_type": "ground"},
        {"threat_level": 0.7, "target_mobility": 0.8, "visibility": 0.6, "target_type": "air"},
        {"threat_level": 0.5, "target_mobility": 0.4, "visibility": 0.3, "target_type": "ground"}
    ]
    
    # Adapt to changing environments
    for i, env in enumerate(environments):
        logger.info(f"Environment {i+1}: {env}")
        
        # Adapt strategy to environment
        strategy = adaptive_targeting.adapt_to_environment(env)
        
        # Simulate performance (would be real performance in actual system)
        performance = 0.7 + random.uniform(-0.2, 0.2)
        adaptive_targeting.record_performance(performance)
        
        logger.info(f"Adapted strategy: {strategy}")
        logger.info(f"Performance: {performance:.2f}")
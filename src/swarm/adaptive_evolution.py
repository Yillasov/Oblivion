#!/usr/bin/env python3
"""
Self-evolving code system for UCAV software adaptation.
Allows software to adapt and evolve based on mission requirements and performance.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
import time
import random
from dataclasses import dataclass

from src.swarm.neuromorphic_ai import NeuromorphicAI

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveParameter:
    """Parameter that can be evolved during operation."""
    name: str
    value: float
    min_value: float
    max_value: float
    mutation_rate: float = 0.1
    
    def mutate(self) -> None:
        """Apply random mutation to parameter."""
        mutation = np.random.normal(0, self.mutation_rate)
        self.value = np.clip(self.value + mutation, self.min_value, self.max_value)

@dataclass
class EvolutionMetrics:
    """Metrics for evaluating evolutionary performance."""
    fitness: float = 0.0
    generation: int = 0
    adaptation_count: int = 0
    last_mutation_time: float = 0.0
    improvement_rate: float = 0.0

class AdaptiveEvolution:
    """
    Self-evolving system that adapts UCAV software based on mission requirements.
    """
    
    def __init__(self, neuromorphic_ai: NeuromorphicAI):
        self.neuromorphic_ai = neuromorphic_ai
        self.parameters: Dict[str, AdaptiveParameter] = {}
        self.metrics = EvolutionMetrics()
        self.fitness_history: List[float] = []
        self.adaptation_enabled = True
        self.fitness_function: Optional[Callable[[Dict[str, float]], float]] = None
        
        # Initialize with default adaptive parameters
        self._initialize_default_parameters()
    
    def _initialize_default_parameters(self) -> None:
        """Initialize default adaptive parameters."""
        # Decision thresholds
        for action, value in self.neuromorphic_ai.decision_thresholds.items():
            self.parameters[f"threshold_{action}"] = AdaptiveParameter(
                name=f"threshold_{action}",
                value=value,
                min_value=0.1,
                max_value=0.9,
                mutation_rate=0.05
            )
        
        # Add neuron parameters
        self.parameters["neuron_decay"] = AdaptiveParameter(
            name="neuron_decay",
            value=0.9,
            min_value=0.7,
            max_value=0.99,
            mutation_rate=0.02
        )
        
        self.parameters["connection_strength"] = AdaptiveParameter(
            name="connection_strength",
            value=0.3,
            min_value=0.1,
            max_value=0.8,
            mutation_rate=0.05
        )
    
    def set_fitness_function(self, fitness_func: Callable[[Dict[str, float]], float]) -> None:
        """Set the fitness function for evaluating adaptations."""
        self.fitness_function = fitness_func
    
    def adapt(self, mission_data: Dict[str, Any]) -> bool:
        """
        Adapt the system based on mission requirements.
        
        Args:
            mission_data: Current mission data and performance metrics
            
        Returns:
            Success status of adaptation
        """
        if not self.adaptation_enabled or not self.fitness_function:
            return False
        
        # Extract current parameter values
        current_params = {name: param.value for name, param in self.parameters.items()}
        
        # Calculate current fitness
        current_fitness = self.fitness_function(current_params)
        
        # Store fitness in history
        self.fitness_history.append(current_fitness)
        if len(self.fitness_history) > 100:
            self.fitness_history.pop(0)
        
        # Decide whether to evolve based on fitness trend
        should_evolve = self._should_evolve(current_fitness)
        
        if should_evolve:
            self._evolve_parameters()
            self.metrics.adaptation_count += 1
            self.metrics.last_mutation_time = time.time()
            
            # Apply evolved parameters to neuromorphic AI
            self._apply_parameters()
            
            # Log evolution event
            logger.info(f"Evolved parameters (gen {self.metrics.generation}): "
                       f"fitness={current_fitness:.4f}, adaptations={self.metrics.adaptation_count}")
            
            return True
        
        return False
    
    def _should_evolve(self, current_fitness: float) -> bool:
        """Determine if the system should evolve based on fitness trends."""
        # Always evolve in early generations
        if self.metrics.generation < 5:
            return True
        
        # Calculate fitness trend
        if len(self.fitness_history) >= 3:
            recent_avg = np.mean(self.fitness_history[-3:])
            if recent_avg < current_fitness:
                # Fitness is decreasing, evolve more aggressively
                return True
            elif random.random() < 0.2:
                # Occasionally evolve even when doing well (exploration)
                return True
        
        # Time-based evolution (ensure regular adaptation)
        time_since_last = time.time() - self.metrics.last_mutation_time
        if time_since_last > 300:  # 5 minutes
            return True
            
        return False
    
    def _evolve_parameters(self) -> None:
        """Evolve parameters through mutation."""
        # Select parameters to mutate (not all at once)
        params_to_mutate = random.sample(
            list(self.parameters.keys()),
            k=max(1, len(self.parameters) // 3)
        )
        
        # Apply mutations
        for param_name in params_to_mutate:
            self.parameters[param_name].mutate()
        
        # Increment generation counter
        self.metrics.generation += 1
    
    def _apply_parameters(self) -> None:
        """Apply evolved parameters to the neuromorphic AI system."""
        # Update decision thresholds
        for action in self.neuromorphic_ai.decision_thresholds.keys():
            param_name = f"threshold_{action}"
            if param_name in self.parameters:
                self.neuromorphic_ai.decision_thresholds[action] = self.parameters[param_name].value
        
        # Update neuron parameters (for new neurons)
        decay = self.parameters["neuron_decay"].value
        for neuron in self.neuromorphic_ai.neurons:
            # Apply with some randomness to maintain diversity
            if random.random() < 0.3:
                neuron.decay = decay * random.uniform(0.95, 1.05)
        
        # Update connection strengths
        connection_strength = self.parameters["connection_strength"].value
        if hasattr(self.neuromorphic_ai, 'connections') and self.neuromorphic_ai.connections is not None:
            # Scale existing connections
            mask = self.neuromorphic_ai.connections > 0
            self.neuromorphic_ai.connections[mask] *= connection_strength / np.mean(self.neuromorphic_ai.connections[mask])
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status and metrics."""
        return {
            "generation": self.metrics.generation,
            "adaptations": self.metrics.adaptation_count,
            "current_fitness": self.fitness_history[-1] if self.fitness_history else 0,
            "parameters": {name: param.value for name, param in self.parameters.items()},
            "improvement_rate": self._calculate_improvement_rate()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate the rate of improvement over time."""
        if len(self.fitness_history) < 10:
            return 0.0
            
        # Compare recent fitness to earlier fitness
        recent = np.mean(self.fitness_history[-5:])
        earlier = np.mean(self.fitness_history[-10:-5])
        
        # Calculate improvement rate
        if earlier == 0:
            return 0.0
            
        improvement = (recent - earlier) / earlier
        self.metrics.improvement_rate = float(improvement)
        return float(improvement)

# Example mission-specific fitness function
def tactical_mission_fitness(parameters: Dict[str, float]) -> float:
    """
    Calculate fitness for tactical missions.
    Higher values for engage and defend thresholds are rewarded.
    """
    fitness = 0.0
    
    # Reward higher engage threshold (more selective targeting)
    if "threshold_engage" in parameters:
        fitness += parameters["threshold_engage"] * 0.4
    
    # Reward higher defend threshold
    if "threshold_defend" in parameters:
        fitness += parameters["threshold_defend"] * 0.3
    
    # Reward lower evade threshold (less retreat)
    if "threshold_evade" in parameters:
        fitness += (1.0 - parameters["threshold_evade"]) * 0.2
    
    # Reward connection strength for faster response
    if "connection_strength" in parameters:
        fitness += parameters["connection_strength"] * 0.1
    
    return fitness

# Example usage
if __name__ == "__main__":
    # Create neuromorphic AI
    ai_model = NeuromorphicAI(num_neurons=64)
    
    # Create adaptive evolution system
    evolution = AdaptiveEvolution(ai_model)
    
    # Set fitness function based on mission type
    evolution.set_fitness_function(tactical_mission_fitness)
    
    # Simulate mission data
    mission_data = {
        "mission_type": "tactical",
        "threat_level": 0.7,
        "target_density": 0.4,
        "environmental_conditions": "adverse"
    }
    
    # Adapt system based on mission
    adapted = evolution.adapt(mission_data)
    
    # Get evolution status
    status = evolution.get_evolution_status()
    logger.info(f"Evolution status: {status}")
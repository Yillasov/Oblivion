#!/usr/bin/env python3
"""
Learning capabilities for stealth effectiveness.

This module provides learning algorithms that enable stealth systems
to improve their effectiveness over time based on experience.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
import time

from src.stealth.base.interfaces import NeuromorphicStealth, StealthInterface
from src.simulation.sensors.stealth_detection import SignatureType


class LearningMode(Enum):
    """Learning modes for stealth effectiveness."""
    SUPERVISED = 0    # Learn from explicit feedback
    REINFORCEMENT = 1 # Learn from success/failure outcomes
    UNSUPERVISED = 2  # Learn from patterns in data
    TRANSFER = 3      # Apply learning from one domain to another


class StealthEffectivenessLearner:
    """Learning system for stealth effectiveness."""
    
    def __init__(self, learning_mode: LearningMode = LearningMode.REINFORCEMENT):
        """Initialize the stealth effectiveness learner."""
        self.learning_mode = learning_mode
        self.stealth_systems: Dict[str, StealthInterface] = {}
        self.experience_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.learning_models: Dict[str, Dict[str, Any]] = {}
        self.max_buffer_size = 1000
        
    def register_stealth_system(self, system_id: str, system: StealthInterface) -> bool:
        """Register a stealth system for learning."""
        if system_id in self.stealth_systems:
            return False
            
        self.stealth_systems[system_id] = system
        self.experience_buffer[system_id] = []
        
        # Initialize learning model for this system
        self._initialize_learning_model(system_id, system)
        
        return True
        
    def record_experience(self, 
                        system_id: str, 
                        state: Dict[str, Any],
                        action: Dict[str, Any],
                        outcome: Dict[str, Any],
                        threat_data: Dict[str, Any]) -> bool:
        """
        Record an experience for learning.
        
        Args:
            system_id: ID of the stealth system
            state: State before action
            action: Action taken
            outcome: Outcome after action
            threat_data: Threat data during the experience
            
        Returns:
            Success status
        """
        if system_id not in self.stealth_systems:
            return False
            
        # Create experience record
        experience = {
            "timestamp": time.time(),
            "state": state,
            "action": action,
            "outcome": outcome,
            "threat_data": threat_data,
            "processed": False
        }
        
        # Add to buffer
        self.experience_buffer[system_id].append(experience)
        
        # Trim buffer if needed
        if len(self.experience_buffer[system_id]) > self.max_buffer_size:
            self.experience_buffer[system_id] = self.experience_buffer[system_id][-self.max_buffer_size:]
            
        return True
        
    def learn(self, system_id: str) -> Dict[str, Any]:
        """
        Perform learning for a specific stealth system.
        
        Args:
            system_id: ID of the stealth system
            
        Returns:
            Learning results
        """
        if system_id not in self.stealth_systems:
            return {"success": False, "reason": "System not registered"}
            
        if system_id not in self.learning_models:
            return {"success": False, "reason": "Learning model not initialized"}
            
        # Get unprocessed experiences
        experiences = [e for e in self.experience_buffer[system_id] if not e["processed"]]
        
        if not experiences:
            return {"success": True, "changes": False, "reason": "No new experiences"}
            
        # Process based on learning mode
        if self.learning_mode == LearningMode.REINFORCEMENT:
            results = self._reinforcement_learning(system_id, experiences)
        elif self.learning_mode == LearningMode.SUPERVISED:
            results = self._supervised_learning(system_id, experiences)
        elif self.learning_mode == LearningMode.UNSUPERVISED:
            results = self._unsupervised_learning(system_id, experiences)
        elif self.learning_mode == LearningMode.TRANSFER:
            results = self._transfer_learning(system_id, experiences)
        else:
            results = self._reinforcement_learning(system_id, experiences)
            
        # Mark experiences as processed
        for e in experiences:
            e["processed"] = True
            
        return results
        
    def get_learned_parameters(self, system_id: str) -> Dict[str, Any]:
        """Get learned parameters for a stealth system."""
        if system_id not in self.learning_models:
            return {}
            
        return self.learning_models[system_id].get("parameters", {})
        
    def _initialize_learning_model(self, system_id: str, system: StealthInterface) -> None:
        """Initialize learning model for a stealth system."""
        # Get system specifications if available
        specs = None
        if hasattr(system, "get_specifications"):
            specs = system.get_specifications()
            
        # Create basic learning model
        self.learning_models[system_id] = {
            "parameters": {
                SignatureType.RADAR.name: {
                    "power_weight": 0.7,
                    "effectiveness": 0.5
                },
                SignatureType.INFRARED.name: {
                    "power_weight": 0.6,
                    "effectiveness": 0.5
                },
                SignatureType.ACOUSTIC.name: {
                    "power_weight": 0.5,
                    "effectiveness": 0.5
                },
                SignatureType.ELECTROMAGNETIC.name: {
                    "power_weight": 0.6,
                    "effectiveness": 0.5
                }
            },
            "learning_rate": 0.1,
            "discount_factor": 0.9,
            "exploration_rate": 0.2,
            "iterations": 0
        }
        
    def _reinforcement_learning(self, 
                              system_id: str, 
                              experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply reinforcement learning to improve stealth effectiveness.
        
        Args:
            system_id: ID of the stealth system
            experiences: List of experiences to learn from
            
        Returns:
            Learning results
        """
        if not experiences:
            return {"success": True, "changes": False}
            
        model = self.learning_models[system_id]
        learning_rate = model["learning_rate"]
        parameters = model["parameters"]
        changes = False
        
        # Process each experience
        for experience in experiences:
            # Extract data
            action = experience["action"]
            outcome = experience["outcome"]
            threat_data = experience["threat_data"]
            
            # Calculate reward based on detection outcomes
            reward = self._calculate_reward(outcome)
            
            # Update parameters for each signature type
            for sig_type in SignatureType:
                sig_name = sig_type.name
                
                if sig_name in parameters:
                    # Extract threat level for this signature type
                    threat_level = self._extract_threat_level(threat_data, sig_type)
                    
                    # Only update if there was a significant threat
                    if threat_level > 0.1:
                        # Update power weight based on reward
                        old_weight = parameters[sig_name]["power_weight"]
                        
                        # If power level was in the action, use it for learning
                        if "power_level" in action:
                            power_used = action["power_level"]
                            
                            # Calculate optimal power weight
                            optimal_weight = power_used / max(0.1, threat_level)
                            
                            # Update weight using TD learning
                            new_weight = old_weight + learning_rate * (reward * optimal_weight - old_weight)
                            new_weight = max(0.1, min(1.0, new_weight))
                            
                            # Store if changed
                            if abs(new_weight - old_weight) > 0.01:
                                parameters[sig_name]["power_weight"] = new_weight
                                changes = True
                        
                        # Update effectiveness estimate
                        if "effectiveness" in outcome:
                            old_effectiveness = parameters[sig_name]["effectiveness"]
                            measured = outcome["effectiveness"].get(sig_name, old_effectiveness)
                            
                            # Update using simple averaging
                            new_effectiveness = old_effectiveness + learning_rate * (measured - old_effectiveness)
                            
                            # Store if changed
                            if abs(new_effectiveness - old_effectiveness) > 0.01:
                                parameters[sig_name]["effectiveness"] = new_effectiveness
                                changes = True
        
        # Update model
        model["iterations"] += len(experiences)
        
        # Reduce exploration rate over time
        model["exploration_rate"] = max(0.05, model["exploration_rate"] * 0.99)
        
        return {
            "success": True,
            "changes": changes,
            "model_updates": len(experiences),
            "exploration_rate": model["exploration_rate"]
        }
        
    def _supervised_learning(self, 
                           system_id: str, 
                           experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple supervised learning implementation."""
        # Simplified implementation
        return {"success": True, "changes": False, "method": "supervised"}
        
    def _unsupervised_learning(self, 
                             system_id: str, 
                             experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple unsupervised learning implementation."""
        # Simplified implementation
        return {"success": True, "changes": False, "method": "unsupervised"}
        
    def _transfer_learning(self, 
                         system_id: str, 
                         experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple transfer learning implementation."""
        # Simplified implementation
        return {"success": True, "changes": False, "method": "transfer"}
        
    def _calculate_reward(self, outcome: Dict[str, Any]) -> float:
        """Calculate reward from outcome."""
        reward = 0.0
        
        # Reward for not being detected
        if "detected" in outcome:
            reward += 1.0 if not outcome["detected"] else -0.5
            
        # Reward for effectiveness
        if "effectiveness" in outcome:
            effectiveness_values = outcome["effectiveness"].values()
            if effectiveness_values:
                avg_effectiveness = sum(effectiveness_values) / len(effectiveness_values)
                reward += avg_effectiveness
                
        # Reward for energy efficiency
        if "energy_efficiency" in outcome:
            reward += outcome["energy_efficiency"] * 0.5
            
        return reward
        
    def _extract_threat_level(self, threat_data: Dict[str, Any], sig_type: SignatureType) -> float:
        """Extract threat level for a specific signature type."""
        # Default mapping of signature types to threat data keys
        threat_keys = {
            SignatureType.RADAR: "radar_threats",
            SignatureType.INFRARED: "ir_threats",
            SignatureType.ACOUSTIC: "acoustic_threats",
            SignatureType.ELECTROMAGNETIC: "em_threats"
        }
        
        # Get threats for this signature type
        key = threat_keys.get(sig_type)
        if not key or key not in threat_data:
            return 0.0
            
        threats = threat_data[key]
        
        # Calculate maximum threat level
        if not threats:
            return 0.0
            
        return max([t.get("threat_level", 0.0) for t in threats])
        
    def recommend_parameters(self, 
                           system_id: str, 
                           threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend stealth parameters based on learned model.
        
        Args:
            system_id: ID of the stealth system
            threat_data: Current threat data
            
        Returns:
            Recommended parameters
        """
        if system_id not in self.learning_models:
            return {}
            
        model = self.learning_models[system_id]
        parameters = model["parameters"]
        
        # Extract threat levels for each signature type
        threat_levels = {}
        for sig_type in SignatureType:
            threat_levels[sig_type.name] = self._extract_threat_level(threat_data, sig_type)
            
        # Calculate optimal power level based on threats and learned weights
        power_level = 0.0
        total_weight = 0.0
        
        for sig_name, threat_level in threat_levels.items():
            if sig_name in parameters and threat_level > 0.0:
                power_weight = parameters[sig_name]["power_weight"]
                power_level += threat_level * power_weight
                total_weight += power_weight
                
        # Normalize power level
        if total_weight > 0.0:
            power_level = power_level / total_weight
            
        # Apply exploration if enabled
        if model["exploration_rate"] > 0.0 and np.random.random() < model["exploration_rate"]:
            # Add some exploration noise
            power_level += np.random.normal(0, 0.1)
            
        # Ensure power level is within bounds
        power_level = max(0.1, min(1.0, power_level))
        
        # Determine mode based on power level
        if power_level > 0.8:
            mode = "maximum"
        elif power_level > 0.4:
            mode = "balanced"
        else:
            mode = "minimal"
            
        return {
            "power_level": power_level,
            "mode": mode,
            "learned": True
        }
import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, Any, List, Optional, Callable
import numpy as np
from enum import Enum
import time

class DecisionPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class Decision:
    """Represents a decision made by the system."""
    
    def __init__(self, 
                decision_type: str, 
                value: Any, 
                confidence: float, 
                priority: DecisionPriority):
        self.decision_type = decision_type
        self.value = value
        self.confidence = confidence
        self.priority = priority
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        return f"Decision({self.decision_type}, {self.value}, conf={self.confidence:.2f}, priority={self.priority.name})"

class DecisionMaker:
    """Base class for real-time decision-making modules."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.decision_history: List[Decision] = []
        self.max_history_size = config.get("max_history_size", 100)
    
    def make_decision(self, 
                     sensor_data: Dict[str, np.ndarray], 
                     mission_params: Dict[str, Any]) -> Decision:
        """Make a decision based on sensor data and mission parameters."""
        # Base implementation returns a default decision
        return Decision("default", None, 0.5, DecisionPriority.MEDIUM)
    
    def add_to_history(self, decision: Decision) -> None:
        """Add a decision to the history."""
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history_size:
            self.decision_history.pop(0)
    
    def get_last_decision(self, decision_type: Optional[str] = None) -> Optional[Decision]:
        """Get the last decision of a specific type."""
        if not self.decision_history:
            return None
        
        if decision_type is None:
            return self.decision_history[-1]
        
        for decision in reversed(self.decision_history):
            if decision.decision_type == decision_type:
                return decision
        
        return None

class ThreatResponseDecisionMaker(DecisionMaker):
    """Decision maker for responding to threats."""
    
    def make_decision(self, 
                     sensor_data: Dict[str, np.ndarray], 
                     mission_params: Dict[str, Any]) -> Decision:
        # Check for threats in sensor data
        threat_level = 0.0
        threat_type = "none"
        
        if "radar_warning" in sensor_data:
            radar_warning = np.max(sensor_data["radar_warning"])
            if radar_warning > threat_level:
                threat_level = radar_warning
                threat_type = "radar"
        
        if "missile_warning" in sensor_data:
            missile_warning = np.max(sensor_data["missile_warning"])
            if missile_warning > threat_level:
                threat_level = missile_warning
                threat_type = "missile"
        
        # Determine response based on threat level
        if threat_level > 0.8:
            response = "evasive_maneuver"
            priority = DecisionPriority.CRITICAL
        elif threat_level > 0.5:
            response = "activate_countermeasures"
            priority = DecisionPriority.HIGH
        elif threat_level > 0.2:
            response = "increase_situational_awareness"
            priority = DecisionPriority.MEDIUM
        else:
            response = "maintain_course"
            priority = DecisionPriority.LOW
        
        decision = Decision("threat_response", response, threat_level, priority)
        self.add_to_history(decision)
        return decision

class NavigationDecisionMaker(DecisionMaker):
    """Decision maker for navigation decisions."""
    
    def make_decision(self, 
                     sensor_data: Dict[str, np.ndarray], 
                     mission_params: Dict[str, Any]) -> Decision:
        # Determine navigation decision based on mission parameters and obstacles
        current_position = sensor_data.get("position", np.zeros(3))
        target_position = mission_params.get("target_position", np.zeros(3))
        
        # Calculate distance to target
        distance = np.linalg.norm(target_position - current_position)
        
        # Check for obstacles
        obstacle_detected = False
        obstacle_direction = np.zeros(3)
        
        if "obstacle_sensors" in sensor_data:
            obstacle_data = sensor_data["obstacle_sensors"]
            if np.max(obstacle_data) > 0.5:
                obstacle_detected = True
                # Simplified obstacle direction calculation
                obstacle_direction = np.array([1.0, 0.0, 0.0])  # Example direction
        
        # Make navigation decision
        if distance < 1.0:  # Close to target
            nav_decision = "arrive_at_target"
            priority = DecisionPriority.HIGH
            confidence = 0.9
        elif obstacle_detected:
            nav_decision = "avoid_obstacle"
            priority = DecisionPriority.CRITICAL
            confidence = 0.8
        else:
            nav_decision = "proceed_to_target"
            priority = DecisionPriority.MEDIUM
            confidence = 0.7
        
        decision = Decision("navigation", nav_decision, confidence, priority)
        self.add_to_history(decision)
        return decision

class MissionDecisionMaker(DecisionMaker):
    """Decision maker for high-level mission decisions."""
    
    def make_decision(self, 
                     sensor_data: Dict[str, np.ndarray], 
                     mission_params: Dict[str, Any]) -> Decision:
        # Determine mission phase and objectives
        mission_type = mission_params.get("mission_type", "reconnaissance")
        mission_phase = mission_params.get("mission_phase", "en_route")
        fuel_level = sensor_data.get("fuel_level", np.array([1.0]))[0]
        
        # Check mission completion criteria
        target_acquired = False
        if "target_tracking" in sensor_data:
            target_acquired = np.max(sensor_data["target_tracking"]) > 0.7
        
        # Make mission decision
        if mission_phase == "en_route" and target_acquired:
            mission_decision = "transition_to_target_engagement"
            priority = DecisionPriority.HIGH
            confidence = 0.85
        elif mission_phase == "target_engagement" and not target_acquired:
            mission_decision = "search_for_target"
            priority = DecisionPriority.MEDIUM
            confidence = 0.7
        elif fuel_level < 0.2:  # Low fuel
            mission_decision = "return_to_base"
            priority = DecisionPriority.CRITICAL
            confidence = 0.95
        else:
            mission_decision = "continue_current_phase"
            priority = DecisionPriority.LOW
            confidence = 0.6
        
        decision = Decision("mission", mission_decision, confidence, priority)
        self.add_to_history(decision)
        return decision

class DecisionMakingSystem:
    """Integrated system for real-time decision making."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.decision_makers = {
            "threat": ThreatResponseDecisionMaker(config.get("threat_config", {})),
            "navigation": NavigationDecisionMaker(config.get("navigation_config", {})),
            "mission": MissionDecisionMaker(config.get("mission_config", {}))
        }
        
        # Decision arbitration weights
        self.priority_weights = {
            DecisionPriority.CRITICAL: 1.0,
            DecisionPriority.HIGH: 0.8,
            DecisionPriority.MEDIUM: 0.5,
            DecisionPriority.LOW: 0.2
        }
    
    def update(self, 
              sensor_data: Dict[str, np.ndarray],
              mission_params: Dict[str, Any]) -> Dict[str, Decision]:
        """Update all decision makers and return their decisions."""
        decisions = {}
        
        for name, maker in self.decision_makers.items():
            decisions[name] = maker.make_decision(sensor_data, mission_params)
        
        return decisions
    
    def get_highest_priority_decision(self, decisions: Dict[str, Decision]) -> Optional[Decision]:
        """Get the highest priority decision from a set of decisions."""
        highest_priority = DecisionPriority.LOW
        highest_confidence = 0.0
        highest_decision = None
        
        for name, decision in decisions.items():
            # Calculate weighted score based on priority and confidence
            priority_weight = self.priority_weights[decision.priority]
            score = priority_weight * decision.confidence
            
            if highest_decision is None or score > highest_confidence:
                highest_priority = decision.priority
                highest_confidence = score
                highest_decision = decision
        
        # Return a default decision if no decisions were provided
        if highest_decision is None:
            return Decision("default", "maintain_course", 0.5, DecisionPriority.LOW)
        
        return highest_decision
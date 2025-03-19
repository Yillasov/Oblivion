"""
Neuromorphic controllers for adaptive stealth systems.

This module provides neuromorphic controllers that enable stealth systems
to adapt to changing threat environments.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum

from src.stealth.base.interfaces import NeuromorphicStealth, StealthInterface, StealthType
from src.simulation.sensors.stealth_detection import SignatureType


class AdaptationStrategy(Enum):
    """Adaptation strategies for neuromorphic stealth controllers."""
    REACTIVE = 0      # React to immediate threats
    PREDICTIVE = 1    # Predict and preemptively adapt
    LEARNING = 2      # Learn from past encounters
    ENERGY_SAVING = 3 # Optimize for energy efficiency


class StealthNeuromorphicController:
    """Neuromorphic controller for adaptive stealth systems."""
    
    def __init__(self, adaptation_strategy: AdaptationStrategy = AdaptationStrategy.REACTIVE):
        """Initialize the neuromorphic controller."""
        self.adaptation_strategy = adaptation_strategy
        self.stealth_systems: Dict[str, NeuromorphicStealth] = {}
        self.learning_data: Dict[str, List[Dict[str, Any]]] = {}
        self.initialized = False
        
        # Simple neural network weights for adaptation
        self.weights = {
            SignatureType.RADAR: np.array([0.7, 0.2, 0.1]),
            SignatureType.INFRARED: np.array([0.2, 0.7, 0.1]),
            SignatureType.ACOUSTIC: np.array([0.1, 0.2, 0.7]),
            SignatureType.ELECTROMAGNETIC: np.array([0.6, 0.3, 0.1])
        }
        
    def initialize(self) -> bool:
        """Initialize the controller."""
        self.initialized = True
        return True
        
    def register_stealth_system(self, system_id: str, system: NeuromorphicStealth) -> bool:
        """Register a stealth system with the controller."""
        if system_id in self.stealth_systems:
            return False
            
        self.stealth_systems[system_id] = system
        self.learning_data[system_id] = []
        return True
        
    def process_cycle(self, 
                     sensor_data: Dict[str, Any],
                     threat_data: Dict[str, Any],
                     dt: float) -> Dict[str, Dict[str, Any]]:
        """
        Process a control cycle and adapt stealth systems.
        
        Args:
            sensor_data: Current sensor readings
            threat_data: Detected threats and their characteristics
            dt: Time step in seconds
            
        Returns:
            Adaptations for each stealth system
        """
        if not self.initialized:
            return {}
            
        adaptations = {}
        
        # Process each stealth system
        for system_id, system in self.stealth_systems.items():
            # Get system specs and status
            specs = system.get_specifications()
            status = system.get_status()
            
            # Skip inactive systems
            if not status.get("active", False):
                continue
                
            # Determine adaptation based on strategy
            if self.adaptation_strategy == AdaptationStrategy.REACTIVE:
                system_adaptations = self._reactive_adaptation(system_id, system, threat_data)
            elif self.adaptation_strategy == AdaptationStrategy.PREDICTIVE:
                system_adaptations = self._predictive_adaptation(system_id, system, threat_data)
            elif self.adaptation_strategy == AdaptationStrategy.LEARNING:
                system_adaptations = self._learning_adaptation(system_id, system, threat_data)
            elif self.adaptation_strategy == AdaptationStrategy.ENERGY_SAVING:
                system_adaptations = self._energy_saving_adaptation(system_id, system, threat_data)
            else:
                system_adaptations = self._reactive_adaptation(system_id, system, threat_data)
                
            # Apply adaptations through neuromorphic processing
            input_data = {
                "computation": "adaptation",
                "threat_data": threat_data,
                "current_status": status,
                "proposed_adaptations": system_adaptations
            }
            
            # Process through neuromorphic hardware
            result = system.process_data(input_data)
            
            # Store the adaptations
            adaptations[system_id] = result.get("refined_adaptations", system_adaptations)
            
            # Store learning data
            self._store_learning_data(system_id, threat_data, adaptations[system_id])
            
        return adaptations
        
    def _reactive_adaptation(self, 
                           system_id: str, 
                           system: NeuromorphicStealth,
                           threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reactive adaptation based on immediate threats."""
        adaptations = {"power_level": 0.2}  # Default low power
        
        # Get system specs
        specs = system.get_specifications()
        stealth_type = specs.stealth_type
        
        # Check for radar threats
        radar_threats = threat_data.get("radar_threats", [])
        if radar_threats:
            # Increase power for radar threats
            threat_level = max([t.get("threat_level", 0.0) for t in radar_threats])
            
            # Adjust power based on stealth type effectiveness against radar
            if stealth_type in [StealthType.RADAR_ABSORBING, 
                               StealthType.RADAR_ABSORBENT_MATERIAL,
                               StealthType.PLASMA_STEALTH,
                               StealthType.METAMATERIAL_CLOAKING]:
                adaptations["power_level"] = min(1.0, 0.4 + (threat_level * 0.6))
                
        # Check for IR threats
        ir_threats = threat_data.get("ir_threats", [])
        if ir_threats:
            # Increase power for IR threats
            threat_level = max([t.get("threat_level", 0.0) for t in ir_threats])
            
            # Adjust power based on stealth type effectiveness against IR
            if stealth_type in [StealthType.INFRARED_SUPPRESSION, 
                               StealthType.THERMAL_CAMOUFLAGE,
                               StealthType.ACTIVE_CAMOUFLAGE]:
                adaptations["power_level"] = min(1.0, 0.4 + (threat_level * 0.6))
                
        return adaptations
        
    def _predictive_adaptation(self, 
                             system_id: str, 
                             system: NeuromorphicStealth,
                             threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predictive adaptation based on threat trends."""
        # Start with reactive adaptation
        adaptations = self._reactive_adaptation(system_id, system, threat_data)
        
        # Check for trend information
        trends = threat_data.get("trends", {})
        
        # If radar threats are increasing, boost power preemptively
        if trends.get("radar_increasing", False):
            adaptations["power_level"] = min(1.0, adaptations.get("power_level", 0.0) + 0.2)
            
        # If IR threats are increasing, boost power preemptively
        if trends.get("ir_increasing", False):
            adaptations["power_level"] = min(1.0, adaptations.get("power_level", 0.0) + 0.2)
            
        return adaptations
        
    def _learning_adaptation(self, 
                           system_id: str, 
                           system: NeuromorphicStealth,
                           threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learning-based adaptation using past encounters."""
        # Start with reactive adaptation
        adaptations = self._reactive_adaptation(system_id, system, threat_data)
        
        # Use learning data if available
        if system_id in self.learning_data and self.learning_data[system_id]:
            # Simple reinforcement: increase power if similar threats were seen before
            similar_encounters = self._find_similar_encounters(system_id, threat_data)
            
            if similar_encounters:
                # Average power level from similar encounters
                avg_power = sum(e["adaptation"].get("power_level", 0.0) for e in similar_encounters) / len(similar_encounters)
                adaptations["power_level"] = min(1.0, avg_power + 0.1)  # Slight increase for learning
                
        return adaptations
        
    def _energy_saving_adaptation(self, 
                                system_id: str, 
                                system: NeuromorphicStealth,
                                threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Energy-saving adaptation that minimizes power usage."""
        # Start with reactive adaptation
        adaptations = self._reactive_adaptation(system_id, system, threat_data)
        
        # Reduce power level to save energy
        adaptations["power_level"] = max(0.1, adaptations.get("power_level", 0.0) * 0.8)
        
        return adaptations
        
    def _store_learning_data(self, 
                           system_id: str, 
                           threat_data: Dict[str, Any],
                           adaptation: Dict[str, Any]) -> None:
        """Store learning data for future adaptations."""
        if system_id not in self.learning_data:
            self.learning_data[system_id] = []
            
        # Store simplified threat data and adaptation
        learning_entry = {
            "threat": self._simplify_threat_data(threat_data),
            "adaptation": adaptation,
            "timestamp": np.datetime64('now')
        }
        
        # Add to learning data
        self.learning_data[system_id].append(learning_entry)
        
        # Limit size of learning data
        if len(self.learning_data[system_id]) > 100:
            self.learning_data[system_id] = self.learning_data[system_id][-100:]
            
    def _simplify_threat_data(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify threat data for storage."""
        simplified = {}
        
        # Extract key threat information
        if "radar_threats" in threat_data:
            simplified["radar_level"] = max([t.get("threat_level", 0.0) for t in threat_data["radar_threats"]])
            
        if "ir_threats" in threat_data:
            simplified["ir_level"] = max([t.get("threat_level", 0.0) for t in threat_data["ir_threats"]])
            
        return simplified
        
    def _find_similar_encounters(self, 
                               system_id: str, 
                               threat_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar threat encounters from learning data."""
        if system_id not in self.learning_data:
            return []
            
        similar = []
        simplified_threat = self._simplify_threat_data(threat_data)
        
        for entry in self.learning_data[system_id]:
            # Simple similarity measure
            similarity = self._calculate_similarity(simplified_threat, entry["threat"])
            
            if similarity > 0.7:  # Threshold for similarity
                similar.append(entry)
                
        return similar
        
    def _calculate_similarity(self, 
                            threat1: Dict[str, float], 
                            threat2: Dict[str, float]) -> float:
        """Calculate similarity between two threat patterns."""
        # Simple similarity calculation
        keys = set(threat1.keys()) | set(threat2.keys())
        if not keys:
            return 0.0
            
        similarity = 0.0
        for key in keys:
            val1 = threat1.get(key, 0.0)
            val2 = threat2.get(key, 0.0)
            similarity += 1.0 - min(1.0, abs(val1 - val2))
            
        return similarity / len(keys)
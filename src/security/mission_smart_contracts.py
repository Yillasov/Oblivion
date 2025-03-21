"""
Smart contracts for mission validation and authentication.
Provides automated rule enforcement for mission data on the blockchain.
"""

import logging
import json
import time
from typing import Dict, Any, List, Callable, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from src.security.blockchain_authentication import MissionAuthenticator

logger = logging.getLogger(__name__)

class ContractStatus(Enum):
    """Status of a smart contract execution."""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class SmartContract:
    """Base smart contract for mission validation."""
    contract_id: str
    creator: str
    creation_time: float
    expiration_time: float
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    status: ContractStatus = ContractStatus.PENDING
    execution_result: Optional[Dict[str, Any]] = None


class MissionValidator:
    """Validates mission parameters against predefined rules."""
    
    def __init__(self):
        """Initialize the mission validator."""
        self.validation_rules = {
            "waypoints": self._validate_waypoints,
            "altitude": self._validate_altitude,
            "duration": self._validate_duration,
            "authorization": self._validate_authorization,
            "target": self._validate_target
        }
    
    def validate_mission(self, mission_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate mission data against all applicable rules.
        
        Args:
            mission_data: Mission data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        is_valid = True
        error_messages = []
        
        # Apply each validation rule that's applicable
        for rule_name, validator in self.validation_rules.items():
            if self._is_rule_applicable(rule_name, mission_data):
                rule_valid, rule_errors = validator(mission_data)
                if not rule_valid:
                    is_valid = False
                    error_messages.extend(rule_errors)
        
        return is_valid, error_messages
    
    def _is_rule_applicable(self, rule_name: str, mission_data: Dict[str, Any]) -> bool:
        """Check if a rule is applicable to the mission data."""
        if rule_name == "waypoints" and "waypoints" in mission_data:
            return True
        elif rule_name == "altitude" and ("altitude" in mission_data or 
                                         "waypoints" in mission_data):
            return True
        elif rule_name == "duration" and "duration" in mission_data:
            return True
        elif rule_name == "authorization" and "authorized_by" in mission_data:
            return True
        elif rule_name == "target" and "target_position" in mission_data:
            return True
        return False
    
    def _validate_waypoints(self, mission_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate waypoints in mission data."""
        waypoints = mission_data.get("waypoints", [])
        errors = []
        
        if not waypoints:
            errors.append("No waypoints defined")
            return False, errors
        
        # Check waypoint format
        for i, wp in enumerate(waypoints):
            if not isinstance(wp, list) or len(wp) < 3:
                errors.append(f"Waypoint {i} has invalid format")
        
        # Check for minimum distance between waypoints
        for i in range(1, len(waypoints)):
            prev = waypoints[i-1]
            curr = waypoints[i]
            distance = sum((a - b) ** 2 for a, b in zip(prev, curr)) ** 0.5
            if distance < 10:  # Minimum 10 units between waypoints
                errors.append(f"Waypoints {i-1} and {i} are too close")
        
        return len(errors) == 0, errors
    
    def _validate_altitude(self, mission_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate altitude in mission data."""
        errors = []
        
        # Check direct altitude
        if "altitude" in mission_data:
            altitude = mission_data["altitude"]
            if altitude < 100:
                errors.append(f"Altitude {altitude} is below minimum safe altitude (100)")
            if altitude > 10000:
                errors.append(f"Altitude {altitude} exceeds maximum allowed altitude (10000)")
        
        # Check waypoint altitudes
        if "waypoints" in mission_data:
            for i, wp in enumerate(mission_data["waypoints"]):
                if len(wp) >= 3:
                    altitude = wp[2]
                    if altitude < 100:
                        errors.append(f"Waypoint {i} altitude {altitude} is below minimum safe altitude (100)")
                    if altitude > 10000:
                        errors.append(f"Waypoint {i} altitude {altitude} exceeds maximum allowed altitude (10000)")
        
        return len(errors) == 0, errors
    
    def _validate_duration(self, mission_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate mission duration."""
        errors = []
        duration = mission_data.get("duration", 0)
        
        if duration <= 0:
            errors.append("Mission duration must be positive")
        if duration > 86400:  # 24 hours
            errors.append("Mission duration exceeds maximum allowed (24 hours)")
        
        return len(errors) == 0, errors
    
    def _validate_authorization(self, mission_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate mission authorization."""
        errors = []
        authorized_by = mission_data.get("authorized_by", "")
        
        if not authorized_by:
            errors.append("Mission must have authorization")
        
        # Could check against a list of authorized entities
        authorized_entities = ["Command Center", "Mission Control", "Strategic Command"]
        if authorized_by not in authorized_entities:
            errors.append(f"Authorization by '{authorized_by}' is not recognized")
        
        return len(errors) == 0, errors
    
    def _validate_target(self, mission_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate target position."""
        errors = []
        target = mission_data.get("target_position", [])
        
        if not target or len(target) < 3:
            errors.append("Target position is invalid")
        
        # Check if target is in restricted area (example)
        restricted_areas = [
            {"center": [1000, 1000, 0], "radius": 100},
            {"center": [2000, 2000, 0], "radius": 200}
        ]
        
        for area in restricted_areas:
            center = area["center"]
            radius = area["radius"]
            distance = sum((a - b) ** 2 for a, b in zip(target, center)) ** 0.5
            if distance <= radius:
                errors.append(f"Target is in restricted area (center: {center}, radius: {radius})")
        
        return len(errors) == 0, errors


class SmartContractEngine:
    """Engine for creating and executing smart contracts."""
    
    def __init__(self, mission_authenticator: MissionAuthenticator):
        """Initialize the smart contract engine."""
        self.mission_authenticator = mission_authenticator
        self.contracts: Dict[str, SmartContract] = {}
        self.validator = MissionValidator()
        self.contract_counter = 0
    
    def create_mission_contract(self, mission_data: Dict[str, Any], 
                               creator: str, expiration_time: float) -> str:
        """
        Create a smart contract for mission validation and execution.
        
        Args:
            mission_data: Mission data
            creator: Contract creator
            expiration_time: Contract expiration time
            
        Returns:
            Contract ID
        """
        # Validate mission data
        is_valid, errors = self.validator.validate_mission(mission_data)
        if not is_valid:
            logger.error(f"Mission validation failed: {errors}")
            return ""
        
        # Create contract ID
        self.contract_counter += 1
        contract_id = f"contract_{int(time.time())}_{self.contract_counter}"
        
        # Define contract conditions
        conditions = [
            {"type": "time_window", "start": time.time(), "end": expiration_time},
            {"type": "validation", "result": True}
        ]
        
        # Define contract actions
        actions = [
            {"type": "register_mission", "mission_data": mission_data},
            {"type": "notify", "recipient": creator, "message": "Mission registered"}
        ]
        
        # Create contract
        contract = SmartContract(
            contract_id=contract_id,
            creator=creator,
            creation_time=time.time(),
            expiration_time=expiration_time,
            conditions=conditions,
            actions=actions
        )
        
        self.contracts[contract_id] = contract
        logger.info(f"Created mission contract {contract_id}")
        
        return contract_id
    
    def execute_contract(self, contract_id: str) -> bool:
        """
        Execute a smart contract.
        
        Args:
            contract_id: Contract ID
            
        Returns:
            True if executed successfully, False otherwise
        """
        if contract_id not in self.contracts:
            logger.error(f"Contract {contract_id} not found")
            return False
        
        contract = self.contracts[contract_id]
        
        # Check if contract is already executed or expired
        if contract.status != ContractStatus.PENDING:
            logger.warning(f"Contract {contract_id} is already {contract.status.value}")
            return False
        
        # Check if contract is expired
        if time.time() > contract.expiration_time:
            contract.status = ContractStatus.EXPIRED
            logger.warning(f"Contract {contract_id} has expired")
            return False
        
        # Check all conditions
        for condition in contract.conditions:
            if not self._check_condition(condition):
                contract.status = ContractStatus.FAILED
                logger.warning(f"Contract {contract_id} condition failed: {condition}")
                return False
        
        # Execute all actions
        results = []
        for action in contract.actions:
            result = self._execute_action(action)
            results.append(result)
        
        # Update contract status
        contract.status = ContractStatus.EXECUTED
        contract.execution_result = {"results": results, "timestamp": time.time()}
        
        logger.info(f"Contract {contract_id} executed successfully")
        return True
    
    def _check_condition(self, condition: Dict[str, Any]) -> bool:
        """Check if a condition is met."""
        condition_type = condition.get("type", "")
        
        if condition_type == "time_window":
            current_time = time.time()
            start_time = condition.get("start", 0)
            end_time = condition.get("end", float("inf"))
            return start_time <= current_time <= end_time
        
        elif condition_type == "validation":
            return condition.get("result", False)
        
        # Add more condition types as needed
        
        return False
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a contract action."""
        action_type = action.get("type", "")
        result = {"type": action_type, "success": False}
        
        if action_type == "register_mission":
            mission_data = action.get("mission_data", {})
            try:
                mission_id = self.mission_authenticator.register_mission(mission_data)
                result["success"] = True
                result["mission_id"] = mission_id
            except Exception as e:
                result["error"] = str(e)
        
        elif action_type == "notify":
            # In a real system, this would send a notification
            recipient = action.get("recipient", "")
            message = action.get("message", "")
            logger.info(f"Notification to {recipient}: {message}")
            result["success"] = True
        
        # Add more action types as needed
        
        return result
    
    def get_contract_status(self, contract_id: str) -> Dict[str, Any]:
        """Get the status of a contract."""
        if contract_id not in self.contracts:
            return {"error": "Contract not found"}
        
        contract = self.contracts[contract_id]
        return {
            "contract_id": contract.contract_id,
            "creator": contract.creator,
            "creation_time": contract.creation_time,
            "expiration_time": contract.expiration_time,
            "status": contract.status.value,
            "execution_result": contract.execution_result
        }


# Example usage
if __name__ == "__main__":
    # Create mission authenticator
    authenticator = MissionAuthenticator(difficulty=2)
    
    # Create smart contract engine
    contract_engine = SmartContractEngine(authenticator)
    
    # Example mission data
    mission_data = {
        "name": "Reconnaissance Alpha",
        "type": "surveillance",
        "waypoints": [[0, 0, 500], [10, 20, 600], [30, 40, 550]],
        "authorized_by": "Command Center",
        "priority": "high",
        "duration": 3600
    }
    
    # Create mission contract
    contract_id = contract_engine.create_mission_contract(
        mission_data=mission_data,
        creator="Mission Control",
        expiration_time=time.time() + 86400  # 24 hours
    )
    
    # Execute contract
    success = contract_engine.execute_contract(contract_id)
    
    # Get contract status
    status = contract_engine.get_contract_status(contract_id)
    
    logger.info(f"Contract execution success: {success}")
    logger.info(f"Contract status: {json.dumps(status, indent=2, default=str)}")
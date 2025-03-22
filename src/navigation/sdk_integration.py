"""
SDK Integration for Navigation Systems.

Provides integration between navigation systems and the SDK
for cross-system communication and resource sharing.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable

from src.navigation.testing import NavigationTestFramework, TestScenario
from src.navigation.base import NavigationSystem
from src.navigation.interfaces import NavigationState
from src.core.messaging.basic_messaging_system import BasicMessagingSystem

# Configure logger
logger = logging.getLogger(__name__)


class NavigationSDKBridge:
    """Bridge between navigation systems and the SDK."""
    
    def __init__(self, test_framework: Optional[NavigationTestFramework] = None):
        """
        Initialize the SDK bridge.
        
        Args:
            test_framework: Optional test framework to connect with
        """
        self.test_framework = test_framework
        self.messaging = BasicMessagingSystem()
        self.callbacks: Dict[str, List[Callable]] = {
            "test_start": [],
            "test_complete": [],
            "navigation_update": [],
            "resource_request": [],
            "error": []
        }
        self.connected_systems: Dict[str, Any] = {}
        self.last_sync_time = time.time()
        
        logger.info("Navigation SDK Bridge initialized")
    
    def connect_test_framework(self, test_framework: NavigationTestFramework) -> bool:
        """
        Connect to a navigation test framework.
        
        Args:
            test_framework: Test framework to connect
            
        Returns:
            Success status
        """
        self.test_framework = test_framework
        return True
    
    def connect_system(self, system_id: str, system: Any) -> bool:
        """
        Connect an external system to the bridge.
        
        Args:
            system_id: System identifier
            system: System instance
            
        Returns:
            Success status
        """
        if system_id in self.connected_systems:
            logger.warning(f"System {system_id} already connected")
            return False
            
        self.connected_systems[system_id] = system
        logger.info(f"Connected system: {system_id}")
        return True
    
    def register_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Register a callback for specific events.
        
        Args:
            event_type: Event type to register for
            callback: Callback function
            
        Returns:
            Success status
        """
        if event_type not in self.callbacks:
            logger.warning(f"Unknown event type: {event_type}")
            return False
            
        self.callbacks[event_type].append(callback)
        return True
    
    def notify_test_start(self, scenario: TestScenario, parameters: Dict[str, Any]) -> None:
        """
        Notify all connected systems about test start.
        
        Args:
            scenario: Test scenario
            parameters: Test parameters
        """
        event_data = {
            "event": "test_start",
            "scenario": scenario.value,
            "parameters": parameters,
            "timestamp": time.time()
        }
        
        # Send message to all connected systems
        for system_id in self.connected_systems:
            self.messaging.send_message(system_id, event_data)
        
        # Execute callbacks
        for callback in self.callbacks["test_start"]:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in test_start callback: {str(e)}")
    
    def notify_test_complete(self, results: Dict[str, Any]) -> None:
        """
        Notify all connected systems about test completion.
        
        Args:
            results: Test results
        """
        event_data = {
            "event": "test_complete",
            "results": results,
            "timestamp": time.time()
        }
        
        # Send message to all connected systems
        for system_id in self.connected_systems:
            self.messaging.send_message(system_id, event_data)
        
        # Execute callbacks
        for callback in self.callbacks["test_complete"]:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in test_complete callback: {str(e)}")
    
    def send_navigation_update(self, nav_state: NavigationState) -> bool:
        """
        Send navigation state update to connected systems.
        
        Args:
            nav_state: Navigation state
            
        Returns:
            Success status
        """
        event_data = {
            "event": "navigation_update",
            "state": {
                "position": nav_state.position,
                "orientation": nav_state.orientation,
                "velocity": nav_state.velocity,
                "timestamp": nav_state.timestamp,
                "system_id": nav_state.system_id
            },
            "timestamp": time.time()
        }
        
        # Send message to all connected systems
        success = True
        for system_id in self.connected_systems:
            if not self.messaging.send_message(system_id, event_data):
                success = False
        
        # Execute callbacks
        for callback in self.callbacks["navigation_update"]:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in navigation_update callback: {str(e)}")
                success = False
        
        return success
    
    def request_resources(self, resource_type: str, amount: int) -> Optional[Dict[str, Any]]:
        """
        Request resources from the SDK.
        
        Args:
            resource_type: Type of resource
            amount: Amount needed
            
        Returns:
            Resource allocation information or None if failed
        """
        request = {
            "event": "resource_request",
            "resource_type": resource_type,
            "amount": amount,
            "requester": "navigation_system",
            "priority": "high",
            "timestamp": time.time()
        }
        
        # Execute callbacks to get resource allocation
        for callback in self.callbacks["resource_request"]:
            try:
                result = callback(request)
                if result and isinstance(result, dict) and "allocated" in result:
                    return result
            except Exception as e:
                logger.error(f"Error in resource_request callback: {str(e)}")
        
        return None
    
    def sync_with_sdk(self) -> bool:
        """
        Synchronize navigation data with the SDK.
        
        Returns:
            Success status
        """
        if not self.test_framework:
            logger.warning("No test framework connected for SDK sync")
            return False
            
        current_time = time.time()
        if current_time - self.last_sync_time < 1.0:  # Limit sync rate to once per second
            return True
            
        self.last_sync_time = current_time
        
        # Get current test scenario if active
        if self.test_framework.current_scenario:
            # Notify about current test status
            status_data = {
                "event": "test_status",
                "scenario": self.test_framework.current_scenario.value,
                "elapsed_time": current_time - self.last_sync_time,
                "systems": list(self.test_framework.integrator.navigation_systems.keys()),
                "timestamp": current_time
            }
            
            # Send status update to all connected systems
            for system_id in self.connected_systems:
                self.messaging.send_message(system_id, status_data)
        
        return True


class SDKResourceProvider:
    """Provider for SDK resources to navigation systems."""
    
    def __init__(self, bridge: NavigationSDKBridge):
        """
        Initialize the SDK resource provider.
        
        Args:
            bridge: Navigation SDK bridge
        """
        self.bridge = bridge
        self.resource_pools: Dict[str, int] = {
            "memory": 1024 * 1024 * 100,  # 100 MB
            "compute": 1000,              # Arbitrary compute units
            "bandwidth": 10000            # Arbitrary bandwidth units
        }
        self.allocations: Dict[str, Dict[str, int]] = {}
        
        # Register as resource provider
        self.bridge.register_callback("resource_request", self.handle_resource_request)
        
        logger.info("SDK Resource Provider initialized")
    
    def handle_resource_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle resource request from navigation systems.
        
        Args:
            request: Resource request
            
        Returns:
            Resource allocation result
        """
        resource_type = request.get("resource_type")
        amount = request.get("amount", 0)
        requester = request.get("requester", "unknown")
        
        if resource_type not in self.resource_pools:
            return {
                "allocated": False,
                "reason": f"Unknown resource type: {resource_type}"
            }
            
        # Check if enough resources available
        if self.resource_pools[resource_type] < amount:
            return {
                "allocated": False,
                "reason": "Insufficient resources",
                "available": self.resource_pools[resource_type]
            }
            
        # Allocate resources
        if requester not in self.allocations:
            self.allocations[requester] = {}
            
        if resource_type not in self.allocations[requester]:
            self.allocations[requester][resource_type] = 0
            
        self.allocations[requester][resource_type] += amount
        self.resource_pools[resource_type] -= amount
        
        logger.info(f"Allocated {amount} {resource_type} to {requester}")
        
        return {
            "allocated": True,
            "resource_type": resource_type,
            "amount": amount,
            "total_allocated": self.allocations[requester][resource_type]
        }
    
    def release_resources(self, requester: str) -> bool:
        """
        Release all resources allocated to a requester.
        
        Args:
            requester: Resource requester
            
        Returns:
            Success status
        """
        if requester not in self.allocations:
            return False
            
        # Return resources to pools
        for resource_type, amount in self.allocations[requester].items():
            self.resource_pools[resource_type] += amount
            
        # Clear allocations
        del self.allocations[requester]
        
        logger.info(f"Released all resources for {requester}")
        return True
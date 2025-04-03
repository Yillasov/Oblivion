"""
Messaging system for landing gear communication with other aircraft systems.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Callable
import time
import logging

from src.core.messaging.messaging_interface import MessagingInterface
from src.core.messaging.basic_messaging_system import BasicMessagingSystem
from src.landing_gear.base import NeuromorphicLandingGear
from src.landing_gear.integration import LandingGearIntegration

logger = logging.getLogger(__name__)

class LandingGearMessaging:
    """Messaging system for landing gear communication."""
    
    def __init__(self, landing_gear: NeuromorphicLandingGear, messaging_system: Optional[MessagingInterface] = None):
        """
        Initialize landing gear messaging system.
        
        Args:
            landing_gear: The landing gear system
            messaging_system: Optional messaging system to use (creates a new one if None)
        """
        self.landing_gear = landing_gear
        self.integration = LandingGearIntegration(landing_gear)
        self.messaging = messaging_system or BasicMessagingSystem()
        self.subscriptions: Dict[str, List[Callable]] = {}
        
        # Register standard message handlers
        self._register_standard_handlers()
        
        logger.info("Landing gear messaging system initialized")
    
    def _register_standard_handlers(self) -> None:
        """Register standard message handlers for landing gear events."""
        # Register integration event handlers to send messages
        self.integration.register_event_handler("pre_landing", self._handle_pre_landing)
        self.integration.register_event_handler("landing", self._handle_landing)
        self.integration.register_event_handler("post_landing", self._handle_post_landing)
        self.integration.register_event_handler("emergency", self._handle_emergency)
    
    def _handle_pre_landing(self, data: Dict[str, Any]) -> None:
        """Handle pre-landing event."""
        message = {
            "event": "pre_landing",
            "timestamp": time.time(),
            "landing_gear_status": self.landing_gear.status,
            "data": data
        }
        self.messaging.send_message("flight_control", message)
        self.messaging.send_message("mission_control", message)
    
    def _handle_landing(self, data: Dict[str, Any]) -> None:
        """Handle landing event."""
        message = {
            "event": "landing",
            "timestamp": time.time(),
            "landing_gear_status": self.landing_gear.status,
            "telemetry": self.landing_gear.get_telemetry().__dict__,
            "data": data
        }
        self.messaging.send_message("flight_control", message)
        self.messaging.send_message("mission_control", message)
        self.messaging.send_message("avionics", message)
    
    def _handle_post_landing(self, data: Dict[str, Any]) -> None:
        """Handle post-landing event."""
        message = {
            "event": "post_landing",
            "timestamp": time.time(),
            "landing_gear_status": self.landing_gear.status,
            "data": data
        }
        self.messaging.send_message("mission_control", message)
    
    def _handle_emergency(self, data: Dict[str, Any]) -> None:
        """Handle emergency event."""
        message = {
            "event": "emergency",
            "timestamp": time.time(),
            "landing_gear_status": self.landing_gear.status,
            "emergency_type": data.get("type", "unknown"),
            "severity": data.get("severity", "high"),
            "data": data
        }
        # Send emergency to all systems
        self.messaging.send_message("flight_control", message)
        self.messaging.send_message("mission_control", message)
        self.messaging.send_message("avionics", message)
        self.messaging.send_message("emergency_systems", message)
    
    def process_incoming_messages(self) -> None:
        """Process incoming messages from other systems."""
        # Check for messages from flight control
        flight_control_msg = self.messaging.receive_message("landing_gear")
        if flight_control_msg:
            self._handle_incoming_message(flight_control_msg)
    
    def _handle_incoming_message(self, message: Dict[str, Any]) -> None:
        """Handle an incoming message."""
        message_type = message.get("type")
        
        if message_type == "deploy":
            # Deploy landing gear
            self.landing_gear.deploy()
            logger.info("Landing gear deployed via message command")
            
        elif message_type == "retract":
            # Retract landing gear
            self.landing_gear.retract()
            logger.info("Landing gear retracted via message command")
            
        elif message_type == "status_request":
            # Send status response
            response = {
                "type": "status_response",
                "timestamp": time.time(),
                "request_id": message.get("request_id"),
                "landing_gear_status": self.landing_gear.status,
                "telemetry": self.landing_gear.get_telemetry().__dict__
            }
            self.messaging.send_message(message.get("reply_to", "flight_control"), response)
            
        elif message_type == "configure":
            # Configure landing gear
            config = message.get("config", {})
            
            # Update mission parameters through integration
            if "mission_params" in config:
                self.integration.set_mission_parameters(config["mission_params"])
                
            # Send configuration confirmation
            response = {
                "type": "configure_response",
                "timestamp": time.time(),
                "request_id": message.get("request_id"),
                "status": "success",
                "applied_config": config
            }
            self.messaging.send_message(message.get("reply_to", "flight_control"), response)
    
    def send_telemetry(self) -> None:
        """Send current landing gear telemetry to monitoring systems."""
        telemetry = self.landing_gear.get_telemetry()
        message = {
            "type": "telemetry",
            "timestamp": time.time(),
            "landing_gear_id": id(self.landing_gear),
            "telemetry": telemetry.__dict__,
            "status": self.landing_gear.status
        }
        self.messaging.send_message("monitoring", message)
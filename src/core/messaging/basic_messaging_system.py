from typing import Any, Dict
from src.core.messaging.messaging_interface import MessagingInterface

class BasicMessagingSystem(MessagingInterface):
    """Basic messaging system for inter-system communication."""
    
    def __init__(self):
        self.message_queues: Dict[str, list[Dict[str, Any]]] = {}
    
    def send_message(self, target_system: str, message: Dict[str, Any]) -> bool:
        """Send a message to a target system."""
        if target_system not in self.message_queues:
            self.message_queues[target_system] = []
        
        self.message_queues[target_system].append(message)
        return True
    
    def receive_message(self, source_system: str) -> Dict[str, Any]:
        """Receive a message from a source system."""
        if source_system in self.message_queues and self.message_queues[source_system]:
            return self.message_queues[source_system].pop(0)
        return {}
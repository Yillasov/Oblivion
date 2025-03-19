from typing import Any, Dict

class MessagingInterface:
    """Interface for messaging between systems."""
    
    def send_message(self, target_system: str, message: Dict[str, Any]) -> bool:
        """Send a message to a target system."""
        raise NotImplementedError("send_message must be implemented by subclasses")
    
    def receive_message(self, source_system: str) -> Dict[str, Any]:
        """Receive a message from a source system."""
        raise NotImplementedError("receive_message must be implemented by subclasses")
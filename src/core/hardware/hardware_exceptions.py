from typing import Optional

class HardwareError(Exception):
    """Base exception for hardware-related errors."""
    def __init__(self, message: str, error_code: Optional[int] = None):
        self.error_code = error_code
        super().__init__(message)

class InitializationError(HardwareError):
    """Raised when hardware initialization fails."""
    pass

class CommunicationError(HardwareError):
    """Raised when communication with hardware fails."""
    pass

class ResourceError(HardwareError):
    """Raised when hardware resource allocation fails."""
    pass

class CapabilityError(HardwareError):
    """Raised when requested capability is not available."""
    pass
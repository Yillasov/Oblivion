from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SensorReading:
    sensor_id: str
    value: float
    timestamp: datetime
    unit: str
    status: str = "ok"

class SensorInterface(ABC):
    """Base interface for hardware sensors."""
    
    def __init__(self):
        self.sensors: Dict[str, Dict[str, Any]] = {}
        self.last_readings: Dict[str, SensorReading] = {}
    
    @abstractmethod
    def initialize_sensors(self) -> bool:
        """Initialize all hardware sensors."""
        pass
    
    @abstractmethod
    def read_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """Read data from a specific sensor."""
        pass
    
    def get_all_readings(self) -> Dict[str, SensorReading]:
        """Get readings from all sensors."""
        readings = {}
        for sensor_id in self.sensors:
            reading = self.read_sensor(sensor_id)
            if reading:
                readings[sensor_id] = reading
        return readings
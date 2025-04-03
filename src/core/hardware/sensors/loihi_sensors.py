#!/usr/bin/env python3
"""
Loihi hardware sensor interface.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime
from typing import Dict, Any, Optional
from .base import SensorInterface, SensorReading

class LoihiSensorInterface(SensorInterface):
    
    
    def initialize_sensors(self) -> bool:
        """Initialize Loihi sensors."""
        self.sensors = {
            "neuron_voltage": {
                "type": "voltage",
                "unit": "volts",
                "range": (0.0, 1.5)
            },
            "synaptic_current": {
                "type": "current",
                "unit": "microamps",
                "range": (0, 100)
            },
            "core_temperature": {
                "type": "temperature",
                "unit": "celsius",
                "range": (0, 85)
            },
            "spike_activity": {
                "type": "frequency",
                "unit": "hz",
                "range": (0, 1000)
            }
        }
        return True
    
    def read_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """Read from Loihi sensor."""
        if sensor_id not in self.sensors:
            return None
            
        sensor_values = {
            "neuron_voltage": 1.2,
            "synaptic_current": 50.0,
            "core_temperature": 65.0,
            "spike_activity": 450.0
        }
        
        reading = SensorReading(
            sensor_id=sensor_id,
            value=sensor_values.get(sensor_id, 0.0),
            timestamp=datetime.now(),
            unit=self.sensors[sensor_id]["unit"]
        )
        
        self.last_readings[sensor_id] = reading
        return reading
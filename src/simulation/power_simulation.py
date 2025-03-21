from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger("power_simulation")

@dataclass
class SimulationConfig:
    """Configuration for power system simulation."""
    duration: float  # Duration of the simulation in seconds
    time_step: float  # Time step for each simulation iteration in seconds
    environment_config: Optional[Dict[str, Any]] = None  # Environmental conditions

class PowerSimulationEnvironment:
    """Simulation environment for power systems."""
    
    def __init__(self, config: SimulationConfig):
        """Initialize power simulation environment."""
        self.config = config
        self.simulation_time = 0.0
        self.is_running = False
        self.current_step = 0
        self.power_systems: Dict[str, Any] = {}  # Placeholder for power systems
        self.simulation_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize environment model
        self.environment = self.config.environment_config or {
            "temperature": 288.15,  # K (15°C)
            "pressure": 101325,     # Pa (sea level)
            "wind": np.zeros(3),    # No wind
        }
        
        logger.info("Power simulation environment initialized")

    def register_system(self, system_id: str, system: Any) -> bool:
        """Register a power system for simulation."""
        if system_id in self.power_systems:
            return False
        
        self.power_systems[system_id] = system
        self.simulation_data[system_id] = []
        system.initialize()
        return True

    def start_simulation(self) -> bool:
        """Start the simulation."""
        if self.is_running:
            return False
        
        self.is_running = True
        self.simulation_time = 0.0
        self.current_step = 0
        logger.info("Simulation started")
        return True

    def step(self) -> Dict[str, Any]:
        """Execute a single simulation step."""
        if not self.is_running:
            return {"status": "not_running"}
        
        self.simulation_time += self.config.time_step
        self.current_step += 1
        
        # Simulate each power system
        results = {}
        for system_id, system in self.power_systems.items():
            # Simulate system performance
            performance = system.calculate_output(self.environment)
            self.simulation_data[system_id].append({
                "time": self.simulation_time,
                "performance": performance
            })
            results[system_id] = performance
        
        return {
            "status": "running",
            "time": self.simulation_time,
            "step": self.current_step,
            "results": results
        }

    def run_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        self.start_simulation()
        
        while self.is_running and self.simulation_time < self.config.duration:
            self.step()
        
        logger.info("Simulation completed")
        return self.get_simulation_results()

    def get_simulation_results(self) -> Dict[str, Any]:
        """Get complete simulation results."""
        if self.is_running:
            return {"error": "Simulation still running"}
        
        # Compile results for each system
        system_results = {}
        for system_id, data in self.simulation_data.items():
            if not data:
                continue
            
            # Calculate average performance metrics
            avg_performance = {}
            for metric in data[0]["performance"].keys():
                if isinstance(data[0]["performance"][metric], (int, float)):
                    values = [d["performance"].get(metric, 0) for d in data]
                    avg_performance[metric] = sum(values) / len(values)
            
            system_results[system_id] = {
                "average_performance": avg_performance,
                "time_series": {
                    "time": [d["time"] for d in data],
                    "performance": [d["performance"] for d in data]
                }
            }
        
        return system_results
"""
Simulation environments for propulsion systems testing and validation.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
import time
from dataclasses import dataclass

from src.propulsion.base import PropulsionInterface, PropulsionSpecs, PropulsionType
from src.propulsion.manufacturing_workflow import ManufacturingWorkflow
from src.propulsion.manufacturing_workflow_integration import ManufacturingWorkflowIntegrator
from src.propulsion.propulsion_optimization_integration import PropulsionOptimizationIntegrator
from src.simulation.physics.environment import EnvironmentModel


class SimulationMode(Enum):
    """Simulation modes for propulsion testing."""
    STATIC = 0       # Static ground test
    WIND_TUNNEL = 1  # Wind tunnel test
    FLIGHT = 2       # In-flight simulation
    EXTREME = 3      # Extreme conditions test
    ENDURANCE = 4    # Long-duration endurance test
    FAILURE = 5      # Failure mode testing


@dataclass
class SimulationConfig:
    """Configuration for propulsion simulation."""
    mode: SimulationMode = SimulationMode.STATIC
    duration: float = 300.0  # Simulation duration in seconds
    time_step: float = 0.1   # Simulation time step in seconds
    record_interval: float = 1.0  # Data recording interval
    environment_config: Dict[str, Any] = dict()
    failure_scenarios: List[Dict[str, Any]] = None
    visualization_enabled: bool = False


class PropulsionSimulationEnvironment:
    """Simulation environment for propulsion systems."""
    
    def __init__(self, config: SimulationConfig):
        """Initialize propulsion simulation environment."""
        self.config = config
        
        # Initialize environment model
        env_config = config.environment_config or {
            "temperature": 288.15,  # K (15°C)
            "pressure": 101325,     # Pa (sea level)
            "wind": np.zeros(3),    # No wind
            "turbulence_intensity": 0.0
        }
        self.environment = EnvironmentModel(env_config)
        
        # Simulation state
        self.simulation_time = 0.0
        self.is_running = False
        self.is_paused = False
        self.current_step = 0
        
        # Systems being simulated
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        
        # Optional integrations
        self.manufacturing_integrator: Optional[ManufacturingWorkflowIntegrator] = None
        self.optimization_integrator: Optional[PropulsionOptimizationIntegrator] = None
        
        # Simulation data
        self.simulation_data: Dict[str, List[Dict[str, Any]]] = {}
        self.events: List[Dict[str, Any]] = []
        
    def register_system(self, system_id: str, system: PropulsionInterface) -> bool:
        """Register a propulsion system for simulation."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        self.simulation_data[system_id] = []
        
        # Initialize the system
        system.initialize()
        
        return True
        
    def set_manufacturing_integrator(self, integrator: ManufacturingWorkflowIntegrator) -> None:
        """Set manufacturing workflow integrator."""
        self.manufacturing_integrator = integrator
        
    def set_optimization_integrator(self, integrator: PropulsionOptimizationIntegrator) -> None:
        """Set propulsion optimization integrator."""
        self.optimization_integrator = integrator
        
    def get_flight_conditions(self, position: np.ndarray) -> Dict[str, float]:
        """Get current flight conditions at a position."""
        # Get atmospheric conditions
        atm_conditions = self.environment.get_atmospheric_conditions(position)
        
        # Get wind vector
        wind = self.environment.get_wind_vector(position)
        
        # Combine into flight conditions
        flight_conditions = {
            **atm_conditions,
            "wind_x": wind[0],
            "wind_y": wind[1],
            "wind_z": wind[2],
            "altitude": position[2]
        }
        
        return flight_conditions
        
    def start_simulation(self) -> bool:
        """Start the simulation."""
        if self.is_running:
            return False
            
        self.is_running = True
        self.is_paused = False
        self.simulation_time = 0.0
        self.current_step = 0
        
        # Clear previous simulation data
        for system_id in self.propulsion_systems:
            self.simulation_data[system_id] = []
            
        self.events = []
        
        # Record start event
        self.events.append({
            "time": self.simulation_time,
            "type": "simulation_start",
            "mode": self.config.mode.name
        })
        
        return True
        
    def pause_simulation(self) -> bool:
        """Pause the simulation."""
        if not self.is_running or self.is_paused:
            return False
            
        self.is_paused = True
        
        # Record pause event
        self.events.append({
            "time": self.simulation_time,
            "type": "simulation_pause"
        })
        
        return True
        
    def resume_simulation(self) -> bool:
        """Resume the simulation."""
        if not self.is_running or not self.is_paused:
            return False
            
        self.is_paused = False
        
        # Record resume event
        self.events.append({
            "time": self.simulation_time,
            "type": "simulation_resume"
        })
        
        return True
        
    def stop_simulation(self) -> bool:
        """Stop the simulation."""
        if not self.is_running:
            return False
            
        self.is_running = False
        self.is_paused = False
        
        # Record stop event
        self.events.append({
            "time": self.simulation_time,
            "type": "simulation_stop"
        })
        
        return True
        
    def inject_failure(self, system_id: str, failure_type: str, severity: float) -> bool:
        """Inject a failure into the simulation."""
        if system_id not in self.propulsion_systems:
            return False
            
        # Record failure event
        self.events.append({
            "time": self.simulation_time,
            "type": "failure_injection",
            "system_id": system_id,
            "failure_type": failure_type,
            "severity": severity
        })
        
        # TODO: Implement actual failure effects on the propulsion system
        # This would depend on the specific propulsion system implementation
        
        return True
        
    def update_environment(self, env_config: Dict[str, Any]) -> None:
        """Update environment configuration during simulation."""
        # Update environment model
        self.environment = EnvironmentModel(env_config)
        
        # Record environment change event
        self.events.append({
            "time": self.simulation_time,
            "type": "environment_change",
            "config": env_config
        })
        
    def step(self) -> Dict[str, Any]:
        """Execute a single simulation step."""
        if not self.is_running or self.is_paused:
            return {"status": "not_running"}
            
        # Update simulation time
        self.simulation_time += self.config.time_step
        self.current_step += 1
        
        # Check if simulation should end
        if self.simulation_time >= self.config.duration:
            self.stop_simulation()
            return {"status": "completed"}
            
        # Process each propulsion system
        results = {}
        for system_id, system in self.propulsion_systems.items():
            # Get current position (simplified for this example)
            position = np.array([0.0, 0.0, 1000.0])  # Example altitude of 1000m
            
            # Get flight conditions at current position
            flight_conditions = self.get_flight_conditions(position)
            
            # Calculate system performance
            performance = system.calculate_performance(flight_conditions)
            
            # Apply optimization if available
            if self.optimization_integrator:
                # Simple constraints for optimization
                from src.propulsion.optimization import OptimizationConstraints
                
                # Get the max fuel consumption as a float
                fuel_consumption_spec = system.get_specifications().fuel_consumption_curve.get("max", 10.0)
                max_fuel_consumption = fuel_consumption_spec[0] if isinstance(fuel_consumption_spec, list) else fuel_consumption_spec
                
                constraints = OptimizationConstraints(
                    max_power=system.get_specifications().power_rating,
                    max_temperature=system.get_specifications().thermal_limits.get("max_operating", 1000.0),
                    min_efficiency=0.7,
                    max_fuel_consumption=max_fuel_consumption
                )
                
                # Run optimization
                opt_result = self.optimization_integrator.optimize(
                    system_id, flight_conditions, constraints
                )
                
                # Record optimization result
                if opt_result.get("success", False):
                    performance["optimized"] = True
                    performance["optimization_improvement"] = opt_result.get("improvement", {})
            
            # Record data if it's time
            if self.current_step % int(self.config.record_interval / self.config.time_step) == 0:
                self.simulation_data[system_id].append({
                    "time": self.simulation_time,
                    "performance": performance,
                    "flight_conditions": flight_conditions
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
        
        while self.is_running:
            self.step()
            
            # Optional: add a small delay to prevent CPU hogging in real applications
            # time.sleep(0.001)
            
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
            
            # Extract time series for key metrics
            time_series = {
                "time": [d["time"] for d in data],
                "thrust": [d["performance"].get("thrust", 0) for d in data],
                "efficiency": [d["performance"].get("efficiency", 0) for d in data],
                "temperature": [d["performance"].get("temperature", 0) for d in data],
                "fuel_consumption": [d["performance"].get("fuel_consumption", 0) for d in data]
            }
            
            system_results[system_id] = {
                "average_performance": avg_performance,
                "time_series": time_series,
                "data_points": len(data)
            }
            
        return {
            "simulation_config": {
                "mode": self.config.mode.name,
                "duration": self.config.duration,
                "time_step": self.config.time_step
            },
            "events": self.events,
            "system_results": system_results,
            "total_steps": self.current_step
        }


class WindTunnelSimulation(PropulsionSimulationEnvironment):
    """Specialized simulation for wind tunnel testing."""
    
    def __init__(self, config: SimulationConfig, wind_speeds: List[float]):
        """Initialize wind tunnel simulation."""
        super().__init__(config)
        self.wind_speeds = wind_speeds
        self.current_wind_speed_index = 0
        
    def step(self) -> Dict[str, Any]:
        """Execute a wind tunnel simulation step with changing wind speeds."""
        # Change wind speed periodically
        if self.current_step % 100 == 0 and self.wind_speeds:
            # Cycle through wind speeds
            self.current_wind_speed_index = (self.current_wind_speed_index + 1) % len(self.wind_speeds)
            wind_speed = self.wind_speeds[self.current_wind_speed_index]
            
            # Update environment with new wind
            new_env_config = self.config.environment_config.copy() if self.config.environment_config else {}
            new_env_config["wind"] = np.array([wind_speed, 0.0, 0.0])  # Wind in x direction
            self.update_environment(new_env_config)
            
        # Continue with normal step
        return super().step()


class ExtremeTempSimulation(PropulsionSimulationEnvironment):
    """Specialized simulation for extreme temperature testing."""
    
    def __init__(self, config: SimulationConfig, temp_range: Tuple[float, float]):
        """Initialize extreme temperature simulation."""
        super().__init__(config)
        self.min_temp, self.max_temp = temp_range
        self.temp_step = (self.max_temp - self.min_temp) / (self.config.duration / 50)
        self.current_temp = self.min_temp
        
    def step(self) -> Dict[str, Any]:
        """Execute a simulation step with gradually changing temperature."""
        # Update temperature every 50 steps
        if self.current_step % 50 == 0:
            # Increase temperature gradually
            self.current_temp = min(self.max_temp, self.current_temp + self.temp_step)
            
            # Update environment with new temperature
            new_env_config = self.config.environment_config.copy() if self.config.environment_config else {}
            new_env_config["temperature"] = self.current_temp
            self.update_environment(new_env_config)
            
        # Continue with normal step
        return super().step()


def create_simulation_environment(sim_type: str, **kwargs) -> PropulsionSimulationEnvironment:
    """Factory function to create appropriate simulation environment."""
    # Create basic configuration
    config = SimulationConfig(
        mode=SimulationMode[sim_type.upper()] if sim_type.upper() in SimulationMode.__members__ else SimulationMode.STATIC,
        duration=kwargs.get("duration", 300.0),
        time_step=kwargs.get("time_step", 0.1),
        record_interval=kwargs.get("record_interval", 1.0),
        environment_config=kwargs.get("environment_config", None),
        failure_scenarios=kwargs.get("failure_scenarios", None),
        visualization_enabled=kwargs.get("visualization_enabled", False)
    )
    
    # Create appropriate simulation type
    if sim_type.upper() == "WIND_TUNNEL":
        return WindTunnelSimulation(config, kwargs.get("wind_speeds", [10.0, 20.0, 30.0]))
    elif sim_type.upper() == "EXTREME":
        return ExtremeTempSimulation(config, kwargs.get("temp_range", (223.15, 373.15)))  # -50°C to 100°C
    else:
        return PropulsionSimulationEnvironment(config)
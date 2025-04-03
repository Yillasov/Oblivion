import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import time

from src.airframe.base import AirframeBase
from src.control.decision.integration import DecisionControlIntegration
from .airframe_sim import AirframeSimulation
from .environment import EnvironmentModel

class SimulationRunner:
    """Runner for physics-based airframe simulations."""
    
    def __init__(self, 
                airframe: AirframeBase, 
                control_system: Optional[DecisionControlIntegration] = None,
                sim_config: Dict[str, Any] = {}):
        if sim_config is None:
            sim_config = {}
        
        self.airframe = airframe
        self.control_system = control_system
        
        # Create simulation components
        self.airframe_sim = AirframeSimulation(airframe, sim_config)
        self.environment = EnvironmentModel(sim_config.get("environment", {}))
        
        self.time_step = sim_config.get("time_step", 0.01)
        self.real_time = sim_config.get("real_time", False)
        self.mission_params = sim_config.get("mission_params", {})
        
        self.simulation_time = 0.0
        self.simulation_steps = 0
        self.state_history = []
    
    def run_step(self, control_inputs: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Run a single simulation step."""
        # Get control inputs from control system if available
        if control_inputs is None:
            if self.control_system is not None:
                # Convert simulation state to sensor data format
                sensor_data = self._state_to_sensor_data()
                
                # Get control outputs from decision-making system
                control_outputs = self.control_system.update(
                    sensor_data, self.mission_params, self.time_step
                )
                
                # Convert control outputs to control inputs format
                control_inputs = self._outputs_to_inputs(control_outputs)
            else:
                control_inputs = {}
        
        # Update environment conditions
        position = self.airframe_sim.state["position"]
        atmos_conditions = self.environment.get_atmospheric_conditions(position)
        wind_vector = self.environment.get_wind_vector(position)
        
        # Apply wind to airframe velocity for relative airspeed
        velocity = self.airframe_sim.state["velocity"]
        self.airframe_sim.state["velocity"] = velocity - wind_vector
        
        # Update airframe simulation
        new_state = self.airframe_sim.update(control_inputs)
        
        # Restore absolute velocity
        self.airframe_sim.state["velocity"] = new_state["velocity"] + wind_vector
        
        # Update simulation time
        self.simulation_time += self.time_step
        self.simulation_steps += 1
        
        # Store state history
        self.state_history.append(new_state.copy())
        if len(self.state_history) > 1000:  # Limit history size
            self.state_history.pop(0)
        
        # Sleep if real-time simulation is enabled
        if self.real_time:
            time.sleep(self.time_step)
        
        return new_state
    
    def run_simulation(self, 
                      duration: float, 
                      control_callback: Optional[Callable] = None) -> List[Dict[str, np.ndarray]]:
        """Run simulation for specified duration."""
        steps = int(duration / self.time_step)
        results = []
        
        for _ in range(steps):
            # Get control inputs from callback if provided
            control_inputs = None
            if control_callback is not None:
                control_inputs = control_callback(self.airframe_sim.state, self.simulation_time)
            
            # Run simulation step
            state = self.run_step(control_inputs)
            results.append(state.copy())
        
        return results
    
    def _state_to_sensor_data(self) -> Dict[str, np.ndarray]:
        """Convert simulation state to sensor data format."""
        state = self.airframe_sim.state
        
        # Create sensor data dictionary
        sensor_data = {
            "position": np.array(state["position"]),
            "velocity": np.array(state["velocity"]),
            "acceleration": np.array(state["acceleration"]),
            "orientation": np.array(state["orientation"]),
            "angular_velocity": np.array(state["angular_velocity"]),
            "altitude": np.array([state["position"][2]]),
            "airspeed": np.array([np.linalg.norm(state["velocity"])]),
            "vertical_speed": np.array([state["velocity"][2]])
        }
        
        return sensor_data
    
    def _outputs_to_inputs(self, control_outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Convert control system outputs to simulation control inputs."""
        control_inputs = {}
        
        # Map control outputs to simulation inputs
        if "roll_command" in control_outputs:
            control_inputs["aileron"] = float(control_outputs["roll_command"][0])
        
        if "pitch_command" in control_outputs:
            control_inputs["elevator"] = float(control_outputs["pitch_command"][0])
        
        if "yaw_command" in control_outputs:
            control_inputs["rudder"] = float(control_outputs["yaw_command"][0])
        
        if "throttle_command" in control_outputs:
            control_inputs["throttle"] = float(control_outputs["throttle_command"][0])
        
        return control_inputs
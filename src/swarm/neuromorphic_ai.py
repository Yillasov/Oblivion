#!/usr/bin/env python3
"""
Neuromorphic AI for real-time decision-making in UCAVs.
Utilizes spiking neural networks for sensor data processing and decision-making.
Optimized for deployment on neuromorphic hardware platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import time

# Import hardware interfaces
from src.core.hardware.neuromorphic_interface import NeuromorphicProcessor
from src.core.neuromorphic.snn import SpikingNetwork, NeuronPopulation, SynapticConnection

logger = logging.getLogger(__name__)

class SpikingNeuron:
    """Represents a single spiking neuron."""
    
    def __init__(self, threshold: float = 1.0, decay: float = 0.9, refractory_period: float = 0.0):
        self.potential = 0.0
        self.threshold = threshold
        self.decay = decay
        self.refractory_period = refractory_period
        self.last_spike_time = 0.0
        self.spike_history = []
    
    def integrate(self, input_signal: float, current_time: Optional[float] = None, duration_ms: float = 1.0) -> bool:
        """Integrate input signal and determine if neuron spikes."""
        if current_time is None:
            current_time = time.time()
            
        # Check if in refractory period
        if self.refractory_period > 0 and current_time - self.last_spike_time < self.refractory_period:
            return False
            
        # Scale input by duration
        scaled_input = input_signal * (duration_ms / 1000.0)
        self.potential += scaled_input
        
        if self.potential >= self.threshold:
            self.potential = 0.0  # Reset potential after spike
            self.last_spike_time = current_time
            self.spike_history.append(current_time)
            return True
            
        self.potential *= self.decay  # Decay potential
        return False

class NeuromorphicAI:
    """Neuromorphic AI model using spiking neural networks."""
    
    def __init__(self, num_neurons: int = 64, hardware_type: str = ""):
        self.num_neurons = num_neurons
        self.hardware_type = hardware_type
        self.hardware = None
        self.network = None
        
        # Create neurons for software simulation
        self.neurons = [SpikingNeuron(
            threshold=np.random.uniform(0.8, 1.2),  # Varied thresholds
            decay=np.random.uniform(0.85, 0.95),    # Varied decay rates
            refractory_period=0.001                 # 1ms refractory period
        ) for _ in range(num_neurons)]
        
        # Create connection matrix (sparse random connections)
        self.connections = np.zeros((num_neurons, num_neurons))
        connection_probability = 0.2
        for i in range(num_neurons):
            for j in range(num_neurons):
                if i != j and np.random.random() < connection_probability:
                    self.connections[i, j] = np.random.uniform(0.1, 0.5)
        
        # Decision thresholds for different actions
        self.decision_thresholds = {
            "engage": 0.6,
            "evade": 0.4,
            "search": 0.3,
            "defend": 0.5
        }
        
        # Initialize hardware if specified
        if hardware_type:
            self._initialize_hardware()
    
    def _initialize_hardware(self) -> bool:
        """Initialize neuromorphic hardware for SNN execution."""
        try:
            # Create network for hardware deployment
            self.network = SpikingNetwork(name="UCAV_Decision_Network")
            
            # Create neuron populations
            input_pop = NeuronPopulation(size=16, neuron_model='lif')
            hidden_pop = NeuronPopulation(size=32, neuron_model='izhikevich')
            output_pop = NeuronPopulation(size=4, neuron_model='lif')
            
            # Add populations to network
            input_idx = self.network.add_population(input_pop)
            hidden_idx = self.network.add_population(hidden_pop)
            output_idx = self.network.add_population(output_pop)
            
            # Create connections between populations
            input_hidden = SynapticConnection(input_pop, hidden_pop)
            input_hidden.connect_random(probability=0.3, weight_range=(0.1, 0.5))
            
            hidden_output = SynapticConnection(hidden_pop, output_pop)
            hidden_output.connect_random(probability=0.5, weight_range=(0.2, 0.8))
            
            # Add connections to network
            self.network.add_connection(input_hidden)
            self.network.add_connection(hidden_output)
            
            logger.info(f"Created SNN with {input_pop.size + hidden_pop.size + output_pop.size} neurons")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize hardware: {str(e)}")
            return False
    
    def deploy_to_hardware(self, hardware: NeuromorphicProcessor) -> bool:
        """Deploy the SNN to neuromorphic hardware."""
        if not self.network:
            logger.error("No network created for hardware deployment")
            return False
            
        try:
            # Allocate network on hardware
            success = self.network.allocate_on_hardware(hardware)
            if success:
                self.hardware = hardware
                logger.info(f"Successfully deployed SNN to {hardware.get_hardware_info()['processor_type']} hardware")
            return success
            
        except Exception as e:
            logger.error(f"Failed to deploy to hardware: {str(e)}")
            return False
    
    def process_sensor_data(self, sensor_data: Dict[str, float], use_hardware: bool = False) -> List[bool]:
        """Process sensor data and return neuron spikes."""
        if use_hardware and self.hardware:
            return self._hardware_process(sensor_data)
        
        # Software simulation
        spikes = []
        current_time = time.time()
        
        # Normalize sensor data
        sensor_values = list(sensor_data.values())
        if sensor_values:
            max_val = max(max(sensor_values), 1.0)  # Avoid division by zero
            normalized_data = {k: v/max_val for k, v in sensor_data.items()}
        else:
            normalized_data = sensor_data
        
        # First layer processing
        for i, neuron in enumerate(self.neurons):
            # Weight the input based on neuron index
            input_signal = 0
            for j, (sensor, value) in enumerate(normalized_data.items()):
                # Simple weight based on neuron and sensor indices
                weight = 0.5 + 0.5 * np.sin(i * j / 10)
                input_signal += value * weight
            
            # Process through neuron
            spike = neuron.integrate(input_signal, current_time, duration_ms=1.0)
            spikes.append(spike)
            
            # Propagate spikes to connected neurons
            if spike:
                for j in range(self.num_neurons):
                    if self.connections[i, j] > 0:
                        # Delayed processing in next cycle
                        self.neurons[j].potential += self.connections[i, j]
        
        return spikes
    
    def _hardware_process(self, sensor_data: Dict[str, float]) -> List[bool]:
        """Process sensor data using neuromorphic hardware."""
        if not self.hardware or not self.network:
            logger.warning("Hardware not available, falling back to software simulation")
            return self.process_sensor_data(sensor_data, use_hardware=False)
        
        try:
            # Convert sensor data to spike inputs
            input_spikes = []
            for sensor_name, value in sensor_data.items():
                # Simple rate coding - higher values = more spikes
                spike_count = int(value * 10)  # Scale to reasonable number
                input_spikes.append([0.001 * i for i in range(spike_count)])  # 1ms intervals
            
            # Set as input to the network
            input_population = self.network.populations[0]
            input_indices = list(range(min(len(input_spikes), input_population.size)))
            input_population.set_inputs(input_indices, input_spikes[:len(input_indices)])
            
            # Run simulation on hardware - fixed parameter name
            self.hardware.run_simulation(duration_ms=100.0)  # 100ms simulation
            
            # Get output spikes
            output_population = self.network.populations[-1]
            output_spikes = [len(spikes) > 0 for spikes in output_population.output_spikes]
            
            return output_spikes
            
        except Exception as e:
            logger.error(f"Hardware processing error: {str(e)}")
            return [False] * 4  # Default to no spikes
    
    def make_decision(self, spikes: List[bool]) -> Tuple[str, float]:
        """Make a decision based on neuron spikes with confidence level."""
        if not spikes:
            return "unknown", 0.0
            
        # Calculate spike ratios for different neuron groups
        spike_count = sum(spikes)
        total_neurons = len(spikes)
        
        if total_neurons == 0:
            return "unknown", 0.0
            
        # Overall activity level
        activity = spike_count / total_neurons
        
        # Analyze spike patterns in different regions
        region_size = max(1, total_neurons // 4)
        region_activities = []
        
        for i in range(0, total_neurons, region_size):
            end_idx = min(i + region_size, total_neurons)
            region_spike_count = sum(spikes[i:end_idx])
            region_activity = region_spike_count / (end_idx - i)
            region_activities.append(region_activity)
        
        # Make decision based on activity patterns
        if len(region_activities) >= 4:
            # Simple decision logic based on region activities
            if region_activities[0] > self.decision_thresholds["engage"]:
                return "engage", region_activities[0]
            elif region_activities[1] > self.decision_thresholds["evade"]:
                return "evade", region_activities[1]
            elif region_activities[2] > self.decision_thresholds["defend"]:
                return "defend", region_activities[2]
            elif region_activities[3] > self.decision_thresholds["search"]:
                return "search", region_activities[3]
        
        # Fallback to simple threshold on overall activity
        if activity > 0.6:
            return "engage", activity
        elif activity > 0.4:
            return "defend", activity
        elif activity > 0.2:
            return "evade", activity
        else:
            return "search", activity

# Example usage
if __name__ == "__main__":
    # Create AI model
    ai_model = NeuromorphicAI(num_neurons=64)
    
    # Process sample sensor data
    sensor_data = {
        "radar_front": 0.8,
        "radar_left": 0.2,
        "radar_right": 0.1,
        "radar_back": 0.0,
        "ir_signature": 0.7,
        "distance": 0.5,
        "relative_speed": 0.3
    }
    
    # Process data and make decision
    spikes = ai_model.process_sensor_data(sensor_data)
    decision, confidence = ai_model.make_decision(spikes)
    
    logger.info(f"Decision made: {decision} (confidence: {confidence:.2f})")
"""
Spiking Neural Network Base Classes

This module provides base classes for creating and simulating spiking neural networks
that can be deployed on neuromorphic hardware.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import logging
from collections import defaultdict

from .primitives import (
    NeuronModel, 
    LIFNeuron, 
    IzhikevichNeuron, 
    SynapticPlasticityRule,
    STDPRule,
    NetworkTopology,
    create_neuron_model,
    create_plasticity_rule,
    create_network_topology
)

from ..hardware.neuromorphic_interface import NeuromorphicProcessor

logger = logging.getLogger(__name__)


class NeuronPopulation:
    """
    A population of neurons with the same model and parameters.
    """
    
    def __init__(self, 
                 size: int, 
                 neuron_model: str = 'lif',
                 neuron_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a population of neurons.
        
        Args:
            size: Number of neurons in the population
            neuron_model: Type of neuron model ('lif', 'izhikevich')
            neuron_params: Parameters for the neuron model
        """
        self.size = size
        self.neuron_model_type = neuron_model
        self.neuron_params = neuron_params or {}
        
        # Create a prototype neuron model
        self.prototype = create_neuron_model(neuron_model, self.neuron_params)
        
        # Hardware-related attributes
        self.hardware_ids = []  # IDs assigned by hardware
        self.is_allocated = False
        
        # Simulation-related attributes
        self.input_spikes = [[] for _ in range(size)]  # Input spike times for each neuron
        self.output_spikes = [[] for _ in range(size)]  # Output spike times for each neuron
        
        logger.info(f"Created neuron population with {size} {neuron_model} neurons")
    
    def set_inputs(self, neuron_indices: List[int], spike_times: List[List[float]]) -> None:
        """
        Set input spike times for specific neurons in the population.
        
        Args:
            neuron_indices: Indices of neurons to receive input
            spike_times: List of spike time lists, one per neuron
        """
        if len(neuron_indices) != len(spike_times):
            raise ValueError("Number of neuron indices must match number of spike trains")
        
        for i, idx in enumerate(neuron_indices):
            if idx < 0 or idx >= self.size:
                raise ValueError(f"Neuron index {idx} out of range [0, {self.size-1}]")
            self.input_spikes[idx] = sorted(spike_times[i])
    
    def get_hardware_params(self) -> Dict[str, Any]:
        """
        Get hardware-compatible parameters for this neuron population.
        
        Returns:
            Dict[str, Any]: Hardware parameters
        """
        # Map our neuron model parameters to hardware-compatible parameters
        if self.neuron_model_type == 'lif':
            return {
                'threshold': self.neuron_params.get('threshold', 1.0),
                'decay': np.exp(-1.0 / self.neuron_params.get('tau', 10.0)),  # Convert tau to decay
                'reset_mode': 0  # Absolute reset
            }
        elif self.neuron_model_type == 'izhikevich':
            # Simplified mapping for Izhikevich model
            return {
                'threshold': 30.0,  # Fixed threshold for Izhikevich
                'reset_value': self.neuron_params.get('c', -65.0),
                'a': self.neuron_params.get('a', 0.02),
                'b': self.neuron_params.get('b', 0.2),
                'c': self.neuron_params.get('c', -65.0),
                'd': self.neuron_params.get('d', 8.0)
            }
        else:
            return self.neuron_params


class SynapticConnection:
    """
    A synaptic connection between two neuron populations.
    """
    
    def __init__(self, 
                 pre_population: NeuronPopulation,
                 post_population: NeuronPopulation,
                 topology: str = 'random',
                 topology_params: Optional[Dict[str, Any]] = None,
                 plasticity_rule: Optional[str] = None,
                 plasticity_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a synaptic connection between two populations.
        
        Args:
            pre_population: Pre-synaptic neuron population
            post_population: Post-synaptic neuron population
            topology: Type of connection topology ('fully_connected', 'random')
            topology_params: Parameters for the connection topology
            plasticity_rule: Type of plasticity rule (None, 'stdp')
            plasticity_params: Parameters for the plasticity rule
        """
        self.pre_population = pre_population
        self.post_population = post_population
        self.topology_type = topology
        self.topology_params = topology_params or {}
        
        # Create the topology
        self.topology = create_network_topology(topology, self.topology_params)
        
        # Generate the connections
        self.connections = self.topology.generate_connections(
            pre_population.size, post_population.size
        )
        
        # Create plasticity rule if specified
        self.plasticity_rule = None
        if plasticity_rule:
            self.plasticity_rule = create_plasticity_rule(
                plasticity_rule, plasticity_params or {}
            )
        
        # Hardware-related attributes
        self.hardware_ids = []  # IDs assigned by hardware
        self.is_allocated = False
        
        logger.info(f"Created {topology} connection from population of {pre_population.size} to "
                   f"population of {post_population.size} with {len(self.connections)} synapses")
    
    def get_connection_list(self) -> List[Tuple[int, int, float]]:
        """
        Get the list of connections as (pre_id, post_id, weight) tuples.
        
        Returns:
            List[Tuple[int, int, float]]: List of connections
        """
        return self.connections
    
    def update_weights(self, current_time: float) -> None:
        """
        Update synaptic weights based on plasticity rule.
        
        Args:
            current_time: Current simulation time
        """
        if not self.plasticity_rule:
            return
        
        # Update each connection weight based on pre and post-synaptic activity
        updated_connections = []
        
        for pre_id, post_id, weight in self.connections:
            pre_spikes = self.pre_population.output_spikes[pre_id]
            post_spikes = self.post_population.output_spikes[post_id]
            
            new_weight = self.plasticity_rule.update_weight(
                weight, pre_spikes, post_spikes, current_time
            )
            
            updated_connections.append((pre_id, post_id, new_weight))
        
        self.connections = updated_connections


class SpikingNetwork:
    """
    A spiking neural network composed of neuron populations and synaptic connections.
    """
    
    def __init__(self, name: str = "SNN"):
        """
        Initialize a spiking neural network.
        
        Args:
            name: Name of the network
        """
        self.name = name
        self.populations = []
        self.connections = []
        self.hardware = None
        self.simulation_time = 0.0
        
        logger.info(f"Created spiking neural network: {name}")
    
    def add_population(self, population: NeuronPopulation) -> int:
        """
        Add a neuron population to the network.
        
        Args:
            population: Neuron population to add
            
        Returns:
            int: Index of the added population
        """
        self.populations.append(population)
        return len(self.populations) - 1
    
    def add_connection(self, connection: SynapticConnection) -> int:
        """
        Add a synaptic connection to the network.
        
        Args:
            connection: Synaptic connection to add
            
        Returns:
            int: Index of the added connection
        """
        self.connections.append(connection)
        return len(self.connections) - 1
    
    def allocate_on_hardware(self, hardware: NeuromorphicProcessor) -> bool:
        """
        Allocate the network on neuromorphic hardware.
        
        Args:
            hardware: Neuromorphic processor to allocate on
            
        Returns:
            bool: True if allocation was successful, False otherwise
        """
        if not hardware:
            logger.error("No hardware specified for allocation")
            return False
        
        self.hardware = hardware
        
        # Allocate neurons for each population
        for population in self.populations:
            hw_params = population.get_hardware_params()
            neuron_ids = hardware.allocate_neurons(population.size, hw_params)
            
            if len(neuron_ids) != population.size:
                logger.error(f"Failed to allocate all neurons for population")
                return False
            
            population.hardware_ids = neuron_ids
            population.is_allocated = True
            
            logger.info(f"Allocated {population.size} neurons on hardware")
        
        # Allocate synapses for each connection
        for connection in self.connections:
            # Map local indices to hardware IDs
            hw_connections = []
            
            for pre_id, post_id, weight in connection.get_connection_list():
                hw_pre_id = connection.pre_population.hardware_ids[pre_id]
                hw_post_id = connection.post_population.hardware_ids[post_id]
                hw_connections.append((hw_pre_id, hw_post_id, weight))
            
            synapse_ids = hardware.create_synapses(hw_connections)
            
            if len(synapse_ids) != len(hw_connections):
                logger.error(f"Failed to allocate all synapses for connection")
                return False
            
            connection.hardware_ids = synapse_ids
            connection.is_allocated = True
            
            logger.info(f"Allocated {len(synapse_ids)} synapses on hardware")
        
        return True
    
    def set_inputs(self, population_idx: int, neuron_indices: List[int], 
                  spike_times: List[List[float]]) -> bool:
        """
        Set input spike times for specific neurons in a population.
        
        Args:
            population_idx: Index of the target population
            neuron_indices: Indices of neurons to receive input
            spike_times: List of spike time lists, one per neuron
            
        Returns:
            bool: True if inputs were set successfully, False otherwise
        """
        if population_idx < 0 or population_idx >= len(self.populations):
            logger.error(f"Population index {population_idx} out of range")
            return False
        
        population = self.populations[population_idx]
        
        try:
            # Set inputs in the population
            population.set_inputs(neuron_indices, spike_times)
            
            # If allocated on hardware, also set inputs there
            if self.hardware and population.is_allocated:
                # Map local indices to hardware IDs
                hw_neuron_ids = [population.hardware_ids[idx] for idx in neuron_indices]
                return self.hardware.set_spike_input(hw_neuron_ids, spike_times)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set inputs: {str(e)}")
            return False
    
    def run_simulation(self, duration_ms: float) -> bool:
        """
        Run the network simulation for the specified duration.
        
        Args:
            duration_ms: Duration of simulation in milliseconds
            
        Returns:
            bool: True if simulation completed successfully, False otherwise
        """
        if not self.hardware:
            logger.error("Cannot run simulation: Network not allocated on hardware")
            return False
        
        try:
            # Run simulation on hardware
            success = self.hardware.run_simulation(duration_ms)
            
            if not success:
                logger.error("Hardware simulation failed")
                return False
            
            # Update simulation time
            self.simulation_time += duration_ms
            
            # Retrieve output spikes for each population
            for population in self.populations:
                if not population.is_allocated:
                    continue
                
                # Get spike output from hardware
                spike_output = self.hardware.get_spike_output(population.hardware_ids)
                
                # Map hardware IDs back to local indices
                for i, hw_id in enumerate(population.hardware_ids):
                    if hw_id in spike_output:
                        population.output_spikes[i].extend(spike_output[hw_id])
            
            # Update synaptic weights if plasticity is enabled
            for connection in self.connections:
                connection.update_weights(self.simulation_time)
            
            logger.info(f"Completed simulation for {duration_ms} ms")
            return True
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            return False
    
    def get_population_activity(self, population_idx: int) -> Dict[int, List[float]]:
        """
        Get the spike activity for a specific population.
        
        Args:
            population_idx: Index of the target population
            
        Returns:
            Dict[int, List[float]]: Dictionary mapping neuron indices to spike times
        """
        if population_idx < 0 or population_idx >= len(self.populations):
            logger.error(f"Population index {population_idx} out of range")
            return {}
        
        population = self.populations[population_idx]
        
        # Create a dictionary of neuron indices to spike times
        activity = {}
        for i, spikes in enumerate(population.output_spikes):
            if spikes:  # Only include neurons that have spiked
                activity[i] = spikes
        
        return activity
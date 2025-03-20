"""
Molecular Communication System for UCAV platforms.

This module provides a molecular communication system that uses chemical
signals for communication in environments where traditional electromagnetic
communication is challenging or impossible.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
from enum import Enum

from src.communication.base import CommunicationSystem, CommunicationSpecs


class MolecularSignalType(Enum):
    """Types of molecular signals used for communication."""
    PHEROMONE = "pheromone"
    CHEMICAL_GRADIENT = "chemical_gradient"
    PROTEIN_ENCODED = "protein_encoded"
    NANO_PARTICLE = "nano_particle"


class DiffusionModel(Enum):
    """Diffusion models for molecular propagation."""
    FICK_LAW = "fick_law"
    BROWNIAN = "brownian"
    TURBULENT = "turbulent"
    DIRECTED = "directed"


class MolecularSpecs(CommunicationSpecs):
    """Specifications for molecular communication system."""
    
    def __init__(self,
                 signal_type: MolecularSignalType = MolecularSignalType.CHEMICAL_GRADIENT,
                 diffusion_model: DiffusionModel = DiffusionModel.FICK_LAW,
                 molecule_concentration: float = 1.0,
                 diffusion_coefficient: float = 0.5,
                 decay_rate: float = 0.1,
                 detection_threshold: float = 0.2,
                 **kwargs):
        """
        Initialize molecular communication specifications.
        
        Args:
            signal_type: Type of molecular signal
            diffusion_model: Model for molecular diffusion
            molecule_concentration: Initial concentration of molecules
            diffusion_coefficient: Coefficient for diffusion rate
            decay_rate: Rate at which molecules decay
            detection_threshold: Minimum concentration for detection
            **kwargs: Additional communication specs
        """
        super().__init__(**kwargs)
        self.signal_type = signal_type
        self.diffusion_model = diffusion_model
        self.molecule_concentration = molecule_concentration
        self.diffusion_coefficient = diffusion_coefficient
        self.decay_rate = decay_rate
        self.detection_threshold = detection_threshold


class MolecularCommunicationSystem(CommunicationSystem):
    """Molecular communication system implementation."""
    
    def __init__(self, 
                 specs: MolecularSpecs,
                 hardware_interface=None):
        """
        Initialize molecular communication system.
        
        Args:
            specs: Molecular communication specifications
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(specs, hardware_interface)
        self.molecular_specs = specs
        
        # Molecular communication specific parameters
        self.target_position: Optional[np.ndarray] = None
        self.current_concentration = 0.0
        self.signal_propagation_time = 0.0
        self.last_emission_time = 0.0
        
        # Add molecular-specific status fields
        self.status.update({
            "signal_type": specs.signal_type.value,
            "diffusion_model": specs.diffusion_model.value,
            "current_concentration": self.current_concentration,
            "propagation_time": self.signal_propagation_time
        })
    
    def initialize(self) -> bool:
        """Initialize the molecular communication system."""
        try:
            super().initialize()
            
            # Simulate molecular emitter initialization
            self._initialize_molecular_emitter()
            
            # Simulate molecular detector initialization
            self._initialize_molecular_detector()
            
            self.initialized = True
            return True
        except Exception as e:
            self.status["error"] = f"Initialization error: {str(e)}"
            return False
    
    def _initialize_molecular_emitter(self) -> None:
        """Initialize molecular signal emitter."""
        # Simulate hardware initialization
        time.sleep(0.1)
    
    def _initialize_molecular_detector(self) -> None:
        """Initialize molecular signal detector."""
        # Simulate hardware initialization
        time.sleep(0.1)
    
    def establish_link(self, target_data: Dict[str, Any]) -> bool:
        """
        Establish molecular communication link with target.
        
        Args:
            target_data: Target information including position
        
        Returns:
            Success status of link establishment
        """
        if not self.initialized:
            return False
        
        # Extract target position
        if "position" not in target_data:
            self.status["error"] = "Target position not provided"
            return False
        
        # Convert position to numpy array
        self.target_position = np.array(target_data["position"], dtype=float)
        
        # Calculate distance to target
        distance = np.linalg.norm(self.target_position)
        
        # Check if target is within range
        if distance > self.specs.range:
            self.status["error"] = f"Target out of range: {distance:.1f}m > {self.specs.range:.1f}m"
            return False
        
        # Calculate signal propagation time based on diffusion model
        self.signal_propagation_time = self._calculate_propagation_time(distance)
        
        # Calculate initial concentration at target
        self.current_concentration = self._calculate_concentration(distance)
        
        # Update status
        self.status.update({
            "current_concentration": self.current_concentration,
            "propagation_time": self.signal_propagation_time
        })
        
        # Check if concentration is above detection threshold
        if self.current_concentration < self.molecular_specs.detection_threshold:
            self.status["error"] = f"Signal concentration below detection threshold: {self.current_concentration:.3f}"
            return False
        
        self.active = True
        self.last_emission_time = time.time()
        return True
    
    def _calculate_propagation_time(self, distance: float) -> float:
        """
        Calculate signal propagation time based on diffusion model.
        
        Args:
            distance: Distance to target in meters
            
        Returns:
            Propagation time in seconds
        """
        # Different models have different propagation characteristics
        if self.molecular_specs.diffusion_model == DiffusionModel.FICK_LAW:
            # Fick's law: t = x²/2D where D is diffusion coefficient
            return (distance ** 2) / (2 * self.molecular_specs.diffusion_coefficient)
        elif self.molecular_specs.diffusion_model == DiffusionModel.BROWNIAN:
            # Brownian motion: slightly slower than Fick's law
            return 1.2 * (distance ** 2) / (2 * self.molecular_specs.diffusion_coefficient)
        elif self.molecular_specs.diffusion_model == DiffusionModel.TURBULENT:
            # Turbulent diffusion: faster propagation
            return 0.7 * (distance ** 2) / (2 * self.molecular_specs.diffusion_coefficient)
        else:  # Directed diffusion
            # Directed diffusion: fastest propagation
            return 0.5 * (distance ** 2) / (2 * self.molecular_specs.diffusion_coefficient)
    
    def _calculate_concentration(self, distance: float) -> float:
        """
        Calculate molecular concentration at target distance.
        
        Args:
            distance: Distance to target in meters
            
        Returns:
            Molecular concentration at target
        """
        # Base concentration calculation using diffusion equation
        initial_concentration = self.molecular_specs.molecule_concentration
        diffusion_coef = self.molecular_specs.diffusion_coefficient
        decay_rate = self.molecular_specs.decay_rate
        
        # Simplified diffusion equation
        concentration = initial_concentration * np.exp(-distance / (2 * diffusion_coef)) * np.exp(-decay_rate * distance)
        
        return max(0.0, concentration)
    
    def send_data(self, data: Dict[str, Any]) -> bool:
        """
        Send data through molecular communication link.
        
        Args:
            data: Data to send
            
        Returns:
            Success status
        """
        if not self.active:
            return False
        
        # Check if concentration is sufficient
        if self.current_concentration < self.molecular_specs.detection_threshold:
            self.status["error"] = f"Signal concentration too low: {self.current_concentration:.3f}"
            return False
        
        # Encode data into molecular signal
        encoded_signal = self._encode_molecular_signal(data)
        
        # Simulate emission of molecular signal
        emission_success = self._emit_molecular_signal(encoded_signal)
        
        if emission_success:
            self.last_emission_time = time.time()
            
        return emission_success
    
    def _encode_molecular_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode data into molecular signal.
        
        Args:
            data: Data to encode
            
        Returns:
            Encoded molecular signal
        """
        # Different encoding based on signal type
        signal_type = self.molecular_specs.signal_type
        
        if signal_type == MolecularSignalType.PHEROMONE:
            # Pheromone encoding: simple on/off patterns
            return {
                "original_data": data,
                "encoding": "pheromone_pattern",
                "pattern": [1 if i % 2 == 0 else 0 for i in range(len(str(data)))]
            }
        elif signal_type == MolecularSignalType.PROTEIN_ENCODED:
            # Protein encoding: more complex patterns
            return {
                "original_data": data,
                "encoding": "protein_sequence",
                "sequence": [hash(str(k) + str(v)) % 20 for k, v in data.items()]
            }
        elif signal_type == MolecularSignalType.NANO_PARTICLE:
            # Nanoparticle encoding: highest information density
            return {
                "original_data": data,
                "encoding": "nanoparticle",
                "particles": {k: hash(str(v)) % 1000 for k, v in data.items()}
            }
        else:  # Chemical gradient
            # Chemical gradient: concentration-based encoding
            return {
                "original_data": data,
                "encoding": "concentration_gradient",
                "gradient": [0.1 * i for i in range(len(str(data)))]
            }
    
    def _emit_molecular_signal(self, encoded_signal: Dict[str, Any]) -> bool:
        """
        Emit encoded molecular signal.
        
        Args:
            encoded_signal: Encoded molecular signal
            
        Returns:
            Success status
        """
        # Simulate signal emission success probability
        # Higher concentration = higher success rate
        success_probability = min(0.95, self.current_concentration * 0.8)
        
        # Update status
        self.status["last_emission"] = time.time()
        
        return np.random.random() < success_probability
    
    def receive_data(self) -> Dict[str, Any]:
        """
        Receive data from molecular communication link.
        
        Returns:
            Received data or empty dict if no data
        """
        if not self.active:
            return {}
        
        # Check if concentration is above detection threshold
        if self.current_concentration < self.molecular_specs.detection_threshold:
            return {}
        
        # Simulate data reception (based on time since last emission)
        time_since_emission = time.time() - self.last_emission_time
        
        # Check if enough time has passed for signal to propagate
        if time_since_emission < self.signal_propagation_time:
            return {}
        
        # Simulate reception with some probability
        if np.random.random() < 0.3:
            # Create simulated received data
            received_signal = self._simulate_signal_reception()
            
            # Decode the received signal
            decoded_data = self._decode_molecular_signal(received_signal)
            
            # Update status
            self.status["last_reception"] = time.time()
            
            return decoded_data
        
        return {}
    
    def _simulate_signal_reception(self) -> Dict[str, Any]:
        """
        Simulate reception of molecular signal.
        
        Returns:
            Simulated received signal
        """
        # Create a simulated signal based on signal type
        signal_type = self.molecular_specs.signal_type
        
        if signal_type == MolecularSignalType.PHEROMONE:
            return {
                "encoding": "pheromone_pattern",
                "pattern": [1, 0, 1, 0, 1],
                "noise": np.random.random() * 0.1
            }
        elif signal_type == MolecularSignalType.PROTEIN_ENCODED:
            return {
                "encoding": "protein_sequence",
                "sequence": [np.random.randint(0, 20) for _ in range(5)],
                "noise": np.random.random() * 0.1
            }
        elif signal_type == MolecularSignalType.NANO_PARTICLE:
            return {
                "encoding": "nanoparticle",
                "particles": {f"p{i}": np.random.randint(0, 1000) for i in range(3)},
                "noise": np.random.random() * 0.1
            }
        else:  # Chemical gradient
            return {
                "encoding": "concentration_gradient",
                "gradient": [0.1 * i + np.random.random() * 0.05 for i in range(5)],
                "noise": np.random.random() * 0.1
            }
    
    def _decode_molecular_signal(self, received_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode received molecular signal.
        
        Args:
            received_signal: Received molecular signal
            
        Returns:
            Decoded data
        """
        # Simple decoding simulation
        return {
            "message_id": f"mol_{int(time.time())}_{np.random.randint(10000)}",
            "timestamp": time.time(),
            "data_type": "molecular_message",
            "content": {
                "status": "received",
                "signal_type": self.molecular_specs.signal_type.value,
                "signal_strength": self.current_concentration,
                "decoded_data": received_signal
            }
        }
    
    def terminate_link(self) -> bool:
        """Terminate molecular communication link."""
        if not self.active:
            return True
        
        # Reset link parameters
        self.target_position = None
        self.current_concentration = 0.0
        self.signal_propagation_time = 0.0
        
        # Update status
        self.status.update({
            "current_concentration": self.current_concentration,
            "propagation_time": self.signal_propagation_time
        })
        
        self.active = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the molecular communication system."""
        # Update dynamic status fields
        if self.active and self.target_position is not None:
            # Recalculate concentration (it decays over time)
            time_elapsed = time.time() - self.last_emission_time
            decay_factor = np.exp(-self.molecular_specs.decay_rate * time_elapsed)
            self.current_concentration *= decay_factor
            
            self.status.update({
                "current_concentration": self.current_concentration,
                "propagation_time": self.signal_propagation_time
            })
        
        return self.status
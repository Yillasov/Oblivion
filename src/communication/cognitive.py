"""
Cognitive Radio System for UCAV platforms.

This module provides implementation of cognitive radio capabilities
with dynamic spectrum sensing and adaptive transmission.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import time
import numpy as np

from src.communication.base import CommunicationSystem, CommunicationSpecs, CommunicationType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


class SpectrumSensingMode(Enum):
    """Spectrum sensing modes for cognitive radio."""
    ENERGY_DETECTION = "energy_detection"
    MATCHED_FILTER = "matched_filter"
    CYCLOSTATIONARY = "cyclostationary"
    WAVEFORM_DETECTION = "waveform_detection"
    COOPERATIVE = "cooperative"


class CognitiveRadioSpecs(CommunicationSpecs):
    """Specifications for cognitive radio systems."""
    
    def __init__(self,
                 frequency_range: Tuple[float, float] = (30.0, 3000.0),  # MHz
                 bandwidth: float = 20.0,  # MHz
                 sensing_sensitivity: float = -110.0,  # dBm
                 adaptation_speed: float = 50.0,  # ms
                 transmit_power: float = 5.0,  # W
                 weight: float = 3.5,  # kg
                 volume: Optional[Dict[str, float]] = None,
                 power_requirements: float = 45.0,  # W
                 encryption_level: int = 256,
                 resilience_rating: float = 0.9):
        """
        Initialize cognitive radio specifications.
        
        Args:
            frequency_range: Min and max frequency in MHz
            bandwidth: Maximum bandwidth in MHz
            sensing_sensitivity: Minimum detectable signal in dBm
            adaptation_speed: Time to adapt to new conditions in ms
            transmit_power: Maximum transmit power in Watts
            weight: Weight of the system in kg
            volume: Volume specifications in meters
            power_requirements: Power requirements in watts
            encryption_level: Encryption level in bits
            resilience_rating: Resilience to interference/jamming (0-1)
        """
        # Set default volume if not provided
        if volume is None:
            volume = {"length": 0.25, "width": 0.15, "height": 0.08}
            
        super().__init__(
            weight=weight,
            volume=volume,
            power_requirements=power_requirements,
            bandwidth=bandwidth,
            range=100.0,  # km, typical range
            latency=10.0,  # ms
            encryption_level=encryption_level,
            resilience_rating=resilience_rating
        )
        
        self.frequency_range = frequency_range
        self.sensing_sensitivity = sensing_sensitivity
        self.adaptation_speed = adaptation_speed
        self.transmit_power = transmit_power
        self.comm_type = CommunicationType.COGNITIVE_RADIO  # Changed from RADIO to COGNITIVE_RADIO


class CognitiveRadioSystem(CommunicationSystem):
    """Cognitive radio system for UCAV platforms."""
    
    def __init__(self, 
                 specs: CognitiveRadioSpecs,
                 neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """
        Initialize cognitive radio system.
        
        Args:
            specs: Cognitive radio specifications
            neuromorphic_system: Optional neuromorphic system for optimization
        """
        super().__init__(specs, neuromorphic_system)
        self.cognitive_specs = specs
        self.current_frequency = (specs.frequency_range[0] + specs.frequency_range[1]) / 2
        self.current_bandwidth = min(20.0, specs.bandwidth)
        self.current_power = specs.transmit_power / 2
        
        # Spectrum sensing and management
        self.sensing_mode = SpectrumSensingMode.ENERGY_DETECTION
        self.spectrum_map = {}
        self.channel_quality = {}
        self.interference_sources = []
        self.last_scan_time = 0
        self.scan_interval = 1.0  # seconds
        
        # Transmission parameters
        self.modulation_scheme = "adaptive_qam"
        self.coding_rate = 0.75
        self.transmission_queue = []
        
    def initialize(self) -> bool:
        """
        Initialize the cognitive radio system.
        
        Returns:
            Success status
        """
        if self.initialized:
            return True
            
        try:
            # Perform initial spectrum scan
            self._scan_spectrum()
            
            # Find optimal initial parameters
            self._optimize_transmission_parameters()
            
            self.initialized = True
            self.status["initialized"] = True
            return True
            
        except Exception as e:
            self.status["error"] = f"Initialization error: {str(e)}"
            return False
    
    def _scan_spectrum(self) -> Dict[str, Any]:
        """
        Scan the radio spectrum to identify available channels.
        
        Returns:
            Spectrum scan results
        """
        # In a real system, this would perform actual spectrum sensing
        # For simulation, we'll generate synthetic spectrum data
        
        min_freq, max_freq = self.cognitive_specs.frequency_range
        freq_step = 5.0  # MHz
        
        # Generate spectrum occupancy data
        spectrum_data = {}
        current_freq = min_freq
        
        while current_freq <= max_freq:
            # Simulate signal strength at this frequency (lower is less occupied)
            # Random value between -120 dBm (very weak) and -40 dBm (strong)
            signal_strength = -120 + 80 * np.random.beta(0.5, 2.0)
            
            # Determine if frequency is occupied (threshold at -100 dBm)
            is_occupied = signal_strength > -100.0
            
            # Calculate noise floor (random between -130 and -110 dBm)
            noise_floor = -130 + 20 * np.random.random()
            
            # Store data for this frequency
            spectrum_data[current_freq] = {
                "signal_strength": signal_strength,
                "noise_floor": noise_floor,
                "snr": signal_strength - noise_floor,
                "occupied": is_occupied,
                "interference": is_occupied and signal_strength > -90.0
            }
            
            current_freq += freq_step
        
        # Update spectrum map
        self.spectrum_map = spectrum_data
        self.last_scan_time = time.time()
        
        # Identify interference sources
        self.interference_sources = [
            freq for freq, data in spectrum_data.items() 
            if data["interference"]
        ]
        
        return {"spectrum_data": spectrum_data, "timestamp": self.last_scan_time}
    
    def _optimize_transmission_parameters(self) -> Dict[str, Any]:
        """
        Optimize transmission parameters based on spectrum conditions.
        
        Returns:
            Optimization results
        """
        # Find best available frequency band
        best_freq = self.current_frequency
        best_snr = -float('inf')
        best_bandwidth = 5.0  # MHz, minimum bandwidth
        
        min_freq, max_freq = self.cognitive_specs.frequency_range
        
        # Check each potential center frequency
        for center_freq in self.spectrum_map.keys():
            # Skip frequencies too close to edges
            if center_freq < min_freq + 10 or center_freq > max_freq - 10:
                continue
                
            # Check if we can fit our bandwidth around this frequency
            bandwidth = self._find_max_bandwidth(center_freq)
            
            if bandwidth >= best_bandwidth:
                # Calculate average SNR across this bandwidth
                avg_snr = self._calculate_average_snr(center_freq, bandwidth)
                
                if avg_snr > best_snr:
                    best_freq = center_freq
                    best_snr = avg_snr
                    best_bandwidth = bandwidth
        
        # Update current parameters
        self.current_frequency = best_freq
        self.current_bandwidth = best_bandwidth
        
        # Adjust power based on conditions
        self.current_power = min(
            self.cognitive_specs.transmit_power,
            self._calculate_optimal_power(best_snr)
        )
        
        # Adjust modulation and coding based on SNR
        self._adjust_modulation_coding(best_snr)
        
        return {
            "frequency": self.current_frequency,
            "bandwidth": self.current_bandwidth,
            "power": self.current_power,
            "modulation": self.modulation_scheme,
            "coding_rate": self.coding_rate,
            "snr": best_snr
        }
    
    def _find_max_bandwidth(self, center_freq: float) -> float:
        """
        Find maximum available bandwidth around a center frequency.
        
        Args:
            center_freq: Center frequency in MHz
            
        Returns:
            Maximum available bandwidth in MHz
        """
        # Start with minimum bandwidth
        bandwidth = 5.0  # MHz
        max_bandwidth = min(40.0, self.cognitive_specs.bandwidth)
        
        # Increase bandwidth until we hit interference or max bandwidth
        while bandwidth < max_bandwidth:
            # Check lower and upper edges
            lower_edge = center_freq - bandwidth / 2
            upper_edge = center_freq + bandwidth / 2
            
            # Find nearest measured frequencies
            lower_measured = max([f for f in self.spectrum_map.keys() if f <= lower_edge], default=lower_edge)
            upper_measured = min([f for f in self.spectrum_map.keys() if f >= upper_edge], default=upper_edge)
            
            # Check if these frequencies are occupied
            lower_occupied = self.spectrum_map.get(lower_measured, {}).get("occupied", True)
            upper_occupied = self.spectrum_map.get(upper_measured, {}).get("occupied", True)
            
            if lower_occupied or upper_occupied:
                # Hit interference, use previous bandwidth
                return bandwidth - 5.0
                
            # Increase bandwidth
            bandwidth += 5.0
        
        return bandwidth
    
    def _calculate_average_snr(self, center_freq: float, bandwidth: float) -> float:
        """
        Calculate average SNR across a frequency band.
        
        Args:
            center_freq: Center frequency in MHz
            bandwidth: Bandwidth in MHz
            
        Returns:
            Average SNR in dB
        """
        lower_edge = center_freq - bandwidth / 2
        upper_edge = center_freq + bandwidth / 2
        
        # Find all measured frequencies in this range
        freqs_in_range = [
            f for f in self.spectrum_map.keys() 
            if lower_edge <= f <= upper_edge
        ]
        
        if not freqs_in_range:
            return -10.0  # Default SNR if no data
            
        # Calculate average SNR
        total_snr = sum(self.spectrum_map[f]["snr"] for f in freqs_in_range)
        return total_snr / len(freqs_in_range)
    
    def _calculate_optimal_power(self, snr: float) -> float:
        """
        Calculate optimal transmission power based on SNR.
        
        Args:
            snr: Signal-to-noise ratio in dB
            
        Returns:
            Optimal power in Watts
        """
        # Simple power control algorithm
        # Lower power when SNR is good, higher power when SNR is poor
        max_power = self.cognitive_specs.transmit_power
        
        if snr > 20.0:
            # Excellent SNR, use minimum power
            return max_power * 0.2
        elif snr > 10.0:
            # Good SNR, use moderate power
            return max_power * 0.5
        else:
            # Poor SNR, use higher power
            return max_power * 0.8
    
    def _adjust_modulation_coding(self, snr: float) -> None:
        """
        Adjust modulation scheme and coding rate based on SNR.
        
        Args:
            snr: Signal-to-noise ratio in dB
        """
        # Adjust modulation based on SNR
        if snr > 25.0:
            self.modulation_scheme = "256QAM"
            self.coding_rate = 0.9
        elif snr > 18.0:
            self.modulation_scheme = "64QAM"
            self.coding_rate = 0.75
        elif snr > 10.0:
            self.modulation_scheme = "16QAM"
            self.coding_rate = 0.67
        elif snr > 6.0:
            self.modulation_scheme = "QPSK"
            self.coding_rate = 0.5
        else:
            self.modulation_scheme = "BPSK"
            self.coding_rate = 0.33
    
    def establish_link(self, target_data: Dict[str, Any]) -> bool:
        """
        Establish cognitive radio link.
        
        Args:
            target_data: Target information
            
        Returns:
            Success status
        """
        if not self.initialized:
            return False
            
        # Check if spectrum conditions have changed
        current_time = time.time()
        if current_time - self.last_scan_time > self.scan_interval:
            self._scan_spectrum()
            self._optimize_transmission_parameters()
        
        # Extract target information
        target_id = target_data.get("target_id")
        if not target_id:
            self.status["error"] = "No target ID provided"
            return False
            
        # In a real system, this would perform handshaking with the target
        # For simulation, we'll assume success if we have good parameters
        
        # Update status
        self.active = True
        self.status["active"] = True
        self.status["target_id"] = target_id
        self.status["frequency"] = self.current_frequency
        self.status["bandwidth"] = self.current_bandwidth
        self.status["power"] = self.current_power
        self.status["modulation"] = self.modulation_scheme
        
        return True
    
    def terminate_link(self) -> bool:
        """
        Terminate cognitive radio link.
        
        Returns:
            Success status
        """
        if not self.active:
            return True
            
        # Reset status
        self.active = False
        self.status["active"] = False
        self.status["target_id"] = None
        
        return True
    
    def send_data(self, data: Dict[str, Any]) -> bool:
        """
        Send data via cognitive radio link.
        
        Args:
            data: Data to send
            
        Returns:
            Success status
        """
        if not self.active:
            return False
            
        # Add message ID if not present
        if "message_id" not in data:
            data["message_id"] = f"cr_{int(time.time())}_{np.random.randint(1000)}"
            
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = time.time()
            
        # In a real system, this would actually transmit the data
        # For simulation, we'll just add to the queue
        self.transmission_queue.append(data)
        
        # Update status
        self.status["last_transmission"] = time.time()
        
        return True
    
    def receive_data(self) -> Dict[str, Any]:
        """
        Receive data from cognitive radio link.
        
        Returns:
            Received data or empty dict if none available
        """
        if not self.active:
            return {}
            
        # In a real system, this would actually receive data
        # For simulation, we'll generate dummy data occasionally
        
        # 25% chance of receiving data when called
        if np.random.random() < 0.25:
            received_data = {
                "message_id": f"cr_rx_{int(time.time())}_{np.random.randint(1000)}",
                "timestamp": time.time(),
                "source": "ground_control",
                "data_type": "status_request",
                "content": {
                    "request_id": np.random.randint(10000),
                    "priority": np.random.choice(["low", "medium", "high"]),
                    "parameters": ["position", "altitude", "heading", "fuel"]
                }
            }
            
            # Update status
            self.status["last_reception"] = time.time()
            
            return received_data
            
        return {}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the cognitive radio system.
        
        Returns:
            Status information
        """
        # Update status with current parameters
        self.status.update({
            "frequency": self.current_frequency,
            "bandwidth": self.current_bandwidth,
            "power": self.current_power,
            "modulation": self.modulation_scheme,
            "coding_rate": self.coding_rate,
            "queue_size": len(self.transmission_queue),
            "last_scan": self.last_scan_time,
            "interference_sources": len(self.interference_sources)
        })
        
        return self.status
    
    def adapt_to_jamming(self) -> Dict[str, Any]:
        """
        Adapt radio parameters to counter jamming or interference.
        
        Returns:
            Adaptation results
        """
        if not self.active:
            return {"adapted": False, "reason": "Radio not active"}
            
        # Perform emergency spectrum scan
        scan_results = self._scan_spectrum()
        
        # Check for jamming (high power across multiple frequencies)
        jamming_detected = len(self.interference_sources) > 5
        
        if jamming_detected:
            # Use neuromorphic system if available
            if self.neuromorphic_system:
                return self._neuromorphic_jamming_response()
            
            # Otherwise use conventional approach
            # Find frequencies with lowest interference
            available_freqs = [
                freq for freq, data in self.spectrum_map.items()
                if not data["occupied"] and data["snr"] > 0
            ]
            
            if not available_freqs:
                # No clear frequencies, use frequency hopping
                self._enable_frequency_hopping()
                return {
                    "adapted": True,
                    "method": "frequency_hopping",
                    "frequencies": len(self.spectrum_map)
                }
            
            # Select best frequency
            best_freq = min(available_freqs, key=lambda f: self.spectrum_map[f]["signal_strength"])
            
            # Update parameters
            self.current_frequency = best_freq
            self.current_bandwidth = min(5.0, self.current_bandwidth)  # Reduce bandwidth
            self.modulation_scheme = "BPSK"  # Use robust modulation
            self.coding_rate = 0.5  # Use stronger coding
            
            return {
                "adapted": True,
                "method": "frequency_change",
                "new_frequency": best_freq,
                "bandwidth": self.current_bandwidth
            }
        
        return {"adapted": False, "reason": "No jamming detected"}
    
    def _neuromorphic_jamming_response(self) -> Dict[str, Any]:
        """
        Use neuromorphic system to respond to jamming.
        
        Returns:
            Response results
        """
        # Prepare spectrum data for neuromorphic processing
        spectrum_data = np.array([
            [freq, data["signal_strength"], data["noise_floor"], data["snr"]]
            for freq, data in self.spectrum_map.items()
        ])
        
        # Process with neuromorphic system
        result = self.neuromorphic_system.process_data({
            "operation": "cognitive_radio_jamming_response",
            "spectrum_data": spectrum_data,
            "current_params": {
                "frequency": self.current_frequency,
                "bandwidth": self.current_bandwidth,
                "power": self.current_power
            }
        })
        
        if result and "optimized_params" in result:
            # Apply optimized parameters
            params = result["optimized_params"]
            self.current_frequency = params.get("frequency", self.current_frequency)
            self.current_bandwidth = params.get("bandwidth", self.current_bandwidth)
            self.current_power = params.get("power", self.current_power)
            self.modulation_scheme = params.get("modulation", self.modulation_scheme)
            
            return {
                "adapted": True,
                "method": "neuromorphic_optimization",
                "parameters": params
            }
        
        return {"adapted": False, "reason": "Neuromorphic processing failed"}
    
    def _enable_frequency_hopping(self) -> None:
        """Enable frequency hopping mode for jamming resistance."""
        # In a real system, this would configure frequency hopping
        # For simulation, we'll just update the status
        self.status["frequency_hopping"] = True
        self.status["hopping_pattern"] = "neuromorphic_adaptive"
        self.status["hop_interval_ms"] = 50
    
    def optimize_power_consumption(self) -> Dict[str, Any]:
        """
        Optimize power consumption based on link quality.
        
        Returns:
            Optimization results
        """
        if not self.active:
            return {"optimized": False}
        
        # Get current SNR
        current_snr = self._calculate_average_snr(
            self.current_frequency, self.current_bandwidth
        )
        
        # Calculate minimum required power
        min_power = 0.5  # Watts, minimum power
        
        # If SNR is good, reduce power
        if current_snr > 15.0:
            # Good SNR, can reduce power
            optimal_power = max(min_power, self.current_power * 0.7)
            power_saved = self.current_power - optimal_power
            self.current_power = optimal_power
            
            return {
                "optimized": True,
                "power_saved": power_saved,
                "new_power": self.current_power,
                "snr": current_snr
            }
        
        return {"optimized": False, "reason": "SNR too low for power optimization"}
"""
Terahertz Communication System for UCAV platforms.

This module provides implementation of terahertz-based communication capabilities
that offer extremely high bandwidth for short-range secure communications.
"""

import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from src.communication.base import CommunicationSystem, CommunicationSpecs, CommunicationType


class TerahertzBand(Enum):
    """Terahertz frequency bands."""
    LOW = "low_band"      # 0.1-0.3 THz
    MID = "mid_band"      # 0.3-0.65 THz
    HIGH = "high_band"    # 0.65-1.0 THz
    ULTRA = "ultra_band"  # 1.0-10.0 THz


class ModulationScheme(Enum):
    """Modulation schemes for terahertz communication."""
    ASK = "amplitude_shift_keying"
    QAM = "quadrature_amplitude_modulation"
    OFDM = "orthogonal_frequency_division_multiplexing"


@dataclass
class TerahertzSpecs(CommunicationSpecs):
    """Specifications for terahertz communication systems."""
    
    frequency_band: TerahertzBand = TerahertzBand.MID
    modulation_scheme: ModulationScheme = ModulationScheme.QAM
    beam_width_degrees: float = 2.0
    atmospheric_absorption: float = 0.5  # dB/m
    
    def __init__(self,
                 frequency_band: TerahertzBand = TerahertzBand.MID,
                 modulation_scheme: ModulationScheme = ModulationScheme.QAM,
                 beam_width_degrees: float = 2.0,
                 atmospheric_absorption: float = 0.5,
                 weight: float = 0.3,  # kg
                 volume: Optional[Dict[str, float]] = None,
                 power_requirements: float = 8.0,  # W
                 bandwidth: float = 100.0,  # GHz
                 range_km: float = 0.5,  # km
                 latency: float = 0.1,  # ms
                 encryption_level: int = 256,
                 resilience_rating: float = 0.8):
        """
        Initialize terahertz communication specifications.
        """
        if volume is None:
            volume = {"length": 0.08, "width": 0.05, "height": 0.02}
            
        super().__init__(
            weight=weight,
            volume=volume,
            power_requirements=power_requirements,
            bandwidth=bandwidth,
            range=range_km,
            latency=latency,
            encryption_level=encryption_level,
            resilience_rating=resilience_rating
        )
        
        self.frequency_band = frequency_band
        self.modulation_scheme = modulation_scheme
        self.beam_width_degrees = beam_width_degrees
        self.atmospheric_absorption = atmospheric_absorption
        self.comm_type = CommunicationType.TERAHERTZ


class TerahertzSystem(CommunicationSystem):
    """Terahertz communication system for UCAV platforms."""
    
    def __init__(self, 
                 specs: TerahertzSpecs,
                 hardware_interface=None):
        """
        Initialize terahertz communication system.
        
        Args:
            specs: Terahertz communication specifications
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(specs, hardware_interface)
        self.thz_specs = specs
        
        # Terahertz-specific parameters
        self.frequency_ghz = self._get_center_frequency()
        self.beam_direction = [0.0, 0.0]  # [azimuth, elevation] in degrees
        self.target_position: Optional[np.ndarray] = None  # Add type annotation
        
        # Communication state
        self.link_quality = 0.0
        self.signal_to_noise = 0.0
        self.bit_error_rate = 0.0
        
        # Add terahertz-specific status fields
        self.status.update({
            "frequency_band": self.thz_specs.frequency_band.value,
            "modulation_scheme": self.thz_specs.modulation_scheme.value,
            "beam_direction": self.beam_direction,
            "link_quality": self.link_quality,
            "signal_to_noise": self.signal_to_noise,
            "bit_error_rate": self.bit_error_rate,
            "atmospheric_conditions": "normal"
        })
    
    def _get_center_frequency(self) -> float:
        """Get center frequency in GHz based on the selected band."""
        if self.thz_specs.frequency_band == TerahertzBand.LOW:
            return 200.0  # 0.2 THz
        elif self.thz_specs.frequency_band == TerahertzBand.MID:
            return 500.0  # 0.5 THz
        elif self.thz_specs.frequency_band == TerahertzBand.HIGH:
            return 800.0  # 0.8 THz
        else:  # ULTRA
            return 2000.0  # 2.0 THz
    
    def initialize(self) -> bool:
        """Initialize the terahertz communication system."""
        if self.initialized:
            return True
            
        try:
            # Initialize neuromorphic system if available
            if self.neuromorphic_system:
                self.neuromorphic_system.initialize()
                
            # Perform hardware self-test
            self._perform_self_test()
                
            self.initialized = True
            self.status["operational"] = True
            return True
            
        except Exception as e:
            self.status["error"] = f"Initialization error: {str(e)}"
            return False
    
    def _perform_self_test(self) -> None:
        """Perform hardware self-test."""
        # Simulate hardware self-test
        time.sleep(0.1)
    
    def establish_link(self, target_data: Dict[str, Any]) -> bool:
        """
        Establish terahertz communication link with target.
        
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
        
        # Convert position to numpy array to ensure it's iterable
        self.target_position = np.array(target_data["position"], dtype=float)
        
        # Calculate distance to target
        distance = np.linalg.norm(self.target_position)
        
        # Check if target is within range
        if distance > self.thz_specs.range * 1000:  # Convert km to m
            self.status["error"] = f"Target out of range: {distance:.1f}m > {self.thz_specs.range * 1000:.1f}m"
            return False
        
        # Calculate beam direction to target
        self._calculate_beam_direction()
        
        # Calculate link quality based on distance and atmospheric conditions
        self.link_quality = self._calculate_link_quality(float(distance))
        
        # Calculate signal-to-noise ratio
        self.signal_to_noise = self._calculate_snr(float(distance))
        
        # Calculate bit error rate
        self.bit_error_rate = self._calculate_ber()
        
        # Update status
        self.status.update({
            "beam_direction": self.beam_direction,
            "link_quality": self.link_quality,
            "signal_to_noise": self.signal_to_noise,
            "bit_error_rate": self.bit_error_rate
        })
        
        # Check if link quality is sufficient
        if self.link_quality < 0.3:
            self.status["error"] = f"Insufficient link quality: {self.link_quality:.2f}"
            return False
        
        self.active = True
        return True
    
    def _calculate_beam_direction(self) -> None:
        """Calculate beam direction (azimuth, elevation) to target."""
        if self.target_position is None:
            x, y, z = 0.0, 0.0, 0.0
        else:
            x, y, z = self.target_position
        
        # Calculate azimuth (horizontal angle)
        azimuth = np.degrees(np.arctan2(y, x))
        
        # Calculate elevation (vertical angle)
        elevation = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        
        self.beam_direction = [azimuth, elevation]
    
    def _calculate_link_quality(self, distance: float) -> float:
        """
        Calculate link quality based on distance and atmospheric conditions.
        
        Args:
            distance: Distance to target in meters
            
        Returns:
            Link quality from 0.0 to 1.0
        """
        # Calculate atmospheric attenuation
        attenuation = distance * self.thz_specs.atmospheric_absorption
        
        # Calculate link quality (1.0 at 0m, decreasing with distance)
        quality = np.exp(-attenuation / 10.0)
        
        return max(0.0, min(1.0, quality))
    
    def _calculate_snr(self, distance: float) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            distance: Distance to target in meters
            
        Returns:
            Signal-to-noise ratio in dB
        """
        # Simple SNR model based on distance
        base_snr = 40.0  # dB at 1m
        path_loss = 20.0 * np.log10(distance)  # Path loss in dB
        atmospheric_loss = distance * self.thz_specs.atmospheric_absorption
        
        snr = base_snr - path_loss - atmospheric_loss
        return max(0.0, snr)
    
    def _calculate_ber(self) -> float:
        """
        Calculate bit error rate based on SNR.
        
        Returns:
            Bit error rate from 0.0 to 1.0
        """
        # Simple BER model based on SNR
        if self.signal_to_noise <= 0:
            return 0.5  # 50% error rate (random guessing)
        
        # Approximate BER for different modulation schemes
        if self.thz_specs.modulation_scheme == ModulationScheme.ASK:
            # ASK BER approximation
            ber = 0.5 * np.exp(-self.signal_to_noise / 2.0)
        elif self.thz_specs.modulation_scheme == ModulationScheme.QAM:
            # 16-QAM BER approximation
            ber = 0.2 * np.exp(-self.signal_to_noise / 10.0)
        else:  # OFDM
            # OFDM BER approximation
            ber = 0.1 * np.exp(-self.signal_to_noise / 15.0)
        
        return max(1e-10, min(0.5, ber))
    
    def terminate_link(self) -> bool:
        """Terminate terahertz communication link."""
        if not self.active:
            return True
        
        # Reset link parameters
        self.target_position = None
        self.beam_direction = [0.0, 0.0]
        self.link_quality = 0.0
        self.signal_to_noise = 0.0
        self.bit_error_rate = 0.0
        
        # Update status
        self.status.update({
            "beam_direction": self.beam_direction,
            "link_quality": self.link_quality,
            "signal_to_noise": self.signal_to_noise,
            "bit_error_rate": self.bit_error_rate
        })
        
        self.active = False
        return True
    
    def send_data(self, data: Dict[str, Any]) -> bool:
        """
        Send data through terahertz link.
        
        Args:
            data: Data to send
            
        Returns:
            Success status
        """
        if not self.active:
            return False
        
        # Check link quality
        if self.link_quality < 0.3:
            self.status["error"] = f"Link quality too low for transmission: {self.link_quality:.2f}"
            return False
        
        # Simulate data transmission
        # In a real system, this would encode and transmit the data
        
        # Update status
        self.status["last_transmission"] = time.time()
        
        # Simulate transmission success based on BER
        success_probability = 1.0 - min(0.9, self.bit_error_rate * 10)
        return np.random.random() < success_probability
    
    def receive_data(self) -> Dict[str, Any]:
        """
        Receive data from terahertz link.
        
        Returns:
            Received data or empty dict if no data
        """
        if not self.active:
            return {}
        
        # Check link quality
        if self.link_quality < 0.3:
            return {}
        
        # Simulate data reception (20% chance of receiving data)
        if np.random.random() < 0.2:
            # Create simulated received data
            received_data = {
                "message_id": f"thz_{int(time.time())}_{np.random.randint(10000)}",
                "timestamp": time.time(),
                "data_type": "status_update",
                "content": {
                    "status": "operational",
                    "position": [
                        np.random.uniform(-100, 100),
                        np.random.uniform(-100, 100),
                        np.random.uniform(1000, 5000)
                    ]
                }
            }
            
            # Update status
            self.status["last_reception"] = time.time()
            
            return received_data
        
        return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the terahertz communication system."""
        # Update dynamic status fields
        if self.active and self.target_position is not None and hasattr(self.target_position, '__iter__'):
            try:
                # Recalculate link quality (it might change due to atmospheric conditions)
                distance = np.linalg.norm(self.target_position)
                self.link_quality = self._calculate_link_quality(float(distance))
                self.signal_to_noise = self._calculate_snr(float(distance))
                self.bit_error_rate = self._calculate_ber()
                
                self.status.update({
                    "link_quality": self.link_quality,
                    "signal_to_noise": self.signal_to_noise,
                    "bit_error_rate": self.bit_error_rate
                })
            except (TypeError, ValueError) as e:
                self.status["error"] = f"Error calculating link metrics: {str(e)}"
        
        return self.status

# Add this section at the end of the file
if __name__ == "__main__":
    print("Terahertz Communication System Module")
    
    print("\nAvailable Terahertz Bands:")
    for band in TerahertzBand:
        print(f"- {band.name}: {band.value}")
    
    print("\nAvailable Modulation Schemes:")
    for scheme in ModulationScheme:
        print(f"- {scheme.name}: {scheme.value}")
    
    # Example usage
    print("\nExample Terahertz System Configuration:")
    thz_specs = TerahertzSpecs(
        frequency_band=TerahertzBand.MID,
        modulation_scheme=ModulationScheme.OFDM,
        beam_width_degrees=1.5,
        atmospheric_absorption=0.4,
        weight=0.25,
        bandwidth=200.0,  # 200 GHz
        range_km=0.8,     # 800 meters
        latency=0.05      # 0.05 ms
    )
    
    print(f"Frequency Band: {thz_specs.frequency_band.value}")
    print(f"Modulation Scheme: {thz_specs.modulation_scheme.value}")
    print(f"Bandwidth: {thz_specs.bandwidth} GHz")
    print(f"Range: {thz_specs.range} km")
    print(f"Beam Width: {thz_specs.beam_width_degrees}°")
    
    # Create a system instance
    print("\nInitializing terahertz communication system...")
    thz_system = TerahertzSystem(thz_specs)
    success = thz_system.initialize()
    print(f"Initialization {'successful' if success else 'failed'}")
    
    if success:
        # Simulate establishing a link
        print("\nSimulating link establishment...")
        target_data = {
            "position": [100.0, 50.0, 20.0]  # Target at 100m east, 50m north, 20m up
        }
        
        link_success = thz_system.establish_link(target_data)
        print(f"Link establishment {'successful' if link_success else 'failed'}")
        
        if link_success:
            # Get link status
            status = thz_system.get_status()
            print("\nLink Status:")
            print(f"- Link Quality: {status['link_quality']:.2f}")
            print(f"- Signal-to-Noise Ratio: {status['signal_to_noise']:.1f} dB")
            print(f"- Bit Error Rate: {status['bit_error_rate']:.2e}")
            print(f"- Beam Direction: Azimuth {status['beam_direction'][0]:.1f}°, Elevation {status['beam_direction'][1]:.1f}°")

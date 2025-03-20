"""
Laser-based optical communication systems for UCAV platforms.

This module provides implementations for high-bandwidth, directional
optical communication with atmospheric propagation modeling.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass, field

from src.communication.base import CommunicationSystem, CommunicationSpecs, CommunicationType
from src.core.integration.neuromorphic_system import NeuromorphicSystem


class AtmosphericCondition(Enum):
    """Atmospheric conditions affecting laser propagation."""
    CLEAR = "clear"
    LIGHT_HAZE = "light_haze"
    MODERATE_HAZE = "moderate_haze"
    HEAVY_HAZE = "heavy_haze"
    LIGHT_RAIN = "light_rain"
    MODERATE_RAIN = "moderate_rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"
    SNOW = "snow"


@dataclass
class OpticalSystemSpecs:
    """Specifications for optical communication systems."""
    laser_wavelength: float  # Wavelength in nanometers
    laser_power: float  # Power in watts
    beam_divergence: float  # Beam divergence in milliradians
    receiver_aperture: float  # Receiver aperture diameter in cm
    modulation_scheme: str  # Modulation scheme (e.g., "OOK", "PPM", "QAM")
    pointing_accuracy: float  # Pointing accuracy in microradians
    additional_specs: Dict[str, Any] = field(default_factory=dict)


class LaserOpticalSystem(CommunicationSystem):
    """Laser-based optical communication system implementation."""
    
    def __init__(self, 
                 specs: CommunicationSpecs, 
                 optical_specs: OpticalSystemSpecs,
                 hardware_interface=None):
        """
        Initialize laser optical communication system.
        
        Args:
            specs: Base communication specifications
            optical_specs: Optical system specific specifications
            hardware_interface: Interface to neuromorphic hardware
        """
        super().__init__(specs, hardware_interface)
        self.optical_specs = optical_specs
        self.current_atmospheric_condition = AtmosphericCondition.CLEAR
        self.target_coordinates = None
        self.beam_steering_state = {
            "azimuth": 0.0,
            "elevation": 0.0,
            "tracking_error": 0.0,
            "stabilized": False
        }
        
        # Add optical-specific status fields
        self.status.update({
            "beam_locked": False,
            "signal_to_noise_ratio": 0.0,
            "bit_error_rate": 0.0,
            "atmospheric_condition": self.current_atmospheric_condition.value
        })
    
    def establish_link(self, target_data: Dict[str, Any]) -> bool:
        """
        Establish optical communication link with target.
        
        Args:
            target_data: Target information including position
        
        Returns:
            Success status of link establishment
        """
        if not self.initialized:
            return False
        
        # Extract target coordinates
        if "position" not in target_data:
            self.status["error"] = "Target position not provided"
            return False
        
        self.target_coordinates = target_data["position"]
        
        # Perform beam steering to target
        steering_success = self._steer_beam_to_target(self.target_coordinates)
        if not steering_success:
            self.status["error"] = "Failed to steer beam to target"
            return False
        
        # Check atmospheric conditions
        if "atmospheric_condition" in target_data:
            self.current_atmospheric_condition = AtmosphericCondition(target_data["atmospheric_condition"])
        
        # Calculate link budget
        link_margin = self._calculate_link_margin()
        if link_margin <= 0:
            self.status["error"] = f"Insufficient link margin: {link_margin} dB"
            return False
        
        # Update status
        self.status["beam_locked"] = True
        self.status["signal_to_noise_ratio"] = self._calculate_snr()
        self.status["bit_error_rate"] = self._calculate_ber()
        self.active = True
        
        return True
    
    def send_data(self, data: Dict[str, Any]) -> bool:
        """Send data through optical link."""
        if not self.active or not self.status["beam_locked"]:
            return False
        
        # Apply modulation based on scheme
        modulated_data = self._apply_modulation(data)
        
        # Apply error correction
        encoded_data = self._apply_error_correction(modulated_data)
        
        # Simulate transmission through atmosphere
        transmission_success = self._simulate_transmission(encoded_data)
        
        return transmission_success
    
    def receive_data(self) -> Dict[str, Any]:
        """Receive data from optical link."""
        if not self.active or not self.status["beam_locked"]:
            return {"error": "Optical link not active"}
        
        # Simulate reception
        received_signal = self._simulate_reception()
        
        # Apply demodulation
        demodulated_data = self._apply_demodulation(received_signal)
        
        # Apply error correction
        corrected_data = self._apply_error_correction_decoding(demodulated_data)
        
        return corrected_data
    
    def terminate_link(self) -> bool:
        """Terminate optical communication link."""
        if not self.active:
            return False
        
        # Reset beam steering
        self.beam_steering_state = {
            "azimuth": 0.0,
            "elevation": 0.0,
            "tracking_error": 0.0,
            "stabilized": False
        }
        
        # Update status
        self.status["beam_locked"] = False
        self.active = False
        self.target_coordinates = None
        
        return True
    
    def _steer_beam_to_target(self, target_position: List[float]) -> bool:
        """
        Steer laser beam to target position.
        
        Args:
            target_position: 3D coordinates of target
            
        Returns:
            Success status of beam steering
        """
        # Calculate azimuth and elevation to target
        # Simplified calculation for demonstration
        dx, dy, dz = target_position
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance < 0.1:  # Prevent division by zero
            return False
        
        azimuth = np.arctan2(dy, dx)
        elevation = np.arcsin(dz / distance)
        
        # Apply pointing accuracy error
        pointing_error = self.optical_specs.pointing_accuracy * 1e-6  # Convert to radians
        tracking_error = np.random.normal(0, pointing_error)
        
        # Update beam steering state
        self.beam_steering_state = {
            "azimuth": float(azimuth),
            "elevation": float(elevation),
            "tracking_error": float(tracking_error),
            "stabilized": True
        }
        
        # Check if pointing error is acceptable
        max_allowed_error = self.optical_specs.beam_divergence * 1e-3 * 0.5  # Half of beam divergence in radians
        return abs(tracking_error) < max_allowed_error
    
    def _calculate_link_margin(self) -> float:
        """
        Calculate optical link margin in dB.
        
        Returns:
            Link margin in dB
        """
        if self.target_coordinates is None:
            return -100.0
        
        # Calculate distance to target
        distance = np.linalg.norm(self.target_coordinates)
        
        # Calculate free space path loss
        wavelength = self.optical_specs.laser_wavelength * 1e-9  # Convert to meters
        fspl = 20 * np.log10(4 * np.pi * distance / wavelength)
        
        # Calculate atmospheric attenuation
        attenuation = self._get_atmospheric_attenuation(float(distance))
        
        # Calculate received power
        tx_power_dbm = 10 * np.log10(self.optical_specs.laser_power * 1000)  # Convert W to dBm
        rx_aperture_gain = 20 * np.log10(np.pi * self.optical_specs.receiver_aperture * 0.01 / wavelength)  # Convert cm to m
        
        # Account for pointing loss
        pointing_loss = -10 * np.log10(1 - self.beam_steering_state["tracking_error"]**2 / 
                                      (self.optical_specs.beam_divergence * 1e-3)**2)
        
        # Calculate received power
        rx_power = tx_power_dbm + rx_aperture_gain - fspl - attenuation - pointing_loss
        
        # Assume minimum required power is -50 dBm
        return rx_power + 50.0
    
    def _get_atmospheric_attenuation(self, distance: float) -> float:
        """
        Get atmospheric attenuation based on conditions.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Attenuation in dB
        """
        # Attenuation coefficients in dB/km for different conditions
        attenuation_coefficients = {
            AtmosphericCondition.CLEAR: 0.5,
            AtmosphericCondition.LIGHT_HAZE: 3.0,
            AtmosphericCondition.MODERATE_HAZE: 6.0,
            AtmosphericCondition.HEAVY_HAZE: 10.0,
            AtmosphericCondition.LIGHT_RAIN: 5.0,
            AtmosphericCondition.MODERATE_RAIN: 15.0,
            AtmosphericCondition.HEAVY_RAIN: 30.0,
            AtmosphericCondition.FOG: 100.0,
            AtmosphericCondition.SNOW: 50.0
        }
        
        # Get coefficient for current condition
        coefficient = attenuation_coefficients.get(self.current_atmospheric_condition, 0.5)
        
        # Calculate attenuation
        return coefficient * distance / 1000.0  # Convert to km
    
    def _calculate_snr(self) -> float:
        """Calculate signal-to-noise ratio."""
        # Simplified SNR calculation
        link_margin = self._calculate_link_margin()
        return max(0.0, 10.0 + link_margin)
    
    def _calculate_ber(self) -> float:
        """Calculate bit error rate based on SNR."""
        # Simplified BER calculation
        snr = self._calculate_snr()
        if snr <= 0:
            return 1.0
        
        # Different modulation schemes have different BER formulas
        if self.optical_specs.modulation_scheme == "OOK":
            # On-Off Keying
            return 0.5 * np.exp(-snr / 2)
        elif self.optical_specs.modulation_scheme == "PPM":
            # Pulse Position Modulation (simplified)
            return 0.5 * np.exp(-snr / 4)
        else:
            # Default formula
            return np.exp(-snr / 10)
    
    def _apply_modulation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modulation to data."""
        # Simplified modulation
        return {
            "original_data": data,
            "modulation_scheme": self.optical_specs.modulation_scheme,
            "modulated": True
        }
    
    def _apply_demodulation(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply demodulation to received signal."""
        # Simplified demodulation
        if "original_data" in signal:
            return signal["original_data"]
        return {}
    
    def _apply_error_correction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply error correction coding."""
        # Simplified error correction
        return {
            "data": data,
            "error_correction": "applied"
        }
    
    def _apply_error_correction_decoding(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply error correction decoding."""
        # Simplified error correction decoding
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data
    
    def _simulate_transmission(self, data: Dict[str, Any]) -> bool:
        """Simulate transmission through atmosphere."""
        # Calculate probability of successful transmission
        ber = self._calculate_ber()
        success_probability = 1.0 - min(ber * 10, 0.9)  # Scale BER for simulation
        
        # Simulate success/failure
        return np.random.random() < success_probability
    
    def _simulate_reception(self) -> Dict[str, Any]:
        """Simulate reception of data."""
        # Simplified reception simulation
        if not self.active or not self.status["beam_locked"]:
            return {}
        
        # Generate dummy received data
        return {
            "original_data": {"message": "Test optical communication"},
            "signal_strength": self._calculate_snr(),
            "modulation_scheme": self.optical_specs.modulation_scheme
        }
    
    def update_atmospheric_condition(self, condition: AtmosphericCondition) -> None:
        """
        Update current atmospheric condition.
        
        Args:
            condition: New atmospheric condition
        """
        self.current_atmospheric_condition = condition
        self.status["atmospheric_condition"] = condition.value
        
        # Recalculate link parameters
        if self.active and self.target_coordinates is not None:
            self.status["signal_to_noise_ratio"] = self._calculate_snr()
            self.status["bit_error_rate"] = self._calculate_ber()
            
            # Check if link is still viable
            if self._calculate_link_margin() <= 0:
                self.terminate_link()
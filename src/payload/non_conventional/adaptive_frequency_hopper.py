"""
Adaptive Frequency Hopping System implementation for UCAV platforms.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
import time
from enum import Enum

from src.payload.non_conventional.countermeasures import AdaptiveCountermeasure, CountermeasureSpecs
from src.payload.types import CountermeasureType, JammingFrequencyBand


class HoppingMode(Enum):
    """Frequency hopping modes."""
    RANDOM = 0
    PATTERN = 1
    ADAPTIVE = 2
    NEUROMORPHIC = 3


class AdaptiveFrequencyHopper(AdaptiveCountermeasure):
    """
    Advanced adaptive frequency hopping system that uses neuromorphic processing
    to dynamically avoid jamming and detection.
    """
    
    def __init__(self, model: str, hardware_interface=None):
        if model == "AFH-50":
            specs = CountermeasureSpecs(
                weight=45.0,
                volume={"length": 0.5, "width": 0.3, "height": 0.2},
                power_requirements=120.0,
                mounting_points=["fuselage", "wing"],
                countermeasure_type=CountermeasureType.ADAPTIVE_FREQUENCY_HOPPER,
                response_time=0.001,  # Very fast response time
                effectiveness_rating=0.82,
                capacity=500,  # Many uses before maintenance
                coverage_angle=360.0,
                energy_consumption=100.0,
                thermal_signature=0.3,
                stealth_impact=0.2,
                cooldown_time=0.05
            )
        elif model == "AFH-100":
            specs = CountermeasureSpecs(
                weight=60.0,
                volume={"length": 0.6, "width": 0.35, "height": 0.25},
                power_requirements=180.0,
                mounting_points=["fuselage", "internal_bay"],
                countermeasure_type=CountermeasureType.ADAPTIVE_FREQUENCY_HOPPER,
                response_time=0.0005,
                effectiveness_rating=0.90,
                capacity=1000,
                coverage_angle=360.0,
                energy_consumption=150.0,
                thermal_signature=0.35,
                stealth_impact=0.25,
                cooldown_time=0.02
            )
        else:
            raise ValueError(f"Unknown frequency hopper model: {model}")
            
        super().__init__(specs, hardware_interface)
        self.model = model
        
        # Frequency hopping properties
        self.hopping_properties = {
            "mode": HoppingMode.ADAPTIVE,
            "hop_interval": 0.05,  # seconds
            "last_hop_time": 0.0,
            "frequency_bands": self._initialize_frequency_bands(),
            "current_band": JammingFrequencyBand.L_BAND,
            "pattern_length": 64,
            "pattern_index": 0,
            "hopping_pattern": [],
            "adaptive_learning_rate": 0.1,
            "band_effectiveness": {}
        }
        
        # Initialize band effectiveness tracking
        for band in JammingFrequencyBand:
            self.hopping_properties["band_effectiveness"][band] = 0.5  # Initial neutral effectiveness
        
        # Generate initial hopping pattern
        self._generate_hopping_pattern()
    
    def _initialize_frequency_bands(self) -> Dict[JammingFrequencyBand, Dict[str, Any]]:
        """Initialize frequency band properties."""
        bands = {}
        
        # VHF band (30-300 MHz)
        bands[JammingFrequencyBand.VHF] = {
            "range": (30, 300),  # MHz
            "channels": 27,
            "jamming_resistance": 0.6,
            "detection_probability": 0.4,
            "power_efficiency": 0.8
        }
        
        # UHF band (300 MHz-3 GHz)
        bands[JammingFrequencyBand.UHF] = {
            "range": (300, 3000),  # MHz
            "channels": 270,
            "jamming_resistance": 0.65,
            "detection_probability": 0.45,
            "power_efficiency": 0.75
        }
        
        # L-band (1-2 GHz)
        bands[JammingFrequencyBand.L_BAND] = {
            "range": (1000, 2000),  # MHz
            "channels": 100,
            "jamming_resistance": 0.7,
            "detection_probability": 0.5,
            "power_efficiency": 0.7
        }
        
        # S-band (2-4 GHz)
        bands[JammingFrequencyBand.S_BAND] = {
            "range": (2000, 4000),  # MHz
            "channels": 200,
            "jamming_resistance": 0.75,
            "detection_probability": 0.55,
            "power_efficiency": 0.65
        }
        
        # X-band (8-12 GHz)
        bands[JammingFrequencyBand.X_BAND] = {
            "range": (8000, 12000),  # MHz
            "channels": 400,
            "jamming_resistance": 0.8,
            "detection_probability": 0.6,
            "power_efficiency": 0.6
        }
        
        # Ka-band (27-40 GHz)
        bands[JammingFrequencyBand.KA_BAND] = {
            "range": (27000, 40000),  # MHz
            "channels": 1300,
            "jamming_resistance": 0.85,
            "detection_probability": 0.7,
            "power_efficiency": 0.5
        }
        
        # Millimeter wave (40-300 GHz)
        bands[JammingFrequencyBand.MILLIMETER] = {
            "range": (40000, 300000),  # MHz
            "channels": 26000,
            "jamming_resistance": 0.9,
            "detection_probability": 0.8,
            "power_efficiency": 0.4
        }
        
        return bands
    
    def _generate_hopping_pattern(self) -> None:
        """Generate a frequency hopping pattern based on current mode."""
        mode = self.hopping_properties["mode"]
        pattern_length = self.hopping_properties["pattern_length"]
        pattern = []
        
        if mode == HoppingMode.RANDOM:
            # Completely random pattern
            bands = list(JammingFrequencyBand)
            pattern = [bands[np.random.randint(0, len(bands))] for _ in range(pattern_length)]
            
        elif mode == HoppingMode.PATTERN:
            # Deterministic pattern with good coverage
            bands = list(JammingFrequencyBand)
            base_pattern = bands[:7]  # Use first 7 bands
            repeats = pattern_length // len(base_pattern) + 1
            pattern = (base_pattern * repeats)[:pattern_length]
            
        elif mode == HoppingMode.ADAPTIVE or mode == HoppingMode.NEUROMORPHIC:
            # Weighted selection based on effectiveness
            bands = list(self.hopping_properties["band_effectiveness"].keys())
            weights = [self.hopping_properties["band_effectiveness"][band] for band in bands]
            
            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w/total for w in weights]
            else:
                weights = [1.0/len(weights)] * len(weights)
                
            # Use numpy's random choice with indices, then map back to bands
            indices = np.random.choice(len(bands), size=pattern_length, p=weights)
            pattern = [bands[i] for i in indices]
        
        self.hopping_properties["hopping_pattern"] = pattern
        self.hopping_properties["pattern_index"] = 0
    
    def set_hopping_mode(self, mode: HoppingMode) -> bool:
        """
        Set the frequency hopping mode.
        
        Args:
            mode: Hopping mode
            
        Returns:
            Success status
        """
        if not isinstance(mode, HoppingMode):
            return False
            
        self.hopping_properties["mode"] = mode
        self._generate_hopping_pattern()
        return True
    
    def set_hop_interval(self, interval: float) -> bool:
        """
        Set the frequency hopping interval.
        
        Args:
            interval: Hopping interval in seconds
            
        Returns:
            Success status
        """
        if 0.001 <= interval <= 1.0:
            self.hopping_properties["hop_interval"] = interval
            return True
        return False
    
    def deploy(self, target_data: Dict[str, Any]) -> bool:
        """
        Deploy frequency hopping against a target threat.
        
        Args:
            target_data: Data about the target threat
            
        Returns:
            Success status
        """
        # First check if base deployment is successful
        if not super().deploy(target_data):
            return False
        
        # Use neuromorphic processing to optimize hopping
        hopping_result = self.process_data({
            "threat": target_data,
            "computation": "frequency_hopping_optimization",
            "current_mode": self.hopping_properties["mode"],
            "band_effectiveness": self.hopping_properties["band_effectiveness"]
        })
        
        # Update hopping properties based on optimization
        if "optimal_mode" in hopping_result:
            self.hopping_properties["mode"] = hopping_result["optimal_mode"]
            
        if "optimal_interval" in hopping_result:
            self.hopping_properties["hop_interval"] = hopping_result["optimal_interval"]
            
        if "band_weights" in hopping_result:
            for band, weight in hopping_result["band_weights"].items():
                self.hopping_properties["band_effectiveness"][band] = weight
        
        # Generate new hopping pattern
        self._generate_hopping_pattern()
        
        # Set initial frequency band
        self._hop_to_next_frequency()
        
        return True
    
    def _hop_to_next_frequency(self) -> JammingFrequencyBand:
        """
        Hop to the next frequency in the pattern.
        
        Returns:
            New frequency band
        """
        pattern = self.hopping_properties["hopping_pattern"]
        index = self.hopping_properties["pattern_index"]
        
        if not pattern:
            # No pattern, use default
            self.hopping_properties["current_band"] = JammingFrequencyBand.L_BAND
            return self.hopping_properties["current_band"]
        
        # Get next band from pattern
        next_band = pattern[index]
        self.hopping_properties["current_band"] = next_band
        
        # Update pattern index
        self.hopping_properties["pattern_index"] = (index + 1) % len(pattern)
        self.hopping_properties["last_hop_time"] = time.time()
        
        return next_band
    
    def update(self, dt: float, environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update frequency hopping system state over time.
        
        Args:
            dt: Time step in seconds
            environment_data: Environmental data
            
        Returns:
            Updated status
        """
        if not self.status["active"]:
            return self.get_status()
        
        # Check if it's time to hop
        current_time = time.time()
        last_hop_time = self.hopping_properties["last_hop_time"]
        hop_interval = self.hopping_properties["hop_interval"]
        
        if current_time - last_hop_time >= hop_interval:
            # Time to hop to next frequency
            new_band = self._hop_to_next_frequency()
            
            # If we have environment data, update band effectiveness
            if environment_data and "jamming_bands" in environment_data:
                jamming_bands = environment_data["jamming_bands"]
                
                # If current band is being jammed, reduce its effectiveness
                if new_band in jamming_bands:
                    self.hopping_properties["band_effectiveness"][new_band] *= 0.9
                else:
                    # Otherwise increase effectiveness
                    current_effectiveness = self.hopping_properties["band_effectiveness"][new_band]
                    self.hopping_properties["band_effectiveness"][new_band] = min(
                        1.0, current_effectiveness * 1.05)
                
                # If in adaptive or neuromorphic mode, regenerate pattern
                if (self.hopping_properties["mode"] == HoppingMode.ADAPTIVE or 
                    self.hopping_properties["mode"] == HoppingMode.NEUROMORPHIC):
                    self._generate_hopping_pattern()
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current frequency hopping status."""
        status = super().get_status()
        status.update({
            "hopping_properties": self.hopping_properties,
            "current_band": self.hopping_properties["current_band"],
            "hop_interval": self.hopping_properties["hop_interval"],
            "mode": self.hopping_properties["mode"]
        })
        return status
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using neuromorphic computing.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processing results
        """
        base_result = super().process_data(input_data)
        
        computation_type = input_data.get("computation", "")
        
        if computation_type == "frequency_hopping_optimization":
            # Neuromorphic frequency hopping optimization
            threat = input_data.get("threat", {})
            threat_type = threat.get("type", "unknown")
            jamming_capabilities = threat.get("jamming_capabilities", [])
            
            # Determine optimal hopping mode based on threat
            optimal_mode = HoppingMode.ADAPTIVE
            if threat_type == "advanced_jammer":
                optimal_mode = HoppingMode.NEUROMORPHIC
            elif threat_type == "basic_jammer":
                optimal_mode = HoppingMode.PATTERN
            
            # Determine optimal hopping interval
            base_interval = 0.05
            if threat_type == "advanced_jammer":
                # Faster hopping for advanced jammers
                base_interval = 0.01
            elif threat_type == "basic_jammer":
                # Slower hopping is sufficient for basic jammers
                base_interval = 0.1
                
            # Add some randomness to interval to avoid predictability
            jitter = np.random.uniform(-0.005, 0.005)
            optimal_interval = max(0.001, base_interval + jitter)
            
            # Calculate band weights based on threat jamming capabilities
            band_weights = {}
            for band in JammingFrequencyBand:
                # Start with current effectiveness
                weight = input_data.get("band_effectiveness", {}).get(band, 0.5)
                
                # Reduce weight if band is in jamming capabilities
                if band in jamming_capabilities:
                    weight *= 0.5
                else:
                    weight *= 1.2
                
                # Ensure weight is in valid range
                band_weights[band] = min(1.0, max(0.1, weight))
            
            # Add results to base result
            base_result["optimal_mode"] = optimal_mode
            base_result["optimal_interval"] = optimal_interval
            base_result["band_weights"] = band_weights
            base_result["effectiveness"] = sum(band_weights.values()) / len(band_weights)
            
        return base_result
    
    def set_power_level(self, power_level: float) -> bool:
        """
        Set the power level for the frequency hopper.
        
        Args:
            power_level: Power level as a percentage (0-100)
            
        Returns:
            Success status
        """
        if 0 <= power_level <= 100:
            # Adjust hop interval based on power level
            # Higher power allows for faster hopping
            min_interval = 0.001 if self.model == "AFH-100" else 0.005
            max_interval = 0.1
            
            # Power affects hop interval inversely (more power = faster hops)
            hop_interval = max_interval - (power_level / 100.0) * (max_interval - min_interval)
            self.hopping_properties["hop_interval"] = hop_interval
            
            return True
        return False
    
    def set_power(self, power_ratio: float) -> bool:
        """
        Set the power ratio for the frequency hopper.
        
        Args:
            power_ratio: Power ratio (0.0-1.0)
            
        Returns:
            Success status
        """
        return self.set_power_level(power_ratio * 100.0)
    
    def analyze_spectrum(self, spectrum_data: Dict[JammingFrequencyBand, float]) -> Dict[str, Any]:
        """
        Analyze spectrum data to identify jamming and optimize hopping.
        
        Args:
            spectrum_data: Signal strength by frequency band
            
        Returns:
            Analysis results
        """
        # Identify potential jamming
        jamming_bands = []
        for band, signal_strength in spectrum_data.items():
            # High signal strength may indicate jamming
            if signal_strength > 0.7:
                jamming_bands.append(band)
                
                # Reduce effectiveness of jammed bands
                if band in self.hopping_properties["band_effectiveness"]:
                    self.hopping_properties["band_effectiveness"][band] *= 0.8
        
        # Increase effectiveness of clear bands
        for band in JammingFrequencyBand:
            if band not in jamming_bands and band in self.hopping_properties["band_effectiveness"]:
                current = self.hopping_properties["band_effectiveness"][band]
                self.hopping_properties["band_effectiveness"][band] = min(1.0, current * 1.1)
        
        # If in adaptive or neuromorphic mode, regenerate pattern
        if (self.hopping_properties["mode"] == HoppingMode.ADAPTIVE or 
            self.hopping_properties["mode"] == HoppingMode.NEUROMORPHIC):
            self._generate_hopping_pattern()
        
        return {
            "jamming_detected": len(jamming_bands) > 0,
            "jamming_bands": jamming_bands,
            "clear_bands": [b for b in JammingFrequencyBand if b not in jamming_bands],
            "recommended_mode": HoppingMode.NEUROMORPHIC if len(jamming_bands) > 2 else HoppingMode.ADAPTIVE
        }
    
    def reset(self) -> None:
        """Reset the frequency hopper to default state."""
        # Reset the status instead of calling super().reset()
        self.status = {
            "active": False,
            "effectiveness": 0.0,
            "last_deployment_time": 0.0,
            "deployment_count": 0
        }
        
        # Reset band effectiveness to neutral
        for band in JammingFrequencyBand:
            self.hopping_properties["band_effectiveness"][band] = 0.5
            
        # Reset to default mode and interval
        self.hopping_properties["mode"] = HoppingMode.ADAPTIVE
        self.hopping_properties["hop_interval"] = 0.05
        
        # Generate new pattern
        self._generate_hopping_pattern()
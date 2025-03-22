"""
Steganographic Communication System for UCAV platforms.

This module provides implementation of steganographic communication capabilities
that can hide data within various carrier formats for covert transmission.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from src.communication.base import CommunicationSystem, CommunicationSpecs

class CarrierType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    NETWORK = "network"

class EncodingMethod(Enum):
    LSB = "least_significant_bit"
    DCT = "discrete_cosine_transform"

@dataclass
class SteganographicSpecs(CommunicationSpecs):
    carrier_type: CarrierType = CarrierType.IMAGE
    encoding_method: EncodingMethod = EncodingMethod.LSB
    max_payload_size: int = 1024  # bytes

class SteganographicSystem(CommunicationSystem):
    def __init__(self, specs: SteganographicSpecs):
        super().__init__(specs)
        self.specs = specs
        self.carrier = None
        
    def encode(self, data: bytes, carrier: np.ndarray) -> np.ndarray:
        """Encode data into carrier using LSB method."""
        if self.specs.encoding_method == EncodingMethod.LSB:
            return self._encode_lsb(data, carrier)
        raise NotImplementedError("Only LSB encoding implemented")
        
    def decode(self, carrier: np.ndarray) -> bytes:
        """Decode data from carrier."""
        if self.specs.encoding_method == EncodingMethod.LSB:
            return self._decode_lsb(carrier)
        raise NotImplementedError("Only LSB decoding implemented")
        
    def _encode_lsb(self, data: bytes, carrier: np.ndarray) -> np.ndarray:
        """Least Significant Bit encoding."""
        flat = carrier.flatten()
        for i, byte in enumerate(data):
            for bit in range(8):
                idx = i * 8 + bit
                flat[idx] = (flat[idx] & ~1) | ((byte >> bit) & 1)
        return flat.reshape(carrier.shape)
        
    def _decode_lsb(self, carrier: np.ndarray) -> bytes:
        """Least Significant Bit decoding."""
        flat = carrier.flatten()
        data = bytearray()
        for i in range(0, len(flat), 8):
            byte = 0
            for bit in range(8):
                if i + bit >= len(flat):
                    break
                byte |= (flat[i + bit] & 1) << bit
            data.append(byte)
        return bytes(data)

# Add this section at the end of the file
if __name__ == "__main__":
    print("Steganographic Communication System Module")
    
    print("\nAvailable Carrier Types:")
    for carrier in CarrierType:
        print(f"- {carrier.name}: {carrier.value}")
    
    print("\nAvailable Encoding Methods:")
    for method in EncodingMethod:
        print(f"- {method.name}: {method.value}")
    
    # Example usage
    print("\nExample Steganographic System Configuration:")
    steg_specs = SteganographicSpecs(
        weight=0.5,
        volume={"length": 0.1, "width": 0.05, "height": 0.02},
        power_requirements=2.0,
        bandwidth=1.0,  # 1 Mbps
        range=5.0,      # 5 km (short range)
        latency=10.0,   # 10 ms
        encryption_level=10,
        resilience_rating=0.99,
        carrier_type=CarrierType.IMAGE,
        encoding_method=EncodingMethod.LSB,
        max_payload_size=4096  # 4 KB
    )
    
    print(f"Carrier Type: {steg_specs.carrier_type.value}")
    print(f"Encoding Method: {steg_specs.encoding_method.value}")
    print(f"Max Payload Size: {steg_specs.max_payload_size} bytes")
    
    # Simple demonstration
    print("\nSimple Encoding/Decoding Demonstration:")
    system = SteganographicSystem(steg_specs)
    
    # Create a simple carrier (8x8 image)
    carrier = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    print(f"Carrier shape: {carrier.shape}")
    
    # Create sample data
    sample_data = b"UCAV"
    print(f"Original data: {sample_data}")
    
    # Encode
    encoded = system.encode(sample_data, carrier)
    print(f"Encoded carrier shape: {encoded.shape}")
    
    # Decode
    decoded = system.decode(encoded)
    print(f"Decoded data: {decoded[:len(sample_data)]}")

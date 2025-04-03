#!/usr/bin/env python3
"""
Communication systems for UCAV platforms.

This module provides implementations of advanced communication systems
that leverage neuromorphic computing for enhanced capabilities.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.communication.base import (
    CommunicationSystem, CommunicationSpecs, CommunicationType
)
from src.communication.integration import CommunicationIntegrator
from src.communication.optical import (
    LaserOpticalSystem, OpticalSystemSpecs, AtmosphericCondition
)
from src.communication.mesh import (
    MeshNetworkSystem, MeshNetworkSpecs, MeshRoutingProtocol, MeshNodeRole
)
from src.communication.satellite import (
    SatelliteCommunicationSystem, SatelliteSystemSpecs, 
    SatelliteNetwork, SatelliteFrequencyBand
)
from src.communication.cognitive import (
    CognitiveRadioSystem, CognitiveRadioSpecs, SpectrumSensingMode
)
from src.communication.steganographic import (
    SteganographicSystem, 
    SteganographicSpecs,
    CarrierType,
    EncodingMethod
)
from src.communication.terahertz import (
    TerahertzSystem,
    TerahertzSpecs,
    TerahertzBand,
    ModulationScheme
)
from src.communication.self_healing import SelfHealingNetwork
from src.communication.molecular import (
    MolecularCommunicationSystem,
    MolecularSpecs,
    MolecularSignalType,
    DiffusionModel
)
from src.communication.quantum import (
    QuantumCommunicationSystem,
    QuantumSpecs,
    QuantumProtocol
)

__all__ = [
    'CommunicationSystem',
    'CommunicationSpecs',
    'CommunicationType',
    'CommunicationIntegrator',
    'LaserOpticalSystem',
    'OpticalSystemSpecs',
    'AtmosphericCondition',
    'MeshNetworkSystem',
    'MeshNetworkSpecs',
    'MeshRoutingProtocol',
    'MeshNodeRole',
    'SatelliteCommunicationSystem',
    'SatelliteSystemSpecs',
    'SatelliteNetwork',
    'SatelliteFrequencyBand',
    'CognitiveRadioSystem',
    'CognitiveRadioSpecs',
    'SpectrumSensingMode',
    'SteganographicSystem',
    'SteganographicSpecs', 
    'CarrierType',
    'EncodingMethod',
    'TerahertzSystem',
    'TerahertzSpecs',
    'TerahertzBand',
    'ModulationScheme',
    'SelfHealingNetwork',
    'MolecularCommunicationSystem',
    'MolecularSpecs',
    'MolecularSignalType',
    'DiffusionModel',
    'QuantumCommunicationSystem',
    'QuantumSpecs',
    'QuantumProtocol'
]
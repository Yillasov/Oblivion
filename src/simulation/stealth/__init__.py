#!/usr/bin/env python3
"""
Stealth simulation modules for Oblivion SDK.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.stealth.rcs_simulator import (
    RCSSimulator,
    RCSSimulationConfig,
    RCSFrequencyBand
)

from src.simulation.stealth.ir_signature_simulator import (
    IRSignatureSimulator,
    IRSignatureConfig,
    IRBand
)

from src.simulation.stealth.acoustic_simulator import (
    AcousticSignatureSimulator,
    AcousticSimConfig,
    FrequencyRange
)

from src.simulation.stealth.em_signature_simulator import (
    EMSignatureSimulator,
    EMSignatureConfig,
    EMBand
)

__all__ = [
    'RCSSimulator',
    'RCSSimulationConfig',
    'RCSFrequencyBand',
    'IRSignatureSimulator',
    'IRSignatureConfig',
    'IRBand',
    'AcousticSignatureSimulator',
    'AcousticSimConfig',
    'FrequencyRange',
    'EMSignatureSimulator',
    'EMSignatureConfig',
    'EMBand'
]
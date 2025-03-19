"""
Stealth simulation modules for Oblivion SDK.
"""

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
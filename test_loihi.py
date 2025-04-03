#!/usr/bin/env python3
"""
Module description
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.hardware.loihi_driver import LoihiProcessor

# Create and test the processor
processor = LoihiProcessor()
processor.initialize({
    'board_id': 1,
    'chip_id': 0,
    'connection_type': 'local'
})

# Print hardware info
print(processor.get_hardware_info())

# Clean up
processor.shutdown()
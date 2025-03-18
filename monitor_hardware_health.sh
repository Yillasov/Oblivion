#!/bin/bash

# Hardware Health Monitoring Script
# Simple script to monitor neuromorphic hardware health

# Add project root to Python path
export PYTHONPATH=/Users/yessine/Oblivion:$PYTHONPATH

# Default command
COMMAND=${1:-status}

# Run the hardware health CLI
python3 /Users/yessine/Oblivion/src/tools/monitoring/hardware_health_cli.py $COMMAND "$@"
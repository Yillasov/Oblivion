#!/bin/bash

# Simple hardware verification script
# Usage: ./verify_hardware.sh [hardware_address] [algorithm_files...]

# Set default values
HARDWARE_ADDRESS=${1:-""}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we have algorithm files
if [ $# -gt 1 ]; then
    ALGORITHM_FILES="${@:2}"
    echo "Verifying algorithms: $ALGORITHM_FILES"
    
    # Run verification with algorithms
    if [ -n "$HARDWARE_ADDRESS" ]; then
        python -m src.tools.deployment.verify --address "$HARDWARE_ADDRESS" --algorithms $ALGORITHM_FILES
    else
        python -m src.tools.deployment.verify --algorithms $ALGORITHM_FILES
    fi
else
    echo "Running basic hardware verification"
    
    # Run basic verification
    if [ -n "$HARDWARE_ADDRESS" ]; then
        python -m src.tools.deployment.verify --address "$HARDWARE_ADDRESS" --basic-only
    else
        python -m src.tools.deployment.verify --basic-only
    fi
fi

# Check exit code
if [ $? -eq 0 ]; then
    echo "Hardware verification successful"
else
    echo "Hardware verification failed"
    exit 1
fi

echo "Done"